#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
J engine using McMurchie-Davidson algorithm
'''

import ctypes
import functools
import math
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import load_library, condense, sandwich_dot, transpose_sum
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import _make_j_engine_pair_locs, RysIntEnvVars

__all__ = [
    'get_j',
]

PTR_BAS_COORD = 7
LMAX = 4
SHM_SIZE = getattr(__config__, 'GPU_SHM_SIZE',
                   int(gpu_specs['sharedMemPerBlockOptin']//9)*8)
THREADS = 256

libvhf_md = load_library('libgvhf_md')
libvhf_md.MD_build_j.restype = ctypes.c_int

def get_j(mol, dm, hermi=1, vhfopt=None, omega=None, verbose=None):
    '''Compute J matrix
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        with mol.with_range_coulomb(omega):
            vhfopt = _VHFOpt(mol).build()
    if omega is None:
        omega = mol.omega

    mol = vhfopt.mol
    nbas = mol.nbas
    nao, nao_orig = vhfopt.coeff.shape
    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    assert n_dm == 1
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')
    if hermi != 1:
        dms = transpose_sum(dms)
    else:
        dms *= 2.

    ao_loc = mol.ao_loc
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_max_dm = dm_cond.max()
    log_cutoff = math.log(vhfopt.direct_scf_tol)

    dms = dms.get()
    pair_loc = _make_j_engine_pair_locs(mol)
    dm_xyz = np.zeros(pair_loc[-1])
    # Must use this modified _env to ensure the consistency with GPU kernel
    # In this _env, normalization coefficients for s and p funcitons are scaled.
    _env = vhfopt._mol_gpu[2].get()
    libvhf_md.Et_dot_dm(
        dm_xyz.ctypes, dms.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
    dm_xyz = cp.asarray(dm_xyz)
    vj_xyz = cp.zeros_like(dm_xyz)

    pair_loc_on_gpu = cp.asarray(pair_loc)
    rys_envs = RysIntEnvVars(
        mol.natm, mol.nbas,
        vhfopt.rys_envs.atm, vhfopt.rys_envs.bas, vhfopt.rys_envs.env,
        pair_loc_on_gpu.data.ptr,
    )

    libvhf_md.init_mdj_constant(ctypes.c_int(SHM_SIZE))
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = len(uniq_l_ctr)
    tile_mappings = {}
    workers = gpu_specs['multiProcessorCount']

    info = cp.empty(2, dtype=np.uint32)

    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            ij_shls = (ish0, ish1, jsh0, jsh1)
            sub_q = vhfopt.q_cond[ish0:ish1,jsh0:jsh1]
            mask = sub_q > log_cutoff# - log_max_dm
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cp.arange(jsh0, jsh1, dtype=np.int32))
            idx = cp.argsort(sub_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    t1 = t2 = log.timer_debug1('q_cond and dm_cond', *cput0)

    timing_collection = {}
    kern_counts = 0
    kern = libvhf_md.MD_build_j

    for i in range(n_groups):
        for j in range(i+1):
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            tile_ij_mapping = tile_mappings[i,j]
            for k in range(i+1):
                for l in range(k+1):
                    if i == k and j < l: continue
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                               l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
                    tile_kl_mapping = tile_mappings[k,l]
                    scheme = _md_j_engine_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err = kern(
                        ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        rys_envs, (ctypes.c_int*3)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p),
                        lib.c_null_ptr(),
                        ctypes.c_float(log_cutoff-log_max_dm),
                        ctypes.cast(info.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(workers), ctypes.c_double(omega),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
                    if err != 0:
                        raise RuntimeError(f'MD_build_j kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        ntasks = tile_ij_mapping.size * tile_kl_mapping.size
                        t1, t1p = log.timer_debug1(f'processing {llll}, tasks ~= {ntasks}', *t1), t1
                        if llll not in timing_collection:
                            timing_collection[llll] = 0
                        timing_collection[llll] += t1[1] - t1p[1]
                        kern_counts += 1

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)
        cp.cuda.Stream.null.synchronize()
        log.timer_debug1('cuda kernel', *t2)

    vj_xyz = vj_xyz.get()
    vj = np.zeros_like(dms)
    libvhf_md.jengine_dot_Et(
        vj.ctypes, vj_xyz.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = sandwich_dot(vj, vhfopt.coeff)
    vj = transpose_sum(vj)
    vj = vj.reshape(dm.shape)
    log.timer('vj', *cput0)
    return vj

class _VHFOpt(jk._VHFOpt):
    def __init__(self, mol, cutoff=1e-13):
        self.mol, self.coeff = mol.decontract_basis(to_cart=True, aggregate=True)
        self.direct_scf_tol = cutoff
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.q_cond = None
        self.tile_q_cond = None
        self.tile = 1

def _md_j_engine_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    unit = order+1 + (order+1)*(order+2)*(2*order+3)//6
    counts = shm_size // (unit*8)
    if counts >= THREADS:
        nsq = THREADS
    else:
        nsq = _nearest_power2(counts)
    ij = _nearest_power2(int(nsq**.5))
    kl = nsq // ij
    tilex, tiley = 2, 4
    cache_size = ij*tilex * (4+nf3ij) + kl*tiley * (4+nf3kl)
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        ij = _nearest_power2(int(nsq**.5))
        kl = nsq // ij
        cache_size = ij*tilex * (4+nf3ij) + kl*tiley * (4+nf3kl)
    gout_stride = THREADS // nsq
    return ij, kl, gout_stride

def _nearest_power2(n):
    t = 0
    while n > 1:
        n >>= 1
        t += 1
    return 2**t
