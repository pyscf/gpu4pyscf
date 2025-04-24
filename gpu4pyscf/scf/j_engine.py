# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
J engine using McMurchie-Davidson algorithm
'''

import ctypes
import functools
import math
import numpy as np
from importlib.util import find_spec
has_dpctl = find_spec("dpctl")
if not has_dpctl:
    import cupy as cp
    from gpu4pyscf.lib.cupy_helper import (
        load_library, condense, sandwich_dot, transpose_sum, asarray)
    
    from gpu4pyscf.lib.cupy_helper import load_library, condense, sandwich_dot, transpose_sum
else:
    import dpnp as cp
    from gpu4pyscf.lib.dpnp_helper import (
        load_library, condense, sandwich_dot, transpose_sum, asarray)    
    from gpu4pyscf.lib.dpnp_helper import load_library, condense, sandwich_dot, transpose_sum
    from dpctl._sycl_device_factory import _cached_default_device as get_default_cached_device    
    from dpctl._sycl_queue_manager import get_device_cached_queue
import scipy.linalg
from pyscf import lib
from pyscf import __config__
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import (
    _make_j_engine_pair_locs, RysIntEnvVars, _scale_sp_ctr_coeff, _nearest_power2)

__all__ = [
    'get_j',
]

PTR_BAS_COORD = 7
LMAX = 4
SHM_SIZE = shm_size - 1024
THREADS = 512

libvhf_md = load_library('libgvhf_md')
libvhf_md.MD_build_j.restype = ctypes.c_int
libvhf_md.init_mdj_constant.restype = ctypes.c_int

def get_j(mol, dm, hermi=1, vhfopt=None, verbose=None):
    '''Compute J matrix
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.sorted_mol
    nbas = mol.nbas
    nao, nao_orig = vhfopt.decontract_coeff.shape
    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    assert n_dm == 1
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.decontract_coeff.T)
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
    #_env = vhfopt._mol_gpu[2].get()
    _env = _scale_sp_ctr_coeff(mol)
    libvhf_md.Et_dot_dm(
        dm_xyz.ctypes, dms.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
    dm_xyz = asarray(dm_xyz)
    vj_xyz = cp.zeros_like(dm_xyz)

    pair_loc_on_gpu = asarray(pair_loc)
    rys_envs = RysIntEnvVars(
        mol.natm, mol.nbas,
        vhfopt.rys_envs.atm, vhfopt.rys_envs.bas, vhfopt.rys_envs.env,
        pair_loc_on_gpu.data.ptr,
    )

    err = libvhf_md.init_mdj_constant(ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA/SYCL kernel initialization')
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
                    scheme = _md_j_engine_quartets_scheme(uniq_l_ctr[[i, j, k, l], 0])
                    err = kern(
                        ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        rys_envs, (ctypes.c_int*5)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p),
                        lib.c_null_ptr(),
                        lib.c_null_ptr(),
                        ctypes.c_float(log_cutoff-log_max_dm),
                        ctypes.cast(info.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(workers),
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
        if not has_dpctl:
            cp.cuda.Stream.null.synchronize()
        else:
            dev = get_default_cached_device()
            get_device_cached_queue(dev).wait()
        log.timer_debug1('cuda/sycl kernel', *t2)

    vj_xyz = vj_xyz.get()
    vj = np.zeros_like(dms)
    libvhf_md.jengine_dot_Et(
        vj.ctypes, vj_xyz.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = sandwich_dot(vj, vhfopt.decontract_coeff)
    vj = transpose_sum(vj)
    vj = vj.reshape(dm.shape)
    log.timer('vj', *cput0)
    return vj

class _VHFOpt(jk._VHFOpt):
    def __init__(self, mol, cutoff=1e-13):
        super().__init__(mol, cutoff)
        self.tile = 1

    def build(self, group_size=None, verbose=None):
        orig_mol = self.mol
        self.mol, decontract_coeff = orig_mol.decontract_basis(to_cart=True, aggregate=True)
        jk._VHFOpt.build(self, group_size, verbose)
        jk_coeff = self.coeff
        self.mol = orig_mol
        self.decontract_coeff = jk_coeff.dot(cp.asarray(decontract_coeff))
        return self

def _md_j_engine_quartets_scheme(ls, shm_size=SHM_SIZE):
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    unit = order+1 + (order+1)*(order+2)*(2*order+3)//6
    counts = shm_size // (unit*8)
    threads = THREADS
    if counts >= threads:
        nsq = threads
    else:
        nsq = _nearest_power2(counts)
    ij = _nearest_power2(int(nsq**.5))
    kl = nsq // ij

    # guess tilex and tiley, tiley ~= tilex * (nf3ij / nf3kl)
    tilex = tiley = 1
    if nf3ij >= nf3kl:
        tiley = _nearest_power2(int(nf3ij//nf3kl), return_leq=False)
    else:
        tilex = _nearest_power2(int(nf3kl//nf3ij), return_leq=False)
    cache_size = ij*tilex * (4+nf3ij) + kl*tiley * (4+nf3kl)
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        ij = _nearest_power2(int(nsq**.5))
        kl = nsq // ij
        cache_size = ij*tilex * (4+nf3ij) + kl*tiley * (4+nf3kl)
    gout_stride = threads // nsq

    tilex_max = _nearest_power2((shm_size//8-nsq*unit)//cache_size)
    if tilex_max > 1:
        tilex *= tilex_max
        tiley *= tilex_max
    return ij, kl, gout_stride, tilex, tiley
