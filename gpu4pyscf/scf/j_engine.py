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
import math
import numpy as np
import cupy as cp
from collections import Counter
from pyscf import lib
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (
    load_library, condense, dist_matrix, transpose_sum, hermi_triu, asarray)
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import RysIntEnvVars, _scale_sp_ctr_coeff, _nearest_power2
from gpu4pyscf.gto.mole import group_basis

__all__ = [
    'get_j',
]

PTR_BAS_COORD = 7
LMAX = 4
SHM_SIZE = shm_size - 1024
THREADS = 256

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

    nao_orig = mol.nao
    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    if hermi != 1:
        dms = transpose_sum(dms)
        dms *= .5

    vj = vhfopt.get_j(dms, log)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = vhfopt.apply_coeff_CT_mat_C(vj)
    vj = vj.reshape(dm.shape)
    log.timer('vj', *cput0)
    return vj

def _to_primitive_bas(sorted_mol):
    # Note, sorted_mol.decontract_basis cannot be used here as that function
    # assumes the basis sets are not grouped by atoms
    prim_mol = sorted_mol.copy()
    prim_mol.cart = True
    repeats = sorted_mol._bas[:,NPRIM_OF]
    prim_mol._bas = np.repeat(sorted_mol._bas, repeats, axis=0)

    address_inc = [np.arange(i) for i in range(repeats.max()+1)]
    address_inc = np.hstack([address_inc[i] for i in repeats])
    prim_mol._bas[:,PTR_EXP] += address_inc
    prim_mol._bas[:,PTR_COEFF] += address_inc
    prim_mol._bas[:,NPRIM_OF] = 1

    p2c_mapping = np.repeat(np.arange(sorted_mol.nbas), repeats)
    return prim_mol, np.asarray(p2c_mapping, dtype=np.int32)

# TODO: This approximate q_cond underestimates some integrals for l>=3
def _estimate_q_cond(prim_mol):
    '''An approxiamte q_cond based on the overlap of primitive GTO shells'''
    ls = cp.asarray(prim_mol._bas[:,ANG_OF])
    es = cp.asarray(prim_mol._env[prim_mol._bas[:,PTR_EXP]])
    cs = abs(cp.asarray(prim_mol._env[prim_mol._bas[:,PTR_COEFF]]))
    log_cs = cp.log(cs)
    bas_coords = cp.asarray(prim_mol.atom_coords()[prim_mol._bas[:,ATOM_OF]])
    li = ls[:,None]
    lj = ls
    aij = es[:,None] + es
    fi = es[:,None] / aij
    fj = es[None,:] / aij
    theta = es[:,None] * fj
    dr = dist_matrix(bas_coords, bas_coords)
    dri = fj * dr
    drj = fi * dr
    fac_dri = (li*.5) * cp.log(li * .5/aij + dri**2 + 1e-9)
    fac_drj = (lj*.5) * cp.log(lj * .5/aij + drj**2 + 1e-9)
    fac_norm = log_cs[:,None]+log_cs + 1.5 * cp.log(np.pi/aij)
    q_cond = fac_norm - theta*dr**2 + fac_dri + fac_drj
    q_cond += .25 * cp.log(2./np.pi * aij)
    return cp.asarray(q_cond, dtype=np.float32)

class _VHFOpt(jk._VHFOpt):
    def __init__(self, mol, cutoff=1e-13):
        super().__init__(mol, cutoff)
        self.tile = 1

    def build(self, group_size=None, verbose=None):
        mol = self.mol
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        assert group_size is None
        sorted_mol, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = \
                group_basis(mol, 1, group_size, sparse_coeff=True)
        self.sorted_mol = sorted_mol
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = l_ctr_pad_counts
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        # very high angular momentum basis are processed on CPU
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        prim_mol, self.prim_to_ctr_mapping = _to_primitive_bas(sorted_mol)
        self.prim_mol = prim_mol

        #q_cond = _estimate_q_cond(prim_mol).get()
        nbas = prim_mol.nbas
        ao_loc = prim_mol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = prim_mol._add_suffix('int2e')
        _vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            prim_mol._atm.ctypes, ctypes.c_int(prim_mol.natm),
            prim_mol._bas.ctypes, ctypes.c_int(prim_mol.nbas),
            prim_mol._env.ctypes)
        q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        self.q_cond_cpu = q_cond

        assert self.tile == 1
        self._tile_q_cond_cpu = q_cond

        log.timer('Initialize q_cond', *cput0)
        return self

    def get_j(self, dms, verbose):
        log = logger.new_logger(self.mol, verbose)
        sorted_mol = self.sorted_mol
        prim_mol = self.prim_mol
        # Small arrays pair_mappings, pair_loc etc may the occupy freed memory
        # created in dms(). Preallocate workspace for these arrays
        workspace = cp.empty(prim_mol.nbas**2*2)
        workspace = None # noqa: F841
        if callable(dms):
            dms = dms()
        p2c_mapping = cp.asarray(self.prim_to_ctr_mapping)
        ao_loc = sorted_mol.ao_loc
        n_dm, nao = dms.shape[:2]
        assert dms.ndim == 3 and nao == ao_loc[-1]
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(self.direct_scf_tol)
        q_cutoff = log_cutoff - log_max_dm
        dm_cond = dm_cond[p2c_mapping[:,None],p2c_mapping]

        l_counts = np.bincount(prim_mol._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
        l_symb = lib.param.ANGULAR
        q_cond = self.q_cond
        pair_mappings = _make_pair_qd_cond(prim_mol, l_ctr_bas_loc, q_cond, dm_cond, q_cutoff)
        dm_cond = q_cond = None

        pair_lst = []
        task_offsets = {} # the pair_loc offsets for each ij pair
        p0 = p1 = 0
        for i in range(n_groups):
            for j in range(i+1):
                pair_ij_mapping = pair_mappings[i,j][0]
                pair_lst.append(pair_ij_mapping)
                p0, p1 = p1, p1 + pair_ij_mapping.size
                task_offsets[i,j] = p0
        pair_mapping_size = p1
        pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)

        ls = cp.asarray(prim_mol._bas[:,ANG_OF], dtype=np.int32)
        ll = ls[:,None] + ls
        ll = ll.ravel()[pair_lst] # drops the pairs that do not contribute to integrals
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        pair_loc = cp.cumsum(cp.append(np.int32(0), xyz_size.ravel()), dtype=np.int32)
        xyz_size = ls = ll = None

        pair_lst = np.asarray(pair_lst.get(), dtype=np.int32)
        pair_loc = pair_loc.get()
        dm_xyz_size = pair_loc[-1]
        log.debug1('dm_xyz_size = %s, nao = %s, pair_mapping_size = %s',
                   dm_xyz_size, nao, pair_mapping_size)
        dms = dms.get()
        dm_xyz = np.zeros((n_dm, dm_xyz_size))
        # Must use this modified _env to ensure the consistency with GPU kernel
        # In this _env, normalization coefficients for s and p funcitons are scaled.
        _env = _scale_sp_ctr_coeff(prim_mol)
        libvhf_md.Et_dot_dm(
            dm_xyz.ctypes, dms.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
            self.prim_to_ctr_mapping.ctypes,
            ctypes.c_int(prim_mol.nbas), ctypes.c_int(sorted_mol.nbas),
            prim_mol._bas.ctypes, _env.ctypes)

        tasks = []
        for i in range(n_groups):
            for j in range(i+1):
                for k in range(i+1):
                    for l in range(k+1):
                        if i == k and j < l: continue
                        tasks.append((i,j,k,l))
        schemes = {t: _md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}

        def proc(dm_xyz):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self.mol, verbose)
            t0 = log.init_timer()
            dm_xyz = asarray(dm_xyz) # transfer to current device
            vj_xyz = cp.zeros_like(dm_xyz)
            _atm_gpu = cp.asarray(prim_mol._atm)
            _bas_gpu = cp.asarray(prim_mol._bas)
            _env_gpu = cp.asarray(_env)
            rys_envs = RysIntEnvVars(
                prim_mol.natm, prim_mol.nbas,
                _atm_gpu.data.ptr, _bas_gpu.data.ptr, _env_gpu.data.ptr, 0,
            )

            err = libvhf_md.init_mdj_constant(ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError('CUDA kernel initialization')

            _pair_mappings = pair_mappings
            if num_devices > 1:
                # Ensure the precomputation copied to each device
                _pair_mappings = {}
                for task, (pair_idx, _, qd) in pair_mappings.items():
                    qd = [cp.asarray(x) for x in qd]
                    addrs = [ctypes.cast(x.data.ptr, ctypes.c_void_p) for x in qd]
                    _pair_mappings[task] = (cp.asarray(pair_idx), addrs, qd)
            pair_loc_on_gpu = asarray(pair_loc)
            q_cond = cp.asarray(self.q_cond)
            t1 = log.timer_debug1(f'q_cond on Device {device_id}', *t0)

            timing_collection = {}
            kern_counts = 0
            kern = libvhf_md.MD_build_j

            while tasks:
                try:
                    task = tasks.pop()
                except IndexError:
                    break

                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, qd_ij_addrs = _pair_mappings[i,j][:2]
                pair_kl_mapping, qd_kl_addrs = _pair_mappings[k,l][:2]
                if len(pair_ij_mapping) == 0 or len(pair_kl_mapping) == 0:
                    continue
                pair_ij_loc = pair_loc_on_gpu[task_offsets[i,j]:]
                pair_kl_loc = pair_loc_on_gpu[task_offsets[k,l]:]
                scheme = schemes[task]
                err = kern(
                    ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
                    rys_envs, (ctypes.c_int*6)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(pair_ij_mapping.size),
                    ctypes.c_int(pair_kl_mapping.size),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                    (ctypes.c_void_p*6)(*qd_ij_addrs),
                    (ctypes.c_void_p*6)(*qd_kl_addrs),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    prim_mol._atm.ctypes, ctypes.c_int(prim_mol.natm),
                    prim_mol._bas.ctypes, ctypes.c_int(prim_mol.nbas), _env.ctypes)

                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'MD_build_j kernel for {llll} failed')

                if log.verbose >= logger.DEBUG1:
                    ntasks = pair_ij_mapping.size * pair_kl_mapping.size
                    t1, t1p = log.timer_debug1(f'processing {llll}, scheme={scheme} tasks ~= {ntasks}', *t1), t1
                    if llll not in timing_collection:
                        timing_collection[llll] = 0
                    timing_collection[llll] += t1[1] - t1p[1]
                    kern_counts += 1
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz,), non_blocking=True)
        kern_counts = 0
        timing_collection = Counter()
        vj_dist = []
        for vj, counts, t_counter in results:
            kern_counts += counts
            timing_collection += t_counter
            vj_dist.append(vj)

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        vj_xyz = multi_gpu.array_reduce(vj_dist, inplace=True)
        vj_xyz = vj_xyz.get()
        vj = np.zeros_like(dms)
        libvhf_md.jengine_dot_Et(
            vj.ctypes, vj_xyz.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
            self.prim_to_ctr_mapping.ctypes,
            ctypes.c_int(prim_mol.nbas), ctypes.c_int(sorted_mol.nbas),
            prim_mol._bas.ctypes, _env.ctypes)
        vj = transpose_sum(asarray(vj))
        vj *= 2.

        h_shls = self.h_shls
        if h_shls:
            mol = self.sorted_mol
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = 'ji->s2kl'
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms, 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            vj1 = asarray(vs_h)
            vj += hermi_triu(vj1)
        return vj

def _make_pair_qd_cond(mol, l_ctr_bas_loc, q_cond, dm_cond, cutoff):
    n_groups = len(l_ctr_bas_loc) - 1
    pair_mappings = {}
    nbas = mol.nbas
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            sub_q = q_cond[ish0:ish1,jsh0:jsh1]
            mask = sub_q > cutoff
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cp.arange(jsh0, jsh1, dtype=np.int32))
            sub_q = sub_q[mask]
            idx = cp.argsort(sub_q)[::-1]

            # qd_tile_max is the product of q_cond and dm_cond within each batch
            sub_q += dm_cond[ish0:ish1,jsh0:jsh1][mask]
            qd_tile_max = cp.zeros((sub_q.size+31) & 0xffffffe0, # 32-element aligned
                                   dtype=np.float32)
            qd_tile_max[:sub_q.size] = sub_q[idx]
            qd_tile2_max = qd_tile_max.reshape(-1,2).max(axis=1)
            qd_tile4_max = qd_tile2_max.reshape(-1,2).max(axis=1)
            qd_tile8_max = qd_tile4_max.reshape(-1,2).max(axis=1)
            qd_tile16_max = qd_tile8_max.reshape(-1,2).max(axis=1)
            qd_tile32_max = qd_tile16_max.reshape(-1,2).max(axis=1)
            qd_tile_addrs = (ctypes.cast(qd_tile_max.data.ptr, ctypes.c_void_p),
                             ctypes.cast(qd_tile2_max.data.ptr, ctypes.c_void_p),
                             ctypes.cast(qd_tile4_max.data.ptr, ctypes.c_void_p),
                             ctypes.cast(qd_tile8_max.data.ptr, ctypes.c_void_p),
                             ctypes.cast(qd_tile16_max.data.ptr, ctypes.c_void_p),
                             ctypes.cast(qd_tile32_max.data.ptr, ctypes.c_void_p))
            qd_batch_max = (qd_tile_max, qd_tile2_max, qd_tile4_max, qd_tile8_max,
                            qd_tile16_max, qd_tile32_max)
            pair_mappings[i,j] = (t_ij[mask][idx], qd_tile_addrs, qd_batch_max)
    return pair_mappings

VJ_IJ_REGISTERS = 9
def _md_j_engine_quartets_scheme(ls, shm_size=SHM_SIZE, n_dm=1):
    if n_dm > 1:
        n_dm = 4

    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt_size = (order+1)*(order+2)*(2*order+3)//6
    gout_stride_min = _nearest_power2(
        int((nf3ij+VJ_IJ_REGISTERS-1) / VJ_IJ_REGISTERS), False)

    unit = order+1 + Rt_size
    #counts = shm_size // ((unit+gout_stride_min-1)//gout_stride_min*8)
    counts = shm_size // (unit*8)
    threads = THREADS
    if counts * gout_stride_min >= threads:
        nsq = threads // gout_stride_min
    else:
        nsq = _nearest_power2(counts)
    kl = _nearest_power2(int(nsq**.5))
    ij = nsq // kl

    tilex = 48
    # Guess number of batches for kl indices
    tiley = (shm_size//8 - nsq*unit - (ij*4+ij*nf3ij*n_dm)) // (kl*4+kl*nf3kl*n_dm)
    tiley = min(tilex, tiley)
    tiley = tiley // 4 * 4
    if tiley < 4:
        tiley = 4
    if li == lk and lj == ll:
        tilex = tiley
    cache_size = ij * 4 + kl*tiley * 4 + ij*nf3ij*n_dm + kl*nf3kl*tiley*n_dm
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        assert nsq >= 1
        kl = _nearest_power2(int(nsq**.5))
        ij = nsq // kl
        cache_size = ij * 4 + kl*tiley * 4 + ij*nf3ij*n_dm + kl*nf3kl*tiley*n_dm
    gout_stride = threads // nsq
    buflen = nsq*unit+cache_size
    return ij, kl, gout_stride, tilex, tiley, buflen
