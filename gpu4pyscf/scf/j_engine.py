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
from pyscf import lib
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF, PTR_COORD
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (
    load_library, condense, dist_matrix, transpose_sum, hermi_triu, asarray)
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import _nearest_power2
from gpu4pyscf.gto.mole import SortedGTO, _scale_sp_ctr_coeff

__all__ = [
    'get_j',
]

PTR_BAS_COORD = 7
LMAX = 4
SHM_SIZE = shm_size - 1024
THREADS = 256

libvhf_md = load_library('libgvhf_md')
libvhf_md.MD_build_j.restype = ctypes.c_int
libvhf_md.init_mdj_constant(ctypes.c_int(SHM_SIZE))

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

class _VHFOpt(jk._VHFOpt):
    def build(self, group_size=None, verbose=None):
        log = logger.new_logger(self.mol, verbose)
        cput0 = log.init_timer()
        mol = self.sorted_mol = SortedGTO.from_mol(
            self.mol, decontract=True, diffuse_cutoff=1e200)

        # very high angular momentum basis are processed on CPU
        uniq_l_ctr = mol.uniq_l_ctr
        l_ctr_counts = mol.l_ctr_counts
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(
            mol, self.rys_envs, self.direct_scf_tol)
        log.timer('Initialize q_cond', *cput0)
        return self

    def get_j(self, dms, verbose):
        log = logger.new_logger(self.mol, verbose)
        mol = self.sorted_mol
        if callable(dms):
            dms = dms()
        ao_loc = mol.ao_loc
        n_dm, nao = dms.shape[:2]
        assert dms.ndim == 3 and nao == ao_loc[-1]
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_cutoff = math.log(self.direct_scf_tol)

        l_counts = np.bincount(mol._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
        l_symb = lib.param.ANGULAR
        pair_mappings = _make_pair_qd_cond(mol, self.bas_pair_cache, dm_cond)
        dm_cond = None

        pair_lst = []
        task_offsets = {} # the pair_loc offsets for each ij pair
        p0 = p1 = 0
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        for key in ij_tasks:
            pair_ij_mapping = pair_mappings[key][0]
            pair_lst.append(pair_ij_mapping)
            p0, p1 = p1, p1 + pair_ij_mapping.size
            task_offsets[key] = p0
        pair_mapping_size = p1
        pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)

        ls = cp.asarray(mol._bas[:,ANG_OF], dtype=np.int32)
        ll = ls[:,None] + ls
        ll = ll.ravel()[pair_lst] # drops the pairs that do not contribute to integrals
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        pair_loc_gpu = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32), xyz_size.ravel()), dtype=np.int32)
        xyz_size = ls = ll = None

        pair_lst = np.asarray(pair_lst.get(), dtype=np.int32)
        pair_loc = pair_loc_gpu.get()
        dm_xyz_size = pair_loc[-1]
        log.debug1('dm_xyz_size = %s, nao = %s, pair_mapping_size = %s',
                   dm_xyz_size, nao, pair_mapping_size)
        dms = dms.get()
        dm_xyz = np.zeros((n_dm, dm_xyz_size))
        # Must use this modified _env to ensure the consistency with GPU kernel
        # In this _env, normalization coefficients for s and p funcitons are scaled.
        _env = _scale_sp_ctr_coeff(mol)
        libvhf_md.Et_dot_dm(
            dm_xyz.ctypes, dms.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
            ctypes.c_int(mol.nbas),
            mol._bas.ctypes, _env.ctypes)

        tasks = []
        for i in range(n_groups):
            for j in range(i+1):
                for k in range(i+1):
                    for l in range(k+1):
                        if i == k and j < l: continue
                        tasks.append((i,j,k,l))
        schemes = {t: _md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}
        tasks = iter(tasks)

        def proc(dm_xyz):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self.mol, verbose)
            t1 = log.init_timer()
            dm_xyz = asarray(dm_xyz) # transfer to current device
            vj_xyz = cp.zeros_like(dm_xyz)

            _pair_mappings = pair_mappings
            if device_id > 0: # Ensure the precomputation avail on each device
                _pair_mappings = {k: [cp.asarray(x) for x in v]
                                  for k, v in pair_mappings.items()}
            _pair_loc_gpu = cp.asarray(pair_loc_gpu)

            timing_collection = jk._TimingCollector(log.timer_debug1)
            kern_counts = 0
            kern = libvhf_md.MD_build_j
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, qd_ij = _pair_mappings[i,j]
                pair_kl_mapping, q_cond_kl, qd_kl = _pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                pair_ij_loc = _pair_loc_gpu[task_offsets[i,j]:]
                pair_kl_loc = _pair_loc_gpu[task_offsets[k,l]:]
                scheme = schemes[task]
                err = kern(
                    ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
                    ctypes.byref(rys_envs), (ctypes.c_int*6)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)

                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'MD_build_j kernel for {llll} failed')
                kern_counts += 1

                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll}, scheme={scheme} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz,), non_blocking=True)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            jk._TimingCollector.summary(log.debug1, (x[2] for x in results))

        vj_xyz = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj_xyz = vj_xyz.get()

        h_shls = self.h_shls
        if h_shls:
            vj = np.zeros_like(dms)
        else:
            vj, dms = dms, None
            vj[:] = 0.
        libvhf_md.jengine_dot_Et(
            vj.ctypes, vj_xyz.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
            ctypes.c_int(mol.nbas),
            mol._bas.ctypes, _env.ctypes)
        vj = transpose_sum(asarray(vj))
        vj *= 2.

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

def _cache_q_cond_and_non0pairs(mol, rys_envs, precision):
    assert all(mol.uniq_l_ctr[:,1] == 1)
    bas_pair_cache = jk._cache_q_cond_and_non0pairs(mol, rys_envs, precision,
                                                    tile=1)
    uniq_l = mol.uniq_l_ctr[:,0]
    lmax = min(uniq_l.max(), LMAX)
    # The pairs for MD-J must include all pairs between angular moments.
    # bas_pair_cache are l_ctr-indexed. The missing (li,lj) entries are filled
    # zero-length orbital-pairs.
    padding = (cp.zeros(0, dtype=np.uint32), cp.zeros(0, dtype=np.float32))
    out = {(li, lj): padding for li in range(lmax+1) for lj in range(li+1)}
    for i, j in bas_pair_cache:
        li = uniq_l[i]
        lj = uniq_l[j]
        pair_ij, q_cond, s_cond = bas_pair_cache[i, j]
        idx = cp.argsort(q_cond)[::-1]
        out[li, lj] = pair_ij[idx], q_cond[idx]
    return out

def _make_pair_qd_cond(mol, bas_pair_cache, dm_cond):
    dm_cond = dm_cond.ravel()
    out = {}
    for k, (pair, q_cond) in bas_pair_cache.items():
        if len(pair) != 0:
            sub_q = q_cond + dm_cond[pair]
            qd_batch_max = _make_tile_max_hierarchy(sub_q)
        else:
            qd_batch_max = q_cond
        out[k] = pair, q_cond, qd_batch_max
    return out

def _make_tile_max_hierarchy(sub_q):
    size_aligned = (sub_q.size+31) & 0xffffffe0 # 32-element aligned
    offset2 = size_aligned
    offset4 = offset2 + size_aligned // 2
    offset8 = offset4 + size_aligned // 4
    offset16 = offset8 + size_aligned // 8
    offset32 = offset16 + size_aligned // 16
    tile_max = cp.full(offset32+size_aligned//32, -700., dtype=np.float32)
    tile_max[:sub_q.size] = sub_q.ravel()
    tile1_max = tile_max[:offset2]
    tile2_max = tile1_max.reshape(-1,2).max(axis=1, out=tile_max[offset2:offset4])
    tile4_max = tile2_max.reshape(-1,2).max(axis=1, out=tile_max[offset4:offset8])
    tile8_max = tile4_max.reshape(-1,2).max(axis=1, out=tile_max[offset8:offset16])
    tile16_max = tile8_max.reshape(-1,2).max(axis=1, out=tile_max[offset16:offset32])
    tile32_max = tile16_max.reshape(-1,2).max(axis=1, out=tile_max[offset32:]) # noqa
    return tile_max

VJ_IJ_REGISTERS = 11
MULTI_VJ_IJ_REGISTERS = 8
RT_TMP_REGISTERS = 31
RT2_IDX_CACHE_SIZE = 35 * 56
def _md_j_engine_quartets_scheme(ls, shm_size=SHM_SIZE, n_dm=1):
    vj_ij_registers = VJ_IJ_REGISTERS
    if n_dm > 1:
        vj_ij_registers = MULTI_VJ_IJ_REGISTERS
        n_dm = 4

    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt_size = (order+1)*(order+2)*(order+3)//6
    gout_stride_min = max(
        _nearest_power2(int((nf3ij+vj_ij_registers-1) / vj_ij_registers), False),
        _nearest_power2(int((Rt_size+RT_TMP_REGISTERS-1) / RT_TMP_REGISTERS), False))

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

    cache_Rt2_idx = nf3ij * nf3kl <= RT2_IDX_CACHE_SIZE
    if cache_Rt2_idx:
        shm_size -= nf3ij * nf3kl * 2

    tilex = 32
    # Guess number of batches for kl indices
    tiley = (shm_size//8 - nsq*unit - (ij*4+ij*nf3ij*n_dm)) // (kl*4+kl*nf3kl*n_dm)
    tiley = min(tilex, tiley)
    tiley = tiley // 4 * 4
    if tiley < 4:
        tiley = 4
    if li == lk and lj == ll:
        tilex = tiley
    # vj_cache reuses the space which was used by ij*4+ij*nf3ij*n_dm+nsq*unit
    vj_cache_reserve = n_dm * threads - nsq*unit
    cache_size = max(ij*4 + ij*nf3ij*n_dm, vj_cache_reserve) + kl*tiley*4 + kl*nf3kl*tiley*n_dm
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        assert nsq >= 1
        kl = _nearest_power2(int(nsq**.5))
        ij = nsq // kl
        cache_size = max(ij*4 + ij*nf3ij*n_dm, vj_cache_reserve) + kl*tiley*4 + kl*nf3kl*tiley*n_dm
    gout_stride = threads // nsq
    buflen = (nsq * unit + cache_size) * 8
    if cache_Rt2_idx:
        buflen += nf3ij * nf3kl * 2
    return ij, kl, gout_stride, tilex, tiley, buflen
