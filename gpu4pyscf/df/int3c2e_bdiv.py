# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
3-center 2-electron Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, conc_env
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import load_library, contract
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.gto.mole import extract_pgto_params

__all__ = [
    'int3c2e',
]
libgint_rys = load_library('libgint_rys')
libgint_rys.fill_int3c2e.restype = ctypes.c_int
libgint_rys.fill_int3c2e_bdiv.restype = ctypes.c_int
libgint_rys.init_constant.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 45
THREADS = 256

def int3c2e(mol, auxmol):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    int3c2e_opt = Int3c2eOpt(mol, auxmol)
    nao, nao_orig = int3c2e_opt.coeff.shape
    naux = int3c2e_opt.aux_coeff.shape[0]

    out = cp.zeros((nao*nao, naux))
    for ij_shls, eri3c, ao_pair_mapping, aux_mapping in int3c2e_opt.int3c2e_kernel():
        out[ao_pair_mapping] = eri3c
        i, j = divmod(ao_pair_mapping, nao)
        out[j*nao+i] = eri3c
    out = out.reshape(nao, nao, naux)
    aux_coeff = cp.empty_like(int3c2e_opt.aux_coeff)
    aux_coeff[aux_mapping] = int3c2e_opt.aux_coeff
    out = contract('pqr,rk->pqk', out, aux_coeff)
    out = contract('pqk,qj->pjk', out, int3c2e_opt.coeff)
    out = contract('pjk,pi->ijk', out, int3c2e_opt.coeff)
    return out

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol, tile=1)
        self.sorted_mol = mol
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = cp.asarray(coeff)

        self.auxmol = auxmol
        auxmol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(auxmol, tile=1)
        self.sorted_auxmol = auxmol
        self.uniq_l_ctr_aux = uniq_l_ctr
        self.l_ctr_aux_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.aux_coeff = cp.asarray(coeff)

    def int3c2e_kernel(self, cutoff=1e-14, verbose=None):
        mol = self.sorted_mol
        auxmol = self.sorted_auxmol
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets

        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off

        ao_loc_cpu = mol.ao_loc
        aux_loc = auxmol.ao_loc

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(ao_loc_cpu, aux_loc)
        int3c2e_envs = Int3c2eEnvVars(
            mol.natm, mol.nbas, _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc.data.ptr, math.log(cutoff),
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        int3c2e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc)

        uniq_l = uniq_l_ctr[:,0]
        nfcart = (uniq_l + 1) * (uniq_l + 2) // 2
        assert uniq_l.max() <= LMAX
        n_groups = len(uniq_l)
        ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]

        ovlp = estimate_shl_ovlp(mol)
        mask = np.tril(ovlp > cutoff)
        shl_pair_idx = []
        npair_ij = 0
        nbas = mol.nbas
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ijoffset = ish0 * nbas + jsh0
            idx = np.where(mask[ish0:ish1,jsh0:jsh1])[0] + ijoffset
            shl_pair_idx.append(idx)
            nfij = nfcart[i] * nfcart[j]
            npair_ij = max(nfij * idx.size, npair_ij)

        aux_mapping = _create_ao_mapping(self.uniq_l_ctr_aux[:,0],
                                         l_ctr_aux_offsets)
        naux = aux_loc[-1]
        nao = ao_loc_cpu[-1]
        buf = cp.empty((npair_ij, naux))

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e
        cp.cuda.Stream.null.synchronize()
        t1 = log.timer_debug1('initialize int3c2e_kernel', *cput0)
        timing_collection = {}
        kern_counts = 0

        for (i, j), bas_ij_idx in zip(ij_tasks, shl_pair_idx):
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]

            # int3c2e CUDA kernel stores intgrals as [ij_shl,j,i,k,ksh].
            # Uisng ao_pair_mapping to indicate ij addresses in eri3c[k,i,j];
            # aux_mapping to indicate the address k.
            ish, jsh = divmod(bas_ij_idx, nbas)
            nfi = nfcart[i]
            nfj = nfcart[j]
            ij = np.arnge(nfi)*nfj + np.arange(nfj)[:,None]
            ao_pair_mapping = ao_loc_cpu[ish] * nao + ao_loc_cpu[jsh]
            ao_pair_mapping = (ao_pair_mapping[:,None] + ij.ravel()).ravel()

            npair_ij = len(bas_ij_idx)
            bas_ij_idx = cp.asarray(bas_ij_idx, dtype=np.int32)
            nfi = nfcart[i]
            nfj = nfcart[j]
            nfij = nfi * nfj
            eri3c = cp.ndarray((npair_ij*nfij, naux), dtype=np.float64, memptr=buf.data)

            for k, lk in enumerate(self.uniq_l_ctr_aux[:,0]):
                ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
                shls_slice = ish0, ish1, jsh0, jsh1, ksh0, ksh1
                lll = f'({ANGULAR[li]}{ANGULAR[lj]}|{ANGULAR[lk]})'
                scheme = int3c2e_scheme(li, lj, lk)
                log.debug2('int3c2e_scheme for %s: %s', lll, scheme)
                err = kern(
                    ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
                    ctypes.byref(int3c2e_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*6)(*shls_slice), aux_loc.ctypes,
                    ctypes.c_int(npair_ij),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    _atm_cpu.ctypes, ctypes.c_int(mol.natm),
                    _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
                if err != 0:
                    raise RuntimeError(f'fill_int3c2e kernel for {lll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {lll}', *t1), t1
                    if lll not in timing_collection:
                        timing_collection[lll] = 0
                    timing_collection[lll] += t1[1] - t1p[1]
                    kern_counts += 1

            ij_shls = ish0, ish1, jsh0, jsh1
            yield ij_shls, eri3c, ao_pair_mapping.ravel(), aux_mapping

        if log.verbose >= logger.DEBUG1:
            log.timer('int3c2e', *cput0)
            log.debug1('kernel launches %d', kern_counts)
            for lll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', lll, t)

    def int3c2e_bdiv_kernel(self, cutoff=1e-14, verbose=None):
        '''Block-divergent kernel'''
        mol = self.sorted_mol
        auxmol = self.sorted_auxmol
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        l_ctr_aux_offsets = self.l_ctr_aux_offsets

        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        _atm_cpu, _bas_cpu, _env_cpu = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas_cpu[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas_cpu[mol.nbas:,PTR_BAS_COORD] += off

        ao_loc_cpu = mol.ao_loc
        aux_loc = auxmol.ao_loc

        _atm = cp.array(_atm_cpu, dtype=np.int32)
        _bas = cp.array(_bas_cpu, dtype=np.int32)
        _env = cp.array(_env_cpu, dtype=np.float64)
        ao_loc = _conc_locs(ao_loc_cpu, aux_loc)
        int3c2e_envs = Int3c2eEnvVars(
            mol.natm, mol.nbas, _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc.data.ptr, math.log(cutoff),
        )

        nst_lookup = cp.asarray(create_nst_lookup_table(), dtype=np.int32)
        nksh_per_block = 16
        ksh_offsets = []
        for ksh0, ksh1 in zip(l_ctr_aux_offsets[:-1], l_ctr_aux_offsets[1:]):
            ksh_offsets.append(np.arange(ksh0, ksh1, nksh_per_block, dtype=np.int32))
        ksh_offsets.append(l_ctr_aux_offsets[-1])
        ksh_offsets = cp.asarray(np.hstack(ksh_offsets), dtype=np.int32)

        uniq_l = uniq_l_ctr[:,0]
        assert uniq_l.max() <= LMAX
        n_groups = len(uniq_l)
        ij_tasks = [(i, j) for i in range(n_groups) for j in range(i+1)]

        ovlp = estimate_shl_ovlp(mol)
        mask = np.tril(ovlp > cutoff)
        shl_pair_idx = []
        shl_pair_offsets = []
        ao_pair_loc = [0]
        nao_pair = 0
        p0 = p1 = 0
        nbas = mol.nbas
        nao = ao_loc_cpu[-1]
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ijoffset = ish0 * nbas + jsh0
            idx = np.where(mask[ish0:ish1,jsh0:jsh1])[0] + ijoffset
            shl_pair_idx.append(idx)
            nfi = (li + 1) * (li + 2) // 2
            nfj = (lj + 1) * (lj + 2) // 2
            nfij = nfi * nfj
            nao_pair += nfij * idx.size

            p0, p1 = p1, p1 + idx.size
            nsp_per_block = 256 // _nearest_power2(li+lj+1)
            shl_pair_offsets.append(np.arange(p0, p1, nsp_per_block, dtype=np.int32))
            ao_pair_loc.append(nao_pair)

            # int3c2e CUDA kernel stores intgrals as [ij_shl,j,i,k,ksh].
            # Uisng ao_pair_mapping to indicate ij addresses in eri3c[k,i,j];
            # aux_mapping to indicate the address k.
            ish, jsh = divmod(idx, nbas)
            ij = np.arnge(nfi)*nfj + np.arange(nfj)[:,None]
            ao_pair_mapping = ao_loc_cpu[ish] * nao + ao_loc_cpu[jsh]
            ao_pair_mapping = (ao_pair_mapping[:,None] + ij.ravel()).ravel()

        shl_pair_idx = cp.asarray(np.hstack(shl_pair_idx), dtype=np.int32)
        shl_pair_offsets.append([p1])
        shl_pair_offsets = cp.asarray(np.hstack(shl_pair_offsets), dtype=np.int32)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        nbatches_ksh = len(ksh_offsets) - 1

        init_constant(mol)
        kern = libgint_rys.fill_int3c2e_bdiv
        cp.cuda.Stream.null.synchronize()
        t1 = log.timer_debug1('initialize int3c2e_bdiv_kernel', *cput0)

        naux = aux_loc[-1]
        eri3c = cp.empty((nao_pair, naux))
        err = kern(
            ctypes.cast(eri3c.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(SHM_SIZE), ctypes.c_int(naux),
            ctypes.c_int(nbatches_shl_pair), ctypes.c_int(nbatches_ksh),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(nst_lookup.data.ptr, ctypes.c_void_p),
            _atm_cpu.ctypes, ctypes.c_int(mol.natm),
            _bas_cpu.ctypes, ctypes.c_int(mol.nbas), _env_cpu.ctypes)
        if err != 0:
            raise RuntimeError('fill_int3c2e_bdiv kernel failed')
        log.timer_debug1('processing int3c2e_bdiv_kernel', *t1)

        aux_mapping = _create_ao_mapping(self.uniq_l_ctr_aux[:,0],
                                         l_ctr_aux_offsets)
        return eri3c, ao_pair_mapping, aux_mapping

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    err = libgint_rys.init_constant(
        g_idx.ctypes, offsets.ctypes, mol._env.ctypes, ctypes.c_int(mol._env.size),
        ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

def int3c2e_scheme(li, lj, lk, shm_size=SHM_SIZE):
    order = li + lj + lk
    nroots = (order//2 + 1) * 2

    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 6
    nst_max = shm_size//(unit*8)
    nst_max = _nearest_power2(nst_max)

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    gout_size = nfi * nfj * nfk
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    gout_stride = min(gout_stride, 64)

    nst_per_block = min(nst_max, THREADS // gout_stride)
    gout_stride = THREADS // nst_per_block
    return nst_per_block, gout_stride

def create_nst_lookup_table():
    nst_lookup = np.empty([L_AUX_MAX+1]*3, dtype=np.int32)
    for lk in range(L_AUX_MAX+1):
        for li in range(lk+1):
            for lj in range(li+1):
                nst_lookup[lk,li,lj] = int3c2e_scheme(li, lj, lk)[0]
    idx = np.arange(L_AUX_MAX+1)
    z, y, x = np.sort(np.meshgrid(idx, idx, idx), axis=0)
    nst_lookup = nst_lookup[x, y, z]
    return nst_lookup[:,:LMAX+1,:LMAX+1]

def estimate_shl_ovlp(mol):
    # consider only the most diffused component of a basis
    exps, cs = extract_pgto_params(mol, 'diffused')
    ls = mol._bas[:,ANG_OF]
    bas_coords = cp.asarray(mol.atom_coords()[mol._bas[:,ATOM_OF]])

    norm = cs * ((2*ls+1)/(4*np.pi))**.5
    aij = exps[:,None] + exps
    fi = exps[:,None] / aij
    fj = exps[None,:] / aij
    theta = exps[:,None] * fj

    rirj = bas_coords[:,None,:] - bas_coords
    dr = np.linalg.norm(rirj, axis=2)
    dri = fj * dr
    drj = fi * dr
    li = ls[:,None]
    lj = ls[None,:]
    fac_dri = (li * .5/aij + dri**2) ** (li*.5)
    fac_drj = (lj * .5/aij + drj**2) ** (lj*.5)
    fac_norm = norm[:,None]*norm * (np.pi/aij)**1.5
    ovlp = fac_norm * np.exp(-theta*dr**2) * fac_dri * fac_drj
    return ovlp

def _create_ao_mapping(uniq_l, l_ctr_offsets):
    '''AO mapping is an array that indicates the index for each element created
    in CUDA int3c2e kernel: int3c2e_on_cpu[ao_mapping] = int3c2e_on_gpu'''
    ao_mapping = []
    koff = 0
    for k, lk in enumerate(uniq_l):
        ksh0, ksh1 = l_ctr_offsets[k], l_ctr_offsets[k+1]
        nksh = ksh1 - ksh0
        nfk = (lk + 1) * (lk + 2) // 2
        idx = koff + np.arange(nksh)[:,None] + np.arange(nfk)*nksh
        ao_mapping.append(idx.ravel())
        koff += nfk * nksh
    return np.hstack(ao_mapping)
