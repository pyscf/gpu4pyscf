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

import ctypes
import numpy as np
import cupy as cp
from pyscf.gto import ATOM_OF, PTR_COORD
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib.cupy_helper import contract, asarray, sandwich_dot
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import libpbc, PBCIntEnvVars
from gpu4pyscf.pbc.df.int3c2e import (
    _estimate_shl_pairs_per_block, fill_triu_bvk_conj, LMAX, L_AUX_MAX, THREADS
)

__all__ = [
    'int1e_ovlp',
    'int1e_kin',
    'int1e_ipovlp',
    'int1e_ipkin',
]

libpbc.PBCint1e_ovlp.restype = ctypes.c_int
libpbc.PBCint1e_kin.restype = ctypes.c_int
libpbc.PBCint1e_ipovlp.restype = ctypes.c_int
libpbc.PBCint1e_ipkin.restype = ctypes.c_int

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ovlp', 1, 1, (0, 0))
    return out

def int1e_kin(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_kin', 1, 1, (2, 0))
    return out

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ipovlp', 0, 3, (1, 0))
    return out

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None, opt=None):
    opt = _check_opt(cell, kpts, bvk_kmesh, opt)
    out = opt.intor('PBCint1e_ipkin', 0, 3, (3, 0))
    return out

def _check_opt(cell, kpts, bvk_kmesh, opt):
    if opt is None:
        opt = _Int1eOpt(cell, kpts, bvk_kmesh)
    else:
        assert kpts is opt.kpts
    return opt

class _Int1eOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None):
        self.cell = cell
        sorted_cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        uniq_l = uniq_l_ctr[:,0]
        lmax = uniq_l.max()
        assert lmax <= LMAX
        self.sorted_cell = sorted_cell
        self.coeff = coeff
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts

        if bvk_kmesh is None:
            if kpts is None:
                bvk_kmesh = np.ones(3, dtype=np.int32)
            else:
                bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1, 3))
        self.kpts = kpts
        self.bvk_kmesh = bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        if bvk_ncells == 1:
            bvkcell = sorted_cell
        else:
            bvkcell = pbctools.super_cell(sorted_cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        Ls = asarray(bvkcell.get_lattice_Ls(rcut=cell.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)

        _atm = cp.array(bvkcell._atm, dtype=np.int32)
        _bas = cp.array(bvkcell._bas, dtype=np.int32)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell), dtype=np.float64)
        ao_loc = bvkcell.ao_loc_nr(cart=True)
        ao_loc_gpu = cp.array(ao_loc, dtype=np.int32)
        int1e_envs = PBCIntEnvVars(
            sorted_cell.natm, sorted_cell.nbas, bvk_ncells, nimgs,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
            ao_loc_gpu.data.ptr, Ls.data.ptr,
        )
        int1e_envs._env_ref_holder = (_atm, _bas, _env, ao_loc_gpu, Ls)
        self.int1e_envs = int1e_envs

    def generate_shl_pairs(self, hermi=1):
        sorted_cell = self.sorted_cell
        l_ctr_offsets = np.append(0, np.cumsum(self.l_ctr_counts))
        uniq_l = self.uniq_l_ctr[:,0]
        bas_ij_idx = [] # The effective shell pair = ish*nbas+jsh
        shl_pair_offsets = [] # the bas_ij_idx offset for each blockIdx.x
        sp0 = sp1 = 0
        nbas = sorted_cell.nbas
        groups = len(uniq_l)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        bvk_ncells = np.prod(self.bvk_kmesh)
        img = cp.arange(bvk_ncells, dtype=np.int32)
        img_offsets = img * nbas
        for i, j in ij_tasks:
            li = uniq_l[i]
            lj = uniq_l[j]
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            ijsh = ish[:,None] * (nbas*bvk_ncells) + jsh
            if hermi and i == j:
                ijsh = ijsh[cp.tril_indices(ish1-ish0)]
            else:
                ijsh = ijsh.ravel()
            idx = (img_offsets[:,None] + ijsh).ravel()
            nshl_pair = len(idx)
            bas_ij_idx.append(idx)
            sp0, sp1 = sp1, sp1 + nshl_pair
            nsp_per_block = _estimate_shl_pairs_per_block(li, lj, nshl_pair)
            shl_pair_offsets.append(np.arange(sp0, sp1, nsp_per_block, dtype=np.int32))

        shl_pair_offsets.append(np.array([sp1], dtype=np.int32))
        shl_pair_offsets = cp.array(np.hstack(shl_pair_offsets), dtype=np.int32)
        bas_ij_idx = cp.array(cp.hstack(bas_ij_idx), dtype=np.int32)
        return bas_ij_idx, shl_pair_offsets

    def create_gout_stride_lookup_table(self, deriv=None, gout_width=36):
        # gout_width should be identical to the setting in cuda kernel
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        if deriv is None:
            deriv = (0, 0)
        i_inc, j_inc = deriv
        lmax = self.uniq_l_ctr[:,0].max()
        gout_stride_lookup = np.empty([L_AUX_MAX+1,L_AUX_MAX+1], dtype=np.int32)
        gout_width = gout_width
        shm_size = SHM_SIZE
        ls = np.arange(lmax+1)
        nf = (ls+1) * (ls+2) // 2
        max_shm_size = 0
        for li in range(lmax+1):
            for lj in range(lmax+1):
                unit = (li+1+i_inc)*(lj+1+j_inc)*3 + 4
                nsp_max = _nearest_power2(shm_size // (unit*8))
                gout_size = nf[li] * nf[lj]
                gout_stride = (gout_size+gout_width-1) / gout_width
                # Round up to the next 2^n
                gout_stride = _nearest_power2(gout_stride, return_leq=False)
                nsp_per_block = min(nsp_max, THREADS // gout_stride)
                gout_stride_lookup[li, lj] = THREADS // nsp_per_block
                max_shm_size = max(max_shm_size, nsp_per_block*unit*8)
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

    def intor(self, kern, hermi, comp, deriv_ij):
        if comp == 1:
            gout_width = 36
        else:
            gout_width = 18

        sorted_cell = self.sorted_cell
        int1e_envs = self.int1e_envs
        nao_cart, nao = self.coeff.shape
        bas_ij_idx, shl_pair_offsets = self.generate_shl_pairs(hermi)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        gout_stride_lookup, shm_size = self.create_gout_stride_lookup_table(
            deriv_ij, gout_width)
        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        out = cp.empty((bvk_ncells, comp, nao_cart, nao_cart))
        drv = getattr(libpbc, kern)
        err = drv(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs), ctypes.c_int(shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
            sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
            sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
            sorted_cell._env.ctypes)
        if err != 0:
            raise RuntimeError(f'{kern} failed')

        if hermi == 1:
            assert comp == 1
            out = fill_triu_bvk_conj(out, nao_cart, bvk_kmesh)
        out = sandwich_dot(out.reshape(-1,nao_cart,nao_cart), self.coeff)
        out = out.reshape(bvk_ncells, comp, nao, nao)

        if self.kpts is not None and not is_zero(self.kpts):
            bvkmesh_Ls = translation_vectors_for_kmesh(self.cell, bvk_kmesh, True)
            kpts = self.kpts.reshape(-1, 3)
            expLk = cp.exp(1j*asarray(bvkmesh_Ls.dot(kpts.T)))
            out = contract('lk,lxpq->kxpq', expLk, out)

        if comp == 1:
            out = out[:,0]
        if self.kpts is not None and self.kpts.ndim == 1:
            out = out[0]
        return out
