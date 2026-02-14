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

import math
import ctypes
import numpy as np
import cupy as cp
from pyscf.gto import ATOM_OF, PTR_COORD, Mole
from pyscf.pbc.tools.pbc import super_cell, _build_supcell_
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, hermi_triu
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, PBCIntEnvVars, _shell_overlap_mask,
    _scale_sp_ctr_coeff)
from gpu4pyscf.scf.jk import _nearest_power2, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.df.int3c2e import (
    fill_triu_bvk, L_AUX_MAX, THREADS
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

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None):
    opt = _Int1eOpt(cell, bvk_kmesh)
    out = opt.intor('PBCint1e_ovlp', 1, 1, (0, 0), kpts=kpts)
    return out

def int1e_kin(cell, kpts=None, bvk_kmesh=None):
    opt = _Int1eOpt(cell, bvk_kmesh)
    out = opt.intor('PBCint1e_kin', 1, 1, (2, 0), kpts=kpts)
    return out

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None):
    opt = _Int1eOpt(cell, bvk_kmesh)
    out = opt.intor('PBCint1e_ipovlp', 0, 3, (1, 0), kpts=kpts)
    return out

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None):
    opt = _Int1eOpt(cell, bvk_kmesh)
    out = opt.intor('PBCint1e_ipkin', 0, 3, (3, 0), kpts=kpts)
    return out

class _Int1eOpt:
    def __init__(self, cell, bvk_kmesh=None):
        self.cell = cell = SortedGTO.from_cell(
            cell, allow_replica=1, allow_split_seg_contraction=False)
        lmax = self.cell.uniq_l_ctr[:,0].max()
        assert lmax <= L_AUX_MAX

        bvk_ncells = 1
        if isinstance(cell, Mole):
            bvk_kmesh = None
            bvkcell = cell
            Ls = cp.zeros((1, 3))
            self.precision = 1e-16
        else:
            self.precision = cell.precision * 1e-4
            if bvk_kmesh is not None:
                bvk_ncells = np.prod(bvk_kmesh)
            if bvk_ncells == 1:
                bvkcell = cell
            else:
                bvkcell = super_cell(cell, bvk_kmesh, wrap_around=True)
                # PTR_BAS_COORD was not initialized in supe_rcell
                bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            Ls = asarray(bvkcell.get_lattice_Ls(rcut=cell.rcut))
            Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        self.bvk_ncells = bvk_ncells
        self.Ls = Ls
        self._int1e_envs = {}

    @multi_gpu.property(cache='_int1e_envs')
    def int1e_envs(self):
        cell = self.cell
        bvkcell = self.bvkcell
        _env = _scale_sp_ctr_coeff(bvkcell)
        nimgs = len(self.Ls)
        return PBCIntEnvVars.new(
            cell.natm, cell.nbas, self.bvk_ncells, nimgs,
            bvkcell._atm, bvkcell._bas, _env, cell.p_ao_loc, self.Ls)

    def aggregate_shl_pairs(self, hermi=1, within_bvkcell=True, Ls=None):
        cell = self.cell
        if isinstance(cell, Mole):
            mol = cell
            bas_ij_cache = mol.generate_shl_pairs(hermi)
            bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(bas_ij_cache)
            return bas_ij_idx, shl_pair_offsets

        nbas = cell.nbas
        bvk_ncells = self.bvk_ncells
        if within_bvkcell:
            exps, coef = extract_pgto_params(self.bvkcell, 'diffuse')
            log_c = cp.log(cp.asarray(coef, dtype=np.float32))
            diffuse_exps = cp.asarray(exps, dtype=np.float32)
            log_cutoff = math.log(self.precision)
            int1e_envs = self.int1e_envs
            img_counts = cp.zeros((nbas*bvk_ncells*nbas), dtype=np.uint32)
            libpbc.bvk_ovlp_img_counts(
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                ctypes.byref(int1e_envs),
                ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff), ctypes.c_int(hermi))
            mask = img_counts.reshape(nbas, bvk_ncells, nbas) > 0
            ncells = bvk_ncells
        else:
            # Generate all shell-paris within super-mol
            if Ls is None:
                Ls = self.Ls
            ncells = len(Ls)
            mask = _shell_overlap_mask(cell, hermi, self.precision, Ls)

        # The effective shell pair = ish*nbas+jsh
        bas_ij_cache = {}
        l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
        groups = len(cell.uniq_l_ctr)
        if hermi == 1:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
        img_offsets = cp.arange(ncells, dtype=np.int32) * nbas
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            ijsh = ish[:,None,None] * (nbas*ncells) + img_offsets[:,None] + jsh
            if hermi == 1 and i == j:
                sub_mask = mask[ish0:ish1,:,jsh0:jsh1].transpose(0,2,1)
                # disable the off-diag blocks
                sub_mask[ish[:,None] < jsh] = False
                sub_mask = sub_mask.transpose(0,2,1)
            else:
                sub_mask = mask[ish0:ish1,:,jsh0:jsh1]
            bas_ij_cache[i,j] = ijsh[sub_mask]

        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(bas_ij_cache)
        return bas_ij_idx, shl_pair_offsets

    def create_gout_stride_lookup_table(self, deriv=None, gout_width=36):
        # gout_width should be identical to the setting in cuda kernel
        # based on the shm_size, find optimal gout_stride for each (li,lj)
        # pattern, store them in the gout_stride_lookup
        if deriv is None:
            deriv = (0, 0)
        i_inc, j_inc = deriv

        ls = np.arange(L_AUX_MAX+1)
        nf = (ls+1) * (ls+2) // 2
        li = ls[:,None]
        lj = ls
        unit = (li+1+i_inc)*(lj+1+j_inc)*3 + 4
        nsp_max = _nearest_power2(SHM_SIZE // (unit*8))
        gout_size = nf[li] * nf[lj]
        gout_stride = (gout_size+gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
        nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
        gout_stride_lookup = THREADS // nsp_per_block
        shm_size = nsp_per_block*unit*8

        lmax = self.cell.uniq_l_ctr[:,0].max()
        max_shm_size = shm_size[:lmax+1,:lmax+1].max()
        return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

    def intor(self, kern, hermi, comp, deriv_ij, kpts=None, out=None, buf=None):
        if comp == 1:
            gout_width = 36
        else:
            gout_width = 18

        cell = self.cell
        nao_cart = cell.nao
        if self.bvk_kmesh is not None or kpts is None:
            # if kpts is None, compute integrals at gamma point
            gout_stride, max_shm_size = self.create_gout_stride_lookup_table(deriv_ij, gout_width)
            bas_ij_idx, shl_pair_offsets = self.aggregate_shl_pairs(hermi, True)
            nbatches_shl_pair = len(shl_pair_offsets) - 1
            ncells = self.bvk_ncells
            int1e_envs = self.int1e_envs
        else:
            assert kpts is not None
            # build supmol for evaluating integrals <cell-0|super-mol>, which
            # can be transformed to integrals at arbitrary k-points
            supmol = cell.copy(deep=False)
            supmol = _build_supcell_(supmol, cell, self.Ls.get())
            supmol._bas[:,PTR_BAS_COORD] = supmol._atm[supmol._bas[:,ATOM_OF],PTR_COORD]
            ncells = len(self.Ls)

            # ket is extended to all images. No symmetry between bra and ket
            hermi = 0
            gout_stride, max_shm_size = self.create_gout_stride_lookup_table(deriv_ij, gout_width)
            bas_ij_idx, shl_pair_offsets = self.aggregate_shl_pairs(hermi, False)
            nbatches_shl_pair = len(shl_pair_offsets) - 1

            Ls = cp.zeros((1, 3))
            _env = _scale_sp_ctr_coeff(supmol)
            int1e_envs = PBCIntEnvVars.new(
                cell.natm, cell.nbas, ncells, 1, supmol._atm, supmol._bas, _env,
                supmol.ao_loc, Ls)

        mat = ndarray((ncells, comp, nao_cart, nao_cart), buffer=buf)
        mat[:] = 0
        drv = getattr(libpbc, kern)
        err = drv(
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs), ctypes.c_int(max_shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError(f'{kern} failed')

        if hermi == 1:
            assert comp == 1
            if self.bvk_kmesh is not None:
                mat = fill_triu_bvk(mat, nao_cart, self.bvk_kmesh, bvk_axis=0)
            else:
                assert kpts is None
                mat = hermi_triu(mat[0], hermi=1, inplace=True)

        if kpts is None or (is_zero(kpts) and kpts.ndim == 1):
            out = cell.apply_CT_mat_C(mat[0], out=out)
            if comp == 1:
                out = out[0]
        else:
            is_single_kpt = kpts.ndim == 1
            kpts = asarray(kpts.reshape(-1, 3))
            nkpts = len(kpts)
            nao = cell.cell.nao
            if self.bvk_kmesh is None:
                expLk = cp.exp(1j*self.Ls.dot(kpts.T))
            else:
                bvkmesh_Ls = translation_vectors_for_kmesh(cell, self.bvk_kmesh, True)
                expLk = cp.exp(1j*asarray(bvkmesh_Ls).dot(kpts.T))
            expLkz = expLk.view(np.float64).reshape(ncells,nkpts,2)
            mat = cell.apply_CT_mat_C(mat.reshape(-1,nao_cart,nao_cart))
            mat = mat.reshape(ncells, comp, nao, nao)
            out = ndarray((nkpts,comp,nao,nao,2), buffer=out, dtype=np.float64)
            out = contract('lkz,lxpq->kxpqz', expLkz, mat, out=out)
            out = out.view(np.complex128)[:,:,:,:,0]
            if comp == 1:
                out = out[:,0]
            if is_single_kpt:
                out = out[0]
        return out

    def get_ovlp_strain_deriv(self, dm, kpts=None):
        '''Computes the strain derivatives for the product of density matrix and
        overlap matrix. In the case of k-points calculations, the derivatives
        are averaged over k-mesh.
        '''
        cell = self.cell
        dm = cell.apply_C_mat_CT(dm)
        if kpts is None:
            kpts = np.zeros((1, 3))
        else:
            kpts = kpts.reshape(-1, 3)
        nkpts = len(kpts)
        nao = dm.shape[-1]
        dm = dm.reshape(-1, nkpts, nao, nao)
        if len(dm) == 1:
            dm = dm[0]
        else:
            dm = dm.sum(axis=0)

        is_gamma_point = is_zero(kpts)
        if is_gamma_point:
            assert dm.dtype == np.float64
        else:
            expLk = cp.exp(1j * asarray(self.Ls).dot(asarray(kpts).T))
            dm = contract('Lk,kpq->Lpq', expLk, dm)
            expLk = None
            dm = dm.real
        dm = cp.asarray(dm, order='C')

        Ls = asarray(cell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

        hermi = 1
        deriv = (1, 0)
        gout_stride_lookup, shm_size = self.create_gout_stride_lookup_table(deriv)
        bas_ij_idx, shl_pair_offsets = self.aggregate_shl_pairs(hermi, False, Ls)
        nbatches_shl_pair = len(shl_pair_offsets) - 1

        _env = _scale_sp_ctr_coeff(cell)
        int1e_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, 1, len(Ls), cell._atm, cell._bas, _env,
            cell.p_ao_loc, Ls)

        sigma = cp.zeros((3, 3))
        libpbc.PBCovlp_strain_deriv(
            ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs),
            ctypes.c_int(shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
            ctypes.c_int(int(is_gamma_point)))
        sigma = sigma.get()
        sigma *= 2 / nkpts
        return sigma
