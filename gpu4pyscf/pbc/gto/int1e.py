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
from pyscf import lib
from pyscf.gto import ATOM_OF, PTR_COORD, Mole
from pyscf.pbc.gto import Cell
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc.tools.pbc import super_cell, _build_supcell_, get_lattice_Ls
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, hermi_triu, dist_matrix)
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, PBCIntEnvVars, most_diffuse_pgto, _scale_sp_ctr_coeff)
from gpu4pyscf.scf.jk import _nearest_power2, SHM_SIZE
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.df.int3c2e import (
    fill_triu_bvk, L_AUX_MAX, THREADS
)
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

__all__ = [
    'int1e_ovlp',
    'int1e_kin',
    'int1e_ipovlp',
    'int1e_ipkin',
    'int1e_r2_origi',
    'int1e_r4_origi',
    'int1e_r2_origi_ip2',
    'int1e_r4_origi_ip2',
    'ovlp_strain_deriv',
    'kin_strain_deriv',
]

libpbc.PBCint1e_ovlp.restype = ctypes.c_int
libpbc.PBCint1e_kin.restype = ctypes.c_int
libpbc.PBCint1e_ipovlp.restype = ctypes.c_int
libpbc.PBCint1e_ipkin.restype = ctypes.c_int
libpbc.PBCint1e_r2_origi.restype = ctypes.c_int
libpbc.PBCint1e_r4_origi.restype = ctypes.c_int
libpbc.PBCint1e_r2_origi_ip2.restype = ctypes.c_int
libpbc.PBCint1e_r4_origi_ip2.restype = ctypes.c_int

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    # Tighten the precision of overlap integrals because errors in overlap
    # matrix will significantly amplifies the error in eigenvectors of the
    # FC=SCe equation, especially when the basis functions are linear
    # dependent or the eigenvalues have small gaps.
    opt = _check_opt(cell, 1, kpts, bvk_kmesh, 1e-4)
    return opt.intor('PBCint1e_ovlp', 1, (0, 0), kpts, sort_output)

def int1e_kin(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    # The Laplacian can increase the integral by ~4 a^2 r^2, so tighten the
    # precision to capture this effect.
    opt = _check_opt(cell, 1, kpts, bvk_kmesh, 1e-2)
    return opt.intor('PBCint1e_kin', 1, (2, 0), kpts, sort_output)

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    # hermi=2 for anti-symmetric matrices
    opt = _check_opt(cell, 2, kpts, bvk_kmesh)
    return opt.intor('PBCint1e_ipovlp', 3, (1, 0), kpts, sort_output)

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    opt = _check_opt(cell, 2, kpts, bvk_kmesh, 1e-2)
    return opt.intor('PBCint1e_ipkin', 3, (3, 0), kpts, sort_output)

def int1e_r2_origi(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh)
    return opt.intor('PBCint1e_r2_origi', 1, (0, 2), kpts, sort_output)

def int1e_r4_origi(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh)
    return opt.intor('PBCint1e_r4_origi', 1, (0, 4), kpts, sort_output)

def int1e_r2_origi_ip2(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh)
    return opt.intor('PBCint1e_r2_origi_ip2', 3, (0, 3), kpts, sort_output)

def int1e_r4_origi_ip2(cell, kpts=None, bvk_kmesh=None, sort_output=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh)
    return opt.intor('PBCint1e_r4_origi_ip2', 3, (0, 5), kpts, sort_output)

def ovlp_strain_deriv(cell, dm, kpts=None):
    assert isinstance(cell, Cell)
    opt = _Int1eOpt(cell, 1)
    return opt.get_ovlp_strain_deriv(dm, kpts)

def kin_strain_deriv(cell, dm, kpts=None):
    assert isinstance(cell, Cell)
    with lib.temporary_env(cell, precision=cell.precision*1e-2):
        opt = _Int1eOpt(cell, 1)
    return opt.get_kin_strain_deriv(dm, kpts)

def _check_opt(cell, hermi, kpts, bvk_kmesh, scale_precision=1):
    if isinstance(cell, Mole):
        return _Int1eOpt(cell, hermi)

    assert isinstance(cell, Cell)
    if kpts is None or is_zero(kpts):
        bvk_kmesh = np.ones(3, dtype=int)
    elif bvk_kmesh is None:
        bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1,3), bound_by_supmol=True)

    rcut = cell.rcut
    precision = cell.precision * scale_precision
    if scale_precision < 1:
        a, c, l = most_diffuse_pgto(cell)
        rcut = _estimate_rcut(a, l, c, precision)

    with lib.temporary_env(cell, precision=precision, rcut=rcut):
        return _Int1eOpt(cell, hermi, bvk_kmesh)

class _Int1eOpt:
    def __init__(self, cell, hermi=0, bvk_kmesh=None):
        self.cell = cell = SortedGTO.from_cell(cell, decontract=True)
        lmax = self.cell.uniq_l_ctr[:,0].max()
        assert lmax <= L_AUX_MAX

        bvk_ncells = 1
        if isinstance(cell, Mole):
            bvk_kmesh = None
            bvkcell = cell
            bvkmesh_Ls = Ls = cp.zeros((1, 3))
        else:
            if bvk_kmesh is None:
                bvkmesh_Ls = cp.zeros((1, 3))
            else:
                bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
            bvk_ncells = len(bvkmesh_Ls)
            if bvk_ncells == 1:
                bvkcell = cell
            else:
                bvkcell = super_cell(cell, bvk_kmesh, wrap_around=True)
                # PTR_BAS_COORD was not initialized in the super_cell function
                bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            Ls = _bvkcell_lattice_sum_Ls(bvkcell)
            Ls = Ls[np.linalg.norm(Ls-.5, axis=1).argsort()]

            rad = bvkcell.rcut / bvkcell.vol**(1./3) + 1
            surface = 4*np.pi * rad**2
            lattice_sum_factor = surface
            precision = bvkcell.precision / lattice_sum_factor

        self.hermi = hermi
        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        self.Ls = Ls
        self.bvkmesh_Ls = bvkmesh_Ls

        _env = _scale_sp_ctr_coeff(bvkcell)
        self.int1e_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, len(Ls),
            bvkcell._atm, bvkcell._bas, _env, cell.p_ao_loc, Ls)

        if isinstance(cell, Mole):
            bas_ij_cache = cell.generate_shl_pairs(hermi)
        else:
            mask = _shell_overlap_mask(cell, hermi, precision, Ls,
                                       self.int1e_envs, bvkmesh_Ls)
            nbas = cell.nbas
            l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
            groups = len(cell.uniq_l_ctr)
            if hermi == 1:
                ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
            else:
                ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
            img_offsets = cp.arange(bvk_ncells, dtype=np.int32) * nbas
            bas_ij_cache = {}
            for i, j in ij_tasks:
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                ish = cp.arange(ish0, ish1, dtype=np.int32)
                jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
                ijsh = ish[:,None,None] * (nbas*bvk_ncells) + img_offsets[:,None] + jsh
                if hermi == 1 and i == j:
                    sub_mask = mask[ish0:ish1,:,jsh0:jsh1].transpose(0,2,1)
                    # disable the off-diag blocks
                    sub_mask[ish[:,None] < jsh] = False
                    sub_mask = sub_mask.transpose(0,2,1)
                else:
                    sub_mask = mask[ish0:ish1,:,jsh0:jsh1]
                bas_ij_cache[i,j] = ijsh[sub_mask]
        self.bas_ij_cache = bas_ij_cache

        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(bas_ij_cache)
        self.bas_ij_idx = bas_ij_idx
        self.shl_pair_offsets = shl_pair_offsets

    @property
    def rys_envs(self):
        return self.int1e_envs

    def intor(self, kern, comp, deriv_ij, kpts=None, sort_output=True,
              out=None, buf=None, shls_slice=None):
        if comp == 1:
            gout_width = 36
        else:
            gout_width = 18

        cell = self.cell

        if shls_slice is None:
            nbas = cell.nbas
            shls_slice = (0, nbas, 0, nbas)
            ij_offset = 0
            naoi = naoj = cell.nao
        else:
            assert not sort_output
            ish0, ish1, jsh0, jsh1 = shls_slice
            ao_loc = cell.ao_loc
            i0 = int(ao_loc[ish0])
            j0 = int(ao_loc[jsh0])
            naoi = ao_loc[ish1] - i0
            naoj = ao_loc[jsh1] - j0
            ij_offset = i0 * naoj + j0

        if isinstance(self.cell, Mole) or self.bvk_kmesh is not None:
            # if kpts is None, compute integrals at gamma point
            ncells = len(self.bvkmesh_Ls)
            int1e_envs = self.int1e_envs
        else:
            assert kpts is not None
            # build supmol for evaluating integrals <cell-0|super-mol>, which
            # can be transformed to integrals at arbitrary k-points
            supmol = cell.copy(deep=False)
            supmol = _build_supcell_(supmol, cell, cp.asnumpy(self.Ls))
            supmol._bas[:,PTR_BAS_COORD] = supmol._atm[supmol._bas[:,ATOM_OF],PTR_COORD]
            ncells = len(self.Ls)
            Ls = cp.zeros((1, 3))
            _env = _scale_sp_ctr_coeff(supmol)
            int1e_envs = PBCIntEnvVars.new(
                cell.natm, cell.nbas, ncells, 1, supmol._atm, supmol._bas, _env,
                supmol.ao_loc, Ls)

        gout_stride, max_shm_size = _gout_stride_lookup_table(cell, deriv_ij, gout_width)
        nbatches_shl_pair = len(self.shl_pair_offsets) - 1

        mat = ndarray((ncells, comp, naoi, naoj), buffer=buf)
        mat.fill(0)
        drv = getattr(libpbc, kern)
        err = drv(
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs), ctypes.c_int(max_shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.c_int(naoi), ctypes.c_int(naoj), ctypes.c_size_t(ij_offset))
        if err != 0:
            raise RuntimeError(f'{kern} failed')

        is_gamma_point = kpts is None or is_zero(kpts)
        if isinstance(cell, Mole) or is_gamma_point:
            if ncells > 1: # corresponding to self.bvk_kmesh is None
                mat = mat.sum(axis=0)
            mat = mat.reshape(comp, naoi, naoj)
            if self.hermi != 0:
                mat = hermi_triu(mat, self.hermi, inplace=True)
            if sort_output:
                out = cell.apply_CT_mat_C(mat, out=out)
            else:
                out = mat
            if comp == 1:
                out = out[0]
            if kpts is not None and kpts.ndim == 2:
                # In k-mesh KS calculations, the leading dimension is the index
                # for k-points.
                out = out[None]
        else:
            is_single_kpt = kpts.ndim == 1
            kpts = asarray(kpts.reshape(-1, 3))
            nkpts = len(kpts)
            if self.bvk_kmesh is None:
                expLk = cp.exp(1j*asarray(self.Ls).dot(kpts.T))
            else:
                expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(kpts.T))
            expLkz = expLk.view(np.float64).reshape(ncells,nkpts,2)
            mat = contract('lkz,lxpq->kxpqz', expLkz, mat)
            mat = mat.view(np.complex128)[:,:,:,:,0]
            mat = mat.reshape(nkpts*comp, naoi, naoj)
            if self.hermi != 0:
                mat = hermi_triu(mat, self.hermi, inplace=True)
            if sort_output:
                out = cell.apply_CT_mat_C(mat, out=out)
            else:
                out = mat
            if comp > 1:
                out = out.reshape((nkpts, comp) + out.shape[-2:])
            if is_single_kpt:
                out = out[0]
        return out

    def strain_deriv_intor(self, dm, kern, deriv, kpts=None):
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
            if self.bvk_kmesh is None:
                expLk = cp.exp(1j*asarray(self.Ls).dot(kpts.T))
            else:
                expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(kpts.T))
            dm = contract('Lk,kpq->Lpq', expLk, dm)
            expLk = None
            dm = dm.real
        dm = cp.asarray(dm, order='C')

        assert self.bvk_kmesh is None
        assert self.hermi == 1
        gout_stride_lookup, shm_size = _gout_stride_lookup_table(cell, deriv)
        nbatches_shl_pair = len(self.shl_pair_offsets) - 1

        sigma = cp.zeros((3, 3))
        drv = getattr(libpbc, kern)
        err = drv(
            ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(self.int1e_envs),
            ctypes.c_int(shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(self.shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError(f'{kern} failed')
        sigma = sigma.get()
        sigma *= 2 / nkpts
        return sigma

    def get_ovlp_strain_deriv(self, dm, kpts=None):
        '''Computes the strain derivatives for the product of density matrix and
        overlap matrix. In the case of k-points calculations, the derivatives
        are averaged over k-mesh.
        '''
        return self.strain_deriv_intor(dm, 'PBCovlp_strain_deriv', (1, 0), kpts)

    def get_kin_strain_deriv(self, dm, kpts=None):
        '''Computes the strain derivatives for the product of density matrix and
        kinetic matrix. In the case of k-points calculations, the derivatives
        are averaged over k-mesh.
        '''
        return self.strain_deriv_intor(dm, 'PBCkin_strain_deriv', (3, 0), kpts)

class CrossInt1e(_Int1eOpt):
    def __init__(self, cell1, cell2, bvk_kmesh=None):
        self.cell1 = cell1 = SortedGTO.from_cell(cell1, decontract=True)
        self.cell2 = cell2 = SortedGTO.from_cell(cell2, decontract=True)
        self.cell = cell = cell12 = cell1 + cell2

        cell._bas[:,PTR_BAS_COORD] = cell._atm[cell._bas[:,ATOM_OF],PTR_COORD]
        cell.uniq_l_ctr = np.vstack([cell1.uniq_l_ctr, cell2.uniq_l_ctr])

        self.hermi = 0

        bvk_ncells = 1
        if isinstance(cell1, Mole):
            assert isinstance(cell2, Mole)
            bvk_kmesh = None
            bvkcell = cell12
            bvkmesh_Ls = Ls = np.zeros((1, 3))
            precision = 1e-14
        else:
            if bvk_kmesh is None:
                bvkmesh_Ls = cp.zeros((1, 3))
            else:
                bvkmesh_Ls = translation_vectors_for_kmesh(cell1, bvk_kmesh, True)
            bvk_ncells = len(bvkmesh_Ls)
            if bvk_ncells == 1:
                bvkcell = cell12
            else:
                bvkcell = super_cell(cell12, bvk_kmesh, wrap_around=True)
                # PTR_BAS_COORD was not initialized in supe_rcell
                bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            Ls = _bvkcell_lattice_sum_Ls(bvkcell)
            Ls = Ls[np.linalg.norm(Ls-.5, axis=1).argsort()]

            rad = bvkcell.rcut / bvkcell.vol**(1./3) + 1
            surface = 4*np.pi * rad**2
            lattice_sum_factor = surface
            precision = bvkcell.precision / lattice_sum_factor

        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        self.Ls = Ls
        self.bvkmesh_Ls = bvkmesh_Ls

        ao_loc = cell12.ao_loc
        _env = _scale_sp_ctr_coeff(bvkcell)
        self.int1e_envs = PBCIntEnvVars.new(
            cell12.natm, cell12.nbas, bvk_ncells, len(Ls),
            bvkcell._atm, bvkcell._bas, _env, ao_loc, Ls)

        shls_slice = (0, cell1.nbas, cell1.nbas, cell12.nbas)
        mask = _shell_overlap_mask(
            cell12, 0, precision, Ls, self.int1e_envs, bvkmesh_Ls,
            shls_slice=shls_slice)

        self.bas_ij_cache = bas_ij_cache = {}
        nbas = cell12.nbas
        img_offsets = cp.arange(bvk_ncells, dtype=np.int32) * nbas
        l_ctr_offsets1 = np.append(0, np.cumsum(cell1.l_ctr_counts))
        l_ctr_offsets2 = np.append(0, np.cumsum(cell2.l_ctr_counts))
        for i in range(len(cell1.uniq_l_ctr)):
            for j in range(len(cell2.uniq_l_ctr)):
                ish0, ish1 = l_ctr_offsets1[i], l_ctr_offsets1[i+1]
                jsh0, jsh1 = l_ctr_offsets2[j], l_ctr_offsets2[j+1]
                ish = cp.arange(ish0, ish1, dtype=np.int32)
                jsh = cp.arange(jsh0, jsh1, dtype=np.int32) + cell1.nbas
                ijsh = ish[:,None,None] * (nbas*bvk_ncells) + img_offsets[:,None] + jsh
                bas_ij_cache[i,j] = ijsh[mask[ish0:ish1,:,jsh0:jsh1]]

        bas_ij_idx, shl_pair_offsets = cell1.aggregate_shl_pairs(bas_ij_cache)
        self.bas_ij_idx = bas_ij_idx
        self.shl_pair_offsets = shl_pair_offsets

    def intor(self, kern, comp, deriv_ij, kpts=None, sort_output=True,
              out=None, buf=None, shls_slice=None):
        shls_slice = (0, self.cell1.nbas, self.cell1.nbas, self.cell.nbas)
        out = super().intor(kern, comp, deriv_ij, kpts, False, out, buf,
                             shls_slice)
        if sort_output:
            leading_shape = out.shape[:-2]
            n1, n2 = out.shape[-2:]
            tmp = self.cell2.apply_CT_dot(out.reshape(-1, n1, n2), axis=2)
            out = self.cell1.apply_CT_dot(tmp, axis=1, out=out)
            out = out.reshape(leading_shape + out.shape[1:3])
        return out

def _gout_stride_lookup_table(cell, deriv=None, gout_width=36):
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

    lmax = cell.uniq_l_ctr[:,0].max()
    max_shm_size = shm_size[:lmax+1,:lmax+1].max()
    return cp.array(gout_stride_lookup, dtype=np.int32), max_shm_size

def _shell_overlap_mask(cell, hermi=1, precision=1e-14, Ls=None, envs=None,
                        bvkmesh_Ls=None, shls_slice=None):
    '''mask[i,bvkL,j] = absmax(<i|bvkL+j>) > precision'''
    exps, cs = extract_pgto_params(cell, 'diffuse')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(asarray(cs, dtype=np.float32)))

    if shls_slice is None:
        nbas = cell.nbas
        shls_slice = (0, nbas, 0, nbas)
    i0, i1, j0, j1 = shls_slice
    nbas1 = i1 - i0
    nbas2 = j1 - j0

    if bvkmesh_Ls is None:
        bvkmesh_Ls = cp.zeros((1, 3))
    else:
        bvkmesh_Ls = cp.asarray(bvkmesh_Ls)
    ncells = len(bvkmesh_Ls)
    ovlp_mask = cp.zeros((nbas1,ncells,nbas2), dtype=bool)

    if envs is None:
        ao_loc = cp.zeros(1, dtype=np.int32)
        if Ls is None:
            Ls = cp.zeros((1, 3))
        else:
            Ls = asarray(Ls)
        envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, ncells, len(Ls), asarray(cell._atm),
            asarray(cell._bas), asarray(_scale_sp_ctr_coeff(cell)), ao_loc, Ls)
    else:
        assert envs.bvk_ncells == ncells

    libpbc.PBCovlp_mask_estimation(
        ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
        ctypes.cast(exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_int(hermi),
        ctypes.c_float(math.log(precision)),
        ctypes.cast(bvkmesh_Ls.data.ptr, ctypes.c_void_p),
        (ctypes.c_int*4)(*shls_slice))
    return ovlp_mask

def _bvkcell_lattice_sum_Ls(bvkcell, rcut=None):
    if rcut is None:
        rcut = bvkcell.rcut
    Ls = get_lattice_Ls(bvkcell, rcut=rcut, discard=False)
    if len(Ls) > 1:
        r = asarray(bvkcell.atom_coords())
        dist_max = dist_matrix(r, r).max().get()
        Ls_mask = np.linalg.norm(Ls, axis=1) < rcut + dist_max
        Ls = Ls[Ls_mask]
    return np.asarray(Ls, order='C')
