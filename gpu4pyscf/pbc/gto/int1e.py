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
from pyscf.pbc.tools.pbc import super_cell, _build_supcell_
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, hermi_triu
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedGTO, PBCIntEnvVars, most_diffuse_pgto, _scale_sp_ctr_coeff)
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
    'ovlp_strain_deriv',
    'kin_strain_deriv',
]

libpbc.PBCint1e_ovlp.restype = ctypes.c_int
libpbc.PBCint1e_kin.restype = ctypes.c_int
libpbc.PBCint1e_ipovlp.restype = ctypes.c_int
libpbc.PBCint1e_ipkin.restype = ctypes.c_int

def int1e_ovlp(cell, kpts=None, bvk_kmesh=None, kpts_in_bvkcell=True):
    # Tighten the precision of overlap integrals because errors in overlap
    # matrix will significantly amplifies the error in eigenvectors of the
    # FC=SCe equation, especially when the basis functions are linear
    # dependent or the eigenvalues have small gaps.
    scale_precision = 1
    if isinstance(cell, Cell):
        scale_precision = min(cell.precision, 1e-4)
    opt = _check_opt(cell, 1, kpts, bvk_kmesh, kpts_in_bvkcell, scale_precision)
    return opt.intor('PBCint1e_ovlp', 1, (0, 0), kpts=kpts)

def int1e_kin(cell, kpts=None, bvk_kmesh=None, kpts_in_bvkcell=True):
    # The Laplacian can increase the integral by ~4 a^2 r^2, so tighten the
    # precision to capture this effect.
    opt = _check_opt(cell, 1, kpts, bvk_kmesh, kpts_in_bvkcell, 1e-4)
    return opt.intor('PBCint1e_kin', 1, (2, 0), kpts=kpts)

def int1e_ipovlp(cell, kpts=None, bvk_kmesh=None, kpts_in_bvkcell=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh, kpts_in_bvkcell, 1e-1)
    return opt.intor('PBCint1e_ipovlp', 3, (1, 0), kpts=kpts)

def int1e_ipkin(cell, kpts=None, bvk_kmesh=None, kpts_in_bvkcell=True):
    opt = _check_opt(cell, 0, kpts, bvk_kmesh, kpts_in_bvkcell, 1e-2)
    return opt.intor('PBCint1e_ipkin', 3, (3, 0), kpts=kpts)

def ovlp_strain_deriv(cell, dm, kpts=None):
    assert isinstance(cell, Cell)
    opt = _Int1eOpt(cell, 1)
    return opt.get_ovlp_strain_deriv(dm, kpts)

def kin_strain_deriv(cell, dm, kpts=None):
    assert isinstance(cell, Cell)
    with lib.temporary_env(cell, precision=cell.precision*1e-2):
        opt = _Int1eOpt(cell, 1)
    return opt.get_kin_strain_deriv(dm, kpts)

def _check_opt(cell, hermi, kpts, bvk_kmesh, kpts_in_bvkcell, scale_precision=1):
    if isinstance(cell, Mole):
        return _Int1eOpt(cell, hermi)

    assert isinstance(cell, Cell)
    if kpts is None:
        bvk_kmesh = np.ones(3, dtype=int)
    elif bvk_kmesh is None and kpts_in_bvkcell:
        bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1,3), bound_by_supmol=True)

    precision = cell.precision * scale_precision
    if scale_precision < 1:
        a, c, l = most_diffuse_pgto(cell)
        rcut = _estimate_rcut(a, l, c, precision)

    with lib.temporary_env(cell, precision=precision, rcut=rcut):
        return _Int1eOpt(cell, hermi, bvk_kmesh)

class _Int1eOpt:
    def __init__(self, cell, hermi=0, bvk_kmesh=None):
        self.cell = cell = SortedGTO.from_cell(
            cell, allow_replica=1, allow_split_seg_contraction=False)
        lmax = self.cell.uniq_l_ctr[:,0].max()
        assert lmax <= L_AUX_MAX

        bvk_ncells = 1
        if isinstance(cell, Mole):
            bvk_kmesh = None
            bvkcell = cell
            Ls = cp.zeros((1, 3))
        else:
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
        self.hermi = hermi
        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        self.bvk_ncells = bvk_ncells
        self.Ls = Ls

        _env = _scale_sp_ctr_coeff(bvkcell)
        self.int1e_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, self.bvk_ncells, len(Ls),
            bvkcell._atm, bvkcell._bas, _env, cell.p_ao_loc, Ls)

        if isinstance(cell, Mole):
            bas_ij_cache = cell.generate_shl_pairs(hermi)
            bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(bas_ij_cache)
        elif bvk_kmesh is not None:
            nbas = cell.nbas
            exps, coef = extract_pgto_params(self.bvkcell, 'diffuse')
            log_c = cp.log(cp.asarray(coef, dtype=np.float32))
            diffuse_exps = cp.asarray(exps, dtype=np.float32)
            log_cutoff = math.log(cell.precision)
            img_counts = cp.zeros((nbas*bvk_ncells*nbas), dtype=np.uint32)
            libpbc.bvk_ovlp_img_counts(
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                ctypes.byref(self.int1e_envs),
                ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff), ctypes.c_int(hermi))
            mask = img_counts.reshape(nbas, bvk_ncells, nbas) > 0
            bas_ij_idx, shl_pair_offsets = _aggregate_shl_pairs(cell, mask, hermi)
        else:
            mask = _shell_overlap_mask(cell, hermi, cell.precision, Ls)
            bas_ij_idx, shl_pair_offsets = _aggregate_shl_pairs(cell, mask, hermi)
        self.bas_ij_idx = bas_ij_idx
        self.shl_pair_offsets = shl_pair_offsets

    def intor(self, kern, comp, deriv_ij, kpts=None, out=None, buf=None):
        if comp == 1:
            gout_width = 36
        else:
            gout_width = 18

        cell = self.cell
        nao_cart = cell.nao

        if isinstance(self.cell, Mole) or self.bvk_kmesh is not None:
            # if kpts is None, compute integrals at gamma point
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
            Ls = cp.zeros((1, 3))
            _env = _scale_sp_ctr_coeff(supmol)
            int1e_envs = PBCIntEnvVars.new(
                cell.natm, cell.nbas, ncells, 1, supmol._atm, supmol._bas, _env,
                supmol.ao_loc, Ls)

        gout_stride, max_shm_size = _gout_stride_lookup_table(cell, deriv_ij, gout_width)
        nbatches_shl_pair = len(self.shl_pair_offsets) - 1

        mat = ndarray((ncells, comp, nao_cart, nao_cart), buffer=buf)
        mat[:] = 0
        drv = getattr(libpbc, kern)
        err = drv(
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.byref(int1e_envs), ctypes.c_int(max_shm_size),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError(f'{kern} failed')

        is_gamma_point = kpts is None or is_zero(kpts)
        if isinstance(cell, Mole) or is_gamma_point:
            if ncells > 1: # corresponding to self.bvk_kmesh is None
                mat = mat.sum(axis=0)
            mat = mat.reshape(comp, nao_cart, nao_cart)
            if self.hermi == 1:
                mat = hermi_triu(mat, hermi=1, inplace=True)
            out = cell.apply_CT_mat_C(mat, out=out)
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
                expLk = cp.exp(1j*self.Ls.dot(kpts.T))
            else:
                bvkmesh_Ls = translation_vectors_for_kmesh(cell, self.bvk_kmesh, True)
                expLk = cp.exp(1j*asarray(bvkmesh_Ls).dot(kpts.T))
            expLkz = expLk.view(np.float64).reshape(ncells,nkpts,2)
            mat = contract('lkz,lxpq->kxpqz', expLkz, mat)
            mat = mat.view(np.complex128)[:,:,:,:,0]
            mat = mat.reshape(nkpts*comp, nao_cart, nao_cart)
            if self.hermi == 1:
                assert comp == 1
                mat = hermi_triu(mat, hermi=1, inplace=True)
            out = cell.apply_CT_mat_C(mat, out=out)
            if comp > 1:
                nao = out.shape[-1]
                out = out.reshape(nkpts, comp, nao, nao)
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
            expLk = cp.exp(1j * asarray(self.Ls).dot(asarray(kpts).T))
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
            ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
            ctypes.c_int(int(is_gamma_point)))
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

def _aggregate_shl_pairs(cell, mask, hermi=1):
    nbas, ncells = mask.shape[:2]
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

def _shell_overlap_mask(mol, hermi=1, precision=1e-14, Ls=None):
    '''absmax(<i|j>) > precision for each shell pair'''
    nbas = mol.nbas
    exps, cs = extract_pgto_params(mol, 'diffuse')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(asarray(cs, dtype=np.float32)))
    ao_loc = cp.arange(0)
    with_images = Ls is not None
    if Ls is None:
        Ls = cp.zeros((1, 3))
    else:
        Ls = asarray(Ls)
    nimgs = len(Ls)
    ovlp_mask = cp.zeros((nbas,nimgs,nbas), dtype=bool)
    envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, nimgs, nimgs, asarray(mol._atm),
        asarray(mol._bas), asarray(_scale_sp_ctr_coeff(mol)), ao_loc, Ls)
    libpbc.PBCovlp_mask_estimation(
        ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
        ctypes.cast(exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_int(hermi),
        ctypes.c_float(math.log(precision)))
    if not with_images:
        ovlp_mask = ovlp_mask[:,0]
    return ovlp_mask
