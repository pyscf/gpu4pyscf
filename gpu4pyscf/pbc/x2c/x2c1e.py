#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#

'''
One-electron X2C for extended systems

This module is experimental and under active development.
APIs may change significantly in future releases.
'''

__all__ = [
    'sfx2c1e', 'sfx2c'
]

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.x2c import x2c as mol_x2c
from gpu4pyscf.x2c.x2c import _block_diag, _sigma_dot
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf import ghf, kghf
from gpu4pyscf.pbc.df import aft, rsdf_builder, ft_ao, int3c2e
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.gto.mole import SortedGTO, extract_pgto_params
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, hermi_triu, empty_aligned)
from gpu4pyscf.__config__ import props as gpu_specs

def sfx2c1e(mf):
    if isinstance(mf, _X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinFreeX2CHelper(mf.mol)
        else:
            assert isinstance(mf.with_x2c, SpinFreeX2CHelper)
        return mf
    return lib.set_class(SFX2C1E_SCF(mf), (SFX2C1E_SCF, mf.__class__))

sfx2c = sfx2c1e

# A tag to label the derived SCF class
class _X2C_SCF:
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.with_x2c:
            self.with_x2c.dump_flags(verbose)
        return self

    def reset(self, cell=None):
        self.with_x2c.reset(cell)
        return super().reset(cell)

class SFX2C1E_SCF(_X2C_SCF):
    __name_mixin__ = 'sfX2C1e'

    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinFreeX2CHelper(mf.mol)

    def get_hcore(self, cell=None, kpts=None, kpt=None):
        if cell is None: cell = self.cell
        if kpts is None:
            if isinstance(self, khf.KSCF):
                kpts = self.kpts
            elif kpt is None:
                kpts = self.kpt
            else:
                kpts = kpt
        if self.with_x2c:
            hcore = self.with_x2c.get_hcore(cell, kpts)
            if isinstance(self, kghf.KGHF):
                hcore = [_block_diag(h) for h in hcore]
            elif isinstance(self, ghf.GHF):
                hcore = _block_diag(hcore)
            return hcore
        else:
            return super(_X2C_SCF, self).get_hcore(cell, kpts)

def x2c1e_gscf(mf):
    raise NotImplementedError
    assert isinstance(mf, (ghf.GHF, kghf.KGHF))
    if isinstance(mf, _X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2C1EHelper(mf.mol)
        else:
            assert isinstance(mf.with_x2c, SpinOrbitalX2C1EHelper)
        return mf
    return lib.set_class(X2C1E_GSCF(mf), (X2C1E_GSCF, mf.__class__))

class X2C1E_GSCF(_X2C_SCF):
    __name_mixin__ = 'X2C1e'

    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinOrbitalX2C1EHelper(mf.cell)

    def get_hcore(self, cell=None, kpts=None, kpt=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            if isinstance(self, khf.KSCF):
                kpts = self.kpts
            elif kpt is None:
                kpts = self.kpt
            else:
                kpts = kpt

        if self.with_x2c is not None:
            hcore = self.with_x2c.get_hcore(cell, kpts)
            return hcore
        else:
            return super(_X2C_SCF).get_hcore(cell, kpts)

class PBCX2CHelper(mol_x2c.X2CHelperBase):

    approx = mol_x2c.X2CHelperBase.approx
    xuncontract = mol_x2c.X2CHelperBase.xuncontract
    basis = mol_x2c.X2CHelperBase.basis

    dump_flags = mol_x2c.X2CHelperBase.dump_flags
    get_xmol = mol_x2c.X2CHelperBase.get_xmol

    def __init__(self, cell, kpts=None):
        self.cell = cell
        mol_x2c.X2C.__init__(self, cell)

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        return self

class SpinFreeX2CHelper(PBCX2CHelper):
    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if cell.has_ecp():
            raise NotImplementedError
        assert '1E' in self.approx.upper()

        c = lib.param.LIGHT_SPEED
        xcell = self.get_xmol(cell)

        is_single_kpt = False
        if kpts is None:
            kpts = self.kpts
            kpts_in_bvkcell = True
        else:
            is_single_kpt = kpts.ndim == 1
            kpts = kpts.reshape(1, 3)
            kpts_in_bvkcell = len(kpts) == len(self.kpts)
        assert kpts_in_bvkcell
        bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1,3), bound_by_supmol=True)

        t = int1e.int1e_kin(xcell, kpts, bvk_kmesh, sort_output=False)
        s = int1e.int1e_ovlp(xcell, kpts, bvk_kmesh, sort_output=False)
        v = _get_pnucp(xcell, kpts, bvk_kmesh, intor='nuc')
        w = _get_pnucp(xcell, kpts, bvk_kmesh, intor='pnucp')

        h1 = []
        if 'ATOM' in self.approx.upper():
            x_no_pbc = mol_x2c._atomic_1e_x(xcell)
            for tk, vk, wk, sk in zip(t, v, w, s):
                h1.append(mol_x2c._get_hcore_fw(tk, vk, wk, sk, x_no_pbc, c))
        else:
            for tk, vk, wk, sk in zip(t, v, w, s):
                h1.append(mol_x2c._x2c1e_get_hcore(tk, vk, wk, sk, c))

        h1 = cp.stack(h1)
        if h1.dtype == np.complex128:
            nkpts, nao = s.shape[:2]
            h1 = h1.view(np.float64).reshape(nkpts, nao, nao, 2)
            h1 = h1.transpose(3,0,1,2).reshape(2*nkpts,nao,nao)
            h1 = mol_x2c._recontract_matrix(xcell, h1)
            h1 = h1.reshape(2,nkpts,nao,nao).transpose(1,2,3,0)
            h1 = h1.reshape(nkpts,nao,nao*2).view(np.complex128)
        else:
            h1 = mol_x2c._recontract_matrix(xcell, h1)

        if is_single_kpt:
            h1 = h1[0]
        return h1

    def get_xmat(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if cell.has_ecp():
            raise NotImplementedError
        assert '1E' in self.approx.upper()

        c = lib.param.LIGHT_SPEED
        xcell = self.get_xmol(cell)

        is_single_kpt = False
        if kpts is None:
            kpts = self.kpts
            kpts_in_bvkcell = True
        else:
            is_single_kpt = kpts.ndim == 1
            kpts = kpts.reshape(1, 3)
            kpts_in_bvkcell = len(kpts) == len(self.kpts)
        assert kpts_in_bvkcell
        bvk_kmesh = kpts_to_kmesh(cell, kpts.reshape(-1,3), bound_by_supmol=True)

        x = []
        if 'ATOM' in self.approx.upper():
            x = x_no_pbc = mol_x2c._atomic_1e_x(xcell)
            if not is_single_kpt:
                x = cp.repeat(x_no_pbc[None], len(kpts), axis=0)
        else:
            t = int1e.int1e_kin(xcell, kpts, bvk_kmesh, sort_output=False)
            s = int1e.int1e_ovlp(xcell, kpts, bvk_kmesh, sort_output=False)
            v = _get_pnucp(xcell, kpts, bvk_kmesh, intor='nuc')
            w = _get_pnucp(xcell, kpts, bvk_kmesh, intor='pnucp')
            for tk, vk, wk, sk in zip(t, v, w, s):
                x.append(mol_x2c._x2c1e_xmatrix(tk, vk, wk, sk, c))
            if is_single_kpt:
                x = x[0]
        return cp.asarray(x)

class SpinOrbitalX2C1EHelper(PBCX2CHelper):

    def get_hcore(self, cell=None, kpts=None):
        raise NotImplementedError

def _get_pnucp(cell, kpts=None, bvk_kmesh=None, intor='pnucp'):
    assert isinstance(cell, SortedGTO)
    cell_exps, cs = extract_pgto_params(cell, 'diffuse')
    omega = 0.3

    is_single_kpt = kpts is not None and kpts.ndim == 1
    is_gamma_point = kpts is None or is_zero(kpts)
    if is_single_kpt:
        kpts = kpts.reshape(1, 3)
    if is_gamma_point:
        bvk_kmesh = np.ones(3, dtype=int)
        bvk_ncells = 1
    elif bvk_kmesh is None:
        bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        bvk_ncells = np.prod(bvk_kmesh)

    ###############################################
    # SR part
    fakenuc = aft._fake_nuc(cell, with_pseudo=False)
    int3c2e_opt = int3c2e.SRInt3c2eOpt(
        cell, fakenuc, omega=-omega, bvk_kmesh=bvk_kmesh).build()
    charges = -cp.asarray(cell.atom_charges(), dtype=np.float64)

    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)

    if intor == 'nuc':
        kern = libpbc.PBCcontract_int3c2e_auxvec
        nsp_per_block, gout_stride, shm_size = int3c2e.int3c2e_scheme(
            cache_cart_idx=True, gout_width=29, gout_ndim='ij', deriv=(0,0,0))
    elif intor == 'pnucp':
        kern = libpbc.PBCcontract_int3c2e_pvp_auxvec
        nsp_per_block, gout_stride, shm_size = int3c2e.int3c2e_scheme(
            cache_cart_idx=True, gout_width=25, gout_ndim='ij', deriv=(1,1,0))
    else:
        raise NotImplementedError
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, 1000000)[0]

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    ksh_offsets = cp.asarray(l_ctr_aux_offsets, dtype=np.int32)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(workers * int3c2e.POOL_SIZE*(int3c2e.MAX_IMGS_PER_TASK+2) + 1,
                    dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned((workers, int3c2e.POOL_SIZE*16), np.int32, alignment=128)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    img_idx = cp.asarray(int3c2e_opt.img_idx)
    img_offsets = cp.asarray(int3c2e_opt.img_offsets)

    nao = cell.nao
    wj = cp.zeros((nao, bvk_ncells, nao))
    err = kern(
        ctypes.cast(wj.data.ptr, ctypes.c_void_p),
        ctypes.cast(charges.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs),
        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
        ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
        ctypes.cast(head.data.ptr, ctypes.c_void_p),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(bas_ij_idx)),
        ctypes.c_int(len(ksh_offsets) - 1),
        ctypes.c_int(auxcell.nbas),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff))
    if err != 0:
        raise RuntimeError('contract_int3c2e_auxvec failed')
    pool = task_pool = head = None

    ###############################################
    # LR part with AFT
    ke_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, _, kws = cell.get_Gv_weights(mesh)
    ZG = aft._get_ZSI(cell, mesh).conj()
    ZG *= rsdf_builder._weighted_coulG_LR(cell, Gv, omega, kws)

    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    cell = ft_opt.cell
    assert cell is int3c2e_opt.cell

    if intor == 'nuc':
        kern = libpbc.contract_ft_aopair
        nsp_per_block, gout_stride, shm_size = ft_ao.ft_ao_scheme(
            deriv=(0,0), nGv_per_block=16)
    elif intor == 'pnucp':
        kern = libpbc.contract_ft_pdotp
        nsp_per_block, gout_stride, shm_size = ft_ao.ft_ao_scheme(
            deriv=(1,1), nGv_per_block=16)
    else:
        raise NotImplementedError
    lmax = cell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(ft_opt.bas_ij_cache, 16)

    aft_envs = ft_opt.aft_envs
    head = cp.empty(1, dtype=np.int32)

    ngrids = len(Gv)
    GvT = cp.asarray(Gv.T.ravel())
    pair_blocks = len(shl_pair_offsets) - 1
    compressing = 0
    err = kern(
        ctypes.cast(wj.data.ptr, ctypes.c_void_p),
        ctypes.cast(ZG.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs),
        ctypes.cast(head.data.ptr, ctypes.c_void_p),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(pair_blocks),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(compressing))
    if err != 0:
        raise RuntimeError('contract_ft_pdotp kernel failed')

    if is_gamma_point:
        if bvk_ncells != 1:
            wj = wj.sum(axis=1)[None]
        else:
            wj = wj.transpose(1,0,2)
    else:
        nkpts = len(kpts)
        expLk = cp.exp(1j*asarray(ft_opt.bvkmesh_Ls).dot(asarray(kpts).T))
        expLkz = expLk.view(np.float64).reshape(bvk_ncells,nkpts,2)
        wj = contract('Lkz,pLq->kpqz', expLkz, wj)
        wj = wj.view(np.complex128)[:,:,:,0]
    wj = hermi_triu(wj)

    # x2c hcore is constructed with un-contracted basis sets.
    # The transformation cell.apply_CT_mat_C(wj) is not needed.

    if is_single_kpt:
        wj = wj[0]
    return wj
