# Copyright 2024-2026 The PySCF Developers. All Rights Reserved.
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
Periodic 3-center 2-electron short-range Coulomb integral helper functions
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf.gto import (
    ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_COEFF, PTR_EXP, PTR_COORD,
    conc_env)
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, transpose_sum, ndarray, empty_aligned, hermi_triu,
    get_avail_mem)
from gpu4pyscf.lib.utils import splits_by_blocksize, nearest_power2
from gpu4pyscf.gto.mole import (
    groupby, PTR_BAS_COORD, extract_pgto_params, SortedCell,
    PBCIntEnvVars, _scale_sp_ctr_coeff)
from gpu4pyscf.scf.jk import SHM_SIZE
from gpu4pyscf.df.int3c2e_bdiv import (
    get_ao_pair_loc, argsort_aux, _split_l_ctr_pattern, libvhf_rys,
    int3c2e_scheme as mol_int3c2e_scheme)
from gpu4pyscf.pbc.df.ft_ao import (
    libpbc, most_diffuse_pgto, FTOpt, ft_ao_scheme)
from gpu4pyscf.pbc.df.int2c2e import _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.pbc.tools.k2gamma import double_translation_indices
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'sr_aux_e2', 'SRInt3c2eOpt'
]

libpbc.bvk_ovlp_img_counts.restype = ctypes.c_int
libpbc.bvk_ovlp_img_idx.restype = ctypes.c_int
libpbc.PBCsr_int3c2e_latsum23.restype = ctypes.c_int
libpbc.PBCcontract_int3c2e_dm.restype = ctypes.c_int
libpbc.PBCcontract_int3c2e_auxvec.restype = ctypes.c_int
libpbc.PBCpair_recontraction_info.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
THREADS = 256
POOL_SIZE = 16384
MAX_IMGS_PER_TASK = 31
GOUT_WIDTH = 54

def sr_aux_e2(cell, auxcell, omega, kpts=None, bvk_kmesh=None, j_only=False):
    r'''
    Short-range 3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    from gpu4pyscf.pbc.df.rsdf_builder import _unpack_cderi_v2
    is_gamma_point = kpts is None or is_zero(kpts)
    if kpts is not None and kpts.ndim == 1: # single k-point
        assert is_gamma_point

    if bvk_kmesh is None:
        if j_only:
            # Coulomb integrals can be converged within a smaller bvk cell.
            bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        else:
            # Remote images may contribute to certain k-point mesh, contributing
            # to the finite-size effects in HFX. For sufficiently large number of
            # kpts, the truncation radius cell.rcut may cause finite-size errors.
            # Use a large radius to generate MP kmesh.
            bvk_kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut*10,
                                      bound_by_supmol=False)

    nao = cell.nao
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)

    eval_j3c, batches, _ = int3c2e_opt.int3c2e_evaluator(cart=cell.cell.cart)
    recontract, pair_address = _create_pair_recontractor(
        cell, batches, cell.cell.cart, bvk_ncells)
    aux_coeff = auxcell.ctr_coeff
    naux_cart, naux = aux_coeff.shape
    j3c = eval_j3c()
    if is_gamma_point or j_only:
        j3c = j3c.sum(axis=1)
        cderi = np.zeros((naux, len(pair_address)))
    else:
        cderi = np.zeros((bvk_ncells*naux, len(pair_address)))
    npair = j3c.shape[0]
    j3c = j3c.reshape(-1, naux_cart).dot(aux_coeff)
    host_j3c = j3c.reshape(npair, -1).get()
    recontract(0, cderi, host_j3c)
    j3c = cp.asarray(cderi)
    cderi = None
    if is_gamma_point:
        out = cp.zeros((nao, nao, naux))
        i, j = divmod(pair_address, nao*bvk_ncells)
        out[i, j] = j3c.T
        out[j, i] += j3c.T

    elif j_only:
        bvkmesh_Ls = cp.asarray(int3c2e_opt.bvkmesh_Ls)
        kpts = cp.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*bvkmesh_Ls.dot(kpts.T))
        conj_mapping = cp.asarray(
            conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh), dtype=np.int32)
        nkpts = len(kpts)
        out = _unpack_cderi_v2(j3c, pair_address, np.arange(nkpts),
                               conj_mapping, expLk, nao, axis=1)
        out = out.transpose(0,2,3,1)

    else:
        j3c = j3c.reshape(bvk_ncells, naux, len(pair_address))
        bvkmesh_Ls = cp.asarray(int3c2e_opt.bvkmesh_Ls)
        kpts = cp.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*bvkmesh_Ls.dot(kpts.T))
        nL, nkpts = expLk.shape
        conj_mapping = cp.asarray(
            conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh), dtype=np.int32)

        axis = 0 # Transform index i
        expLk_conjz = expLk.conj().view(np.float64).reshape(nL,nkpts,2)
        j3c = contract('Lqt,LKz->Kqtz', j3c, expLk_conjz)
        j3c = j3c.view(np.complex128)[...,0]
        out = cp.empty((nkpts,nkpts,naux,nao,nao), dtype=np.complex128)
        kk_conserv = double_translation_indices(int3c2e_opt.bvk_kmesh)
        for k in range(nkpts):
            ki_idx, kj_idx = np.where(kk_conserv == k)
            out[k] = _unpack_cderi_v2(j3c[k], pair_address, kj_idx,
                                      conj_mapping, expLk, nao, axis)
        j3c = None

        # k=ijk_conserv[i,j] provides: -i + j - k = 2n\pi
        # therefore, i=ijk_conserv[k,j]
        ijk_conserv = double_translation_indices(int3c2e_opt.bvk_kmesh)
        if axis == 0:
            #for ki in range(nkpts):
            #    for kj in range(nkpts):
            #        out[ki,kj] += j3c[ijk_conserv[ki,kj],ki]
            #        => order_KI = ijk_conserv[ki,kj] * nkpts + ki
            order = (ijk_conserv * nkpts + cp.arange(nkpts)[:,None]).ravel()
        else:
            #for ki in range(nkpts):
            #    for kj in range(nkpts):
            #        out[ki,kj] = j3c[ijk_conserv[ki,kj],kj]
            #        => order_KJ = ijk_conserv[ki,kj] * nkpts + kj
            order = (ijk_conserv * nkpts + cp.arange(nkpts)).ravel()
        out = out.reshape(nkpts**2, -1)[order]
        out = out.reshape(nkpts, nkpts, naux, nao, nao).transpose(0,1,3,4,2)

    if is_gamma_point and kpts is not None:
        if j_only:
            out = out[None]
        else:
            out = out[None,None]
    return out

def fill_triu_bvk(a, nao, bvk_kmesh, pair_address=None, conj_mapping=None, bvk_axis=0):
    '''Perform
    a[j,conj_mapping[L],i] = a[i,L,j]
    or
    a[conj_mapping[L],j,i] = a[L,i,j]
    '''
    assert a.flags.c_contiguous
    assert a.dtype == np.float64

    if conj_mapping is None:
        conj_mapping = conj_images_in_bvk_cell(bvk_kmesh)
    conj_mapping = cp.asarray(conj_mapping, dtype=np.int32)
    bvk_ncells = np.prod(bvk_kmesh)

    if bvk_axis == 0:
        assert a.size == nao*bvk_ncells*nao
        assert pair_address is None
        err = libpbc.fill_bvk_triu_axis0(
            ctypes.cast(a.data.ptr, ctypes.c_void_p),
            ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(bvk_ncells))
        if err != 0:
            raise RuntimeError('fill_bvk_triu failed')
    else:
        assert bvk_axis == 1
        assert pair_address is not None
        if a.ndim == 1:
            naux = 1
            a = a[:,None]
        else:
            naux = a.shape[-1]
            a = a.reshape(-1, naux)
        assert a.shape[0] == nao*bvk_ncells*nao
        err = libpbc.fill_bvk_triu(
            ctypes.cast(a.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_address.data.ptr, ctypes.c_void_p),
            ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(pair_address)),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nao), ctypes.c_int(naux))
        if err != 0:
            raise RuntimeError('fill_bvk_triu failed')
    return a

class SRInt3c2eOpt:
    def __init__(self, cell, auxcell, omega, bvk_kmesh=None):
        self.omega = abs(omega)
        self.cell = cell
        self.auxcell = auxcell

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh

        self.rcut = None
        self._mesh = None
        self._int3c2e_envs = None
        self.bvkcell = None
        self.bvkmesh_Ls = None
        self.bvk_auxcell = None
        self.bas_ij_cache = None
        self.img_idx = None
        self.img_offsets = None
        self.dd_ft_opt = None

    def build(self, separate_dd=False):
        """Build pair and image lists, optionally separating diffuse pairs.

        With separate_dd=True, the SR evaluator contains only compact pairs;
        dd_ft_opt holds the complementary Fourier-transform pair list.
        """
        auxcell = self.auxcell = SortedCell.from_cell(self.auxcell)
        assert auxcell.uniq_l_ctr[:,0].max() <= L_AUX_MAX

        cell = self.cell = SortedCell.from_cell(
            self.cell, decontract=True, diffuse_cutoff=0.25)
        assert cell.uniq_l_ctr[:,0].max() <= LMAX

        omega = self.omega
        cell.omega = -omega
        auxcell.omega = -omega
        # Adjust the rcut because the default cell.rcut is estimated based on
        # overlap integrals
        self.auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, -omega, cell.precision*1e-3)

        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        self.bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
            bvk_auxcell = auxcell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
            bvk_auxcell = pbctools.super_cell(auxcell, bvk_kmesh, wrap_around=True)
            bvk_auxcell._bas[:,PTR_BAS_COORD] = bvk_auxcell._atm[bvk_auxcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell
        self.bvk_auxcell = bvk_auxcell

        if self.rcut is None:
            rcut = max(estimate_rcut(cell, auxcell, -omega).max(), cell.rcut)
            self.rcut = rcut
        Ls = asarray(bvkcell.get_lattice_Ls(rcut=self.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        logger.debug(cell, 'int3c2e_kernel omega = %g, rcut = %g, nimgs = %d',
                     omega, rcut, nimgs)

        _atm, _bas, _env = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            bvk_auxcell._atm, bvk_auxcell._bas, _scale_sp_ctr_coeff(bvk_auxcell))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas[bvkcell.nbas,PTR_EXP] - bvk_auxcell._bas[0,PTR_EXP]
        _bas[bvkcell.nbas:,PTR_BAS_COORD] += off
        ao_loc = bvkcell.ao_loc
        aux_loc = bvk_auxcell.ao_loc_nr(cart=True)
        ao_loc = cp.asarray(_conc_locs(ao_loc, aux_loc), dtype=np.int32)
        self._int3c2e_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs, _atm, _bas, _env, ao_loc, Ls)

        exps, coef = extract_pgto_params(bvkcell, 'diffuse')
        aux_exps, aux_coef = extract_pgto_params(bvk_auxcell, 'diffuse')
        self.diffuse_exps = cp.asarray(np.append(exps, aux_exps), dtype=np.float32)
        self.diffuse_coefs = cp.asarray(np.append(coef, aux_coef), dtype=np.float32)
        log_c = cp.log(self.diffuse_coefs)

        self.cutoff = cutoff = self.estimate_cutoff_with_penalty()
        log_cutoff = math.log(cutoff)

        nbas = cell.nbas
        img_counts = cp.zeros((nbas*bvk_ncells*nbas), dtype=np.uint32)
        symmetric = 1
        libpbc.bvk_ovlp_img_counts(
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            ctypes.byref(self._int3c2e_envs),
            ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff), ctypes.c_int(symmetric))

        mask = img_counts > 0
        dd_bas_ij_cache = {}
        if separate_dd:
            # Needs more tests to determine which scheme to use
            if 1:
                from gpu4pyscf.pbc.scf.rsjk import _search_diffuse_pairs
                mask = mask.reshape(nbas, bvk_ncells, nbas)
                pair_mask = _search_diffuse_pairs(cell, self.mesh)
                dd_mask = mask & pair_mask[:,None,:]
                # Exclude diffuse pairs from bas_ij_cache
                mask &= ~pair_mask[:,None,:]
            else:
                from gpu4pyscf.pbc.tools.pbc import mesh_to_ke
                exps, coef = extract_pgto_params(cell, 'compact')
                exps = cp.asarray(exps, dtype=np.float32)
                coef = cp.asarray(coef, dtype=np.float32)
                bas_ij_idx = cp.asarray(cp.where(mask.ravel())[0], dtype=cp.int64)
                npairs = len(bas_ij_idx)
                dressed_precision = cell.precision * max(1, 1e-2*cell.vol)
                Ecut = cp.empty(npairs, dtype=np.float32)
                err = libpbc.estimate_aft_Ecut1(
                    ctypes.cast(Ecut.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.byref(self._int3c2e_envs),
                    ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(coef.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(npairs),
                    ctypes.c_float(math.log(dressed_precision)))
                if err != 0:
                    raise RuntimeError('estimate_aft_Ecut kernel failed')
                ke_cutoff = mesh_to_ke(cell.lattice_vectors(), self.mesh).min()
                dd_mask = mask.copy()
                mask[bas_ij_idx[Ecut <= ke_cutoff]] = False
                dd_mask[bas_ij_idx[Ecut > ke_cutoff]] = False
                dd_mask = dd_mask.reshape(nbas, bvk_ncells, nbas)

        mask = mask.reshape(nbas, bvk_ncells, nbas)

        self.bas_ij_cache = bas_ij_cache = {}
        groups = len(cell.uniq_l_ctr)
        l_ctr_offsets = _counts_to_offsets(cell.l_ctr_counts)
        ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        img = cp.arange(bvk_ncells, dtype=np.uint32) * nbas
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.uint32)
            jsh = img[:,None] + cp.arange(jsh0, jsh1, dtype=np.uint32)
            bas_ij = ish[:,None,None] * (nbas*bvk_ncells) + jsh
            assert np.all(bas_ij < np.iinfo(np.uint32).max), "uint32 overflow"
            bas_ij = bas_ij.astype(np.uint32)
            sub_mask = mask[ish0:ish1,:,jsh0:jsh1]
            bas_ij_cache[i, j] = bas_ij[sub_mask]
            if separate_dd:
                sub_mask = dd_mask[ish0:ish1,:,jsh0:jsh1]
                dd_bas_ij_cache[i, j] = bas_ij[sub_mask]

        bas_ij_idx = cp.hstack(list(bas_ij_cache.values()), dtype=np.uint32)
        img_offsets = cp.empty(bas_ij_idx.size+1, dtype=np.uint32)
        img_counts[bas_ij_idx].cumsum(out=img_offsets[1:])
        img_offsets[0] = 0
        img_idx_size = img_offsets[-1].get()
        assert img_idx_size < 2**32
        img_idx = cp.zeros(img_idx_size, dtype=np.int32)
        if len(bas_ij_idx) > 0:
            libpbc.bvk_ovlp_img_idx(
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(bas_ij_idx)),
                ctypes.byref(self._int3c2e_envs),
                ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
        self.img_idx = img_idx
        self.img_offsets = img_offsets

        self.dd_ft_opt = None
        if separate_dd:
            bas_ij_idx = cp.hstack(list(dd_bas_ij_cache.values()), dtype=np.uint32)
            if len(bas_ij_idx) > 0:
                img_offsets = cp.empty(bas_ij_idx.size+1, dtype=np.uint32)
                img_counts[bas_ij_idx].cumsum(out=img_offsets[1:])
                img_offsets[0] = 0
                img_idx_size = img_offsets[-1].get()
                assert img_idx_size < 2**32
                img_idx = cp.zeros(img_idx_size, dtype=np.int32)
                libpbc.bvk_ovlp_img_idx(
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(len(bas_ij_idx)),
                    ctypes.byref(self._int3c2e_envs),
                    ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
                    ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff))
                self.dd_ft_opt = dd_ft_opt = FTOpt.from_intopt(self)
                dd_ft_opt.bas_ij_cache = dd_bas_ij_cache
                dd_ft_opt.img_idx = img_idx
                dd_ft_opt.img_offsets = img_offsets
                logger.debug(cell, 'Separated %d diffuse shell pairs', len(bas_ij_idx))
        return self

    @property
    def int3c2e_envs(self):
        _int3c2e_envs = self._int3c2e_envs
        if _int3c2e_envs is None or cp.cuda.device.get_device_id() == _int3c2e_envs.device:
            return self._int3c2e_envs
        return _int3c2e_envs.copy()
    @property
    def rys_envs(self):
        return self.int3c2e_envs

    @property
    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        cell = self.cell
        omega = self.omega
        ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
        mesh = cell.cutoff_to_mesh(ke_cutoff)
        mesh = cell.symmetrize_mesh(mesh)
        return mesh
    @mesh.setter
    def mesh(self, x):
        self._mesh = x

    def estimate_cutoff_with_penalty(self):
        cell = self.cell.cell
        auxcell = self.auxcell.cell
        vol = self.bvkcell.vol
        omega = self.omega
        aux_exp, _, aux_l = most_diffuse_pgto(auxcell)
        cell_exp, _, cell_l = most_diffuse_pgto(cell)
        if omega == 0:
            theta = 1./(1./(cell_exp*2) + 1./aux_exp)
        else:
            theta = 1./(1./(cell_exp*2) + 1./aux_exp + omega**-2)
        rcut = self.rcut
        lsum = cell_l * 2 + aux_l + 1
        rad = vol**(-1./3) * rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*rcut*lsum/(vol*theta) + surface
        cutoff = cell.precision / lattice_sum_factor
        logger.debug1(cell, 'int3c_kernel integral omega=%g theta=%g cutoff=%g',
                      omega, theta, cutoff)
        return cutoff

    def get_n_compact_pairs(self):
        '''Count Cartesian AO pairs in the built SR shell-pair cache.'''
        l = self.cell.uniq_l_ctr[:,0]
        nf = (l + 1) * (l + 2) // 2
        return sum(len(pairs) * int(nf[i]) * int(nf[j])
                   for (i, j), pairs in self.bas_ij_cache.items())

    def _split_bas_ij_idx(self, batch_size, pair_per_block):
        '''Split consecutive shell pairs into Cartesian AO-pair batches.
        batch_size is the target number of Cartesian AO pairs per batch.
        '''
        cell = self.cell
        if batch_size is None:
            return [cell.aggregate_shl_pairs(self.bas_ij_cache, pair_per_block)]

        # If int3c2e nsp_per_block is smaller than ft_aopair nsp_per_block,
        # multiple int3c2e batches can correspond to one ft_aopair batch.
        # An int3c2e batch boundary may cut through an ft_aopair batch,
        # causing SR and LR batch misaligned.
        nsp_per_block = ft_ao_scheme(cache_cart_idx=True)[0]
        nsp_per_block = np.maximum(pair_per_block, nsp_per_block)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            self.bas_ij_cache, nsp_per_block)

        ao_pair_loc = get_ao_pair_loc(
            cell.uniq_l_ctr[:,0], self.bas_ij_cache, cart=True)
        ao_pair_size_offsets = ao_pair_loc[shl_pair_offsets].get()
        splits = splits_by_blocksize(ao_pair_size_offsets, batch_size)
        batch_splits = shl_pair_offsets[splits].get()

        nbatches = len(splits) - 1
        batches = []
        for n in range(nbatches):
            p0, p1 = batch_splits[n:n+2]
            assert p0 != p1
            s0, s1 = splits[n:n+2]
            batches.append((bas_ij_idx[p0:p1], shl_pair_offsets[s0:s1+1]-p0))
        return batches

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          cart=None):
        '''Return the SR evaluator, primitive shell-pair batches and aux offsets.

        evaluate_j3c(shl_pair_batch_id, aux_batch_id, out=None) returns a
        GPU [primitive pair, image, aux] buffer for the selected batches.
        aux_offsets delimit each auxiliary batch in the sorted auxiliary basis.
        Pair recontraction is set up separately with _create_pair_recontractor.
        '''
        if self.bvkmesh_Ls is None:
            self.build()

        cell = self.cell
        auxcell = self.auxcell
        bvk_ncells = np.prod(self.bvk_kmesh)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            gout_width=GOUT_WIDTH, cache_cart_idx=True)
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

        uniq_l_ctr_aux = auxcell.uniq_l_ctr
        l_ctr_aux_offsets = _counts_to_offsets(auxcell.l_ctr_counts)
        # Split auxbasis in the unit cell
        if aux_batch_size is None:
            _aux_batch_size = POOL_SIZE // bvk_ncells
        else:
            _aux_batch_size = aux_batch_size
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, uniq_l_ctr_aux, _aux_batch_size)

        aux_loc = auxcell.ao_loc
        aux_groups, aux_offsets = _group_ksh_batches(
            l_ctr_aux_offsets, uniq_l_ctr_aux, aux_loc, aux_batch_size)

        pair_per_block = _get_shl_pair_per_block(np.diff(l_ctr_aux_offsets), bvk_ncells)
        bas_ij_batches = self._split_bas_ij_idx(ao_pair_batch_size, pair_per_block)
        shl_pair_batch_offsets = _counts_to_offsets(
            np.asarray([len(pairs) for pairs, _ in bas_ij_batches], dtype=np.int64))
        ao_pair_counts = _count_ao_pairs(cell, bas_ij_batches, cart, bvk_ncells)

        if cart is None:
            cart = cell.cell.cart

        l = cp.asarray(cell._bas[:,ANG_OF], dtype=np.int32)
        if cart:
            nf = (l + 1) * (l + 2) // 2
        else:
            nf = l * 2 + 1

        logger.debug1(self.cell, 'sp_batches = %d, ksh_batches = %d',
                      len(bas_ij_batches), len(aux_groups))
        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        c2s_pool = cp.empty((workers, THREADS*GOUT_WIDTH))
        int3c2e_envs = self.int3c2e_envs
        img_idx = cp.asarray(self.img_idx)
        img_offsets = cp.asarray(self.img_offsets)
        kern = libpbc.PBCsr_int3c2e_latsum23

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            bas_ij_idx, shl_pair_offsets = bas_ij_batches[shl_pair_batch_id]
            i, j = divmod(bas_ij_idx, cell.nbas*bvk_ncells)
            j = j % cell.nbas
            ao_pair_loc = _counts_to_offsets(nf[i] * nf[j])
            nao_pair = ao_pair_counts[shl_pair_batch_id]

            # Indexing the aux-basis within the first cell
            aux_group = aux_groups[aux_batch_id]
            aux_ao_offset = aux_group.ao_range[0]
            naux = aux_group.ao_range[1] - aux_ao_offset
            out = ndarray((nao_pair, bvk_ncells, naux), buffer=out)
            # The output buffer must be initialized because integral screening
            # based on SR integrals is performed in the kernel, and certain ~0
            # shell-tritets are not evaluated, leaving the output buffer untouched
            out.fill(0.)
            if out.size == 0:
                return out
            p0, p1 = shl_pair_batch_offsets[shl_pair_batch_id:shl_pair_batch_id+2]
            # The batch is contiguous in cache order. Keep absolute offsets
            # into the shared image array; no image copying is needed.
            batch_img_offsets = img_offsets[p0:p1+1]
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.c_double(-self.omega),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(c2s_pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(head.data.ptr, ctypes.c_void_p),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(len(aux_group.sub_batch_offsets) - 1),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_group.sub_batch_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(batch_img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(0), ctypes.c_int(aux_ao_offset),
                ctypes.c_int(auxcell.nbas), ctypes.c_int(naux),
                ctypes.c_int(not cart),
                ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel')
            return out

        return evaluate_j3c, bas_ij_batches, aux_offsets

    pair_and_diag_indices = FTOpt.pair_and_diag_indices

    def contract_dm(self, dm, kpts=None, hermi=0, sort_output=True):
        assert dm.shape[1] == self.cell.nao
        if self.bvkmesh_Ls is None:
            self.build()

        if hermi != 1:
            dm = transpose_sum(dm, inplace=False)
        if kpts is None:
            assert dm.dtype == np.float64
        elif is_zero(kpts):
            dm = cp.asarray(dm.real, order='C')
        else:
            expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(asarray(kpts).T))
            dm = contract('Lk,kpq->Lpq', expLk, dm)
            dm = cp.asarray(dm.real, order='C')
            dm *= 1./len(kpts)
        assert dm.dtype == np.float64
        assert dm.flags.c_contiguous

        cell = self.cell
        auxcell = self.auxcell

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            cache_cart_idx=True, gout_width=28, gout_ndim='k')
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bvk_ncells = len(self.bvkmesh_Ls)
        pair_per_block = _get_shl_pair_per_block(auxcell.l_ctr_counts, bvk_ncells)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            self.bas_ij_cache, pair_per_block)

        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = self.int3c2e_envs
        img_idx = cp.asarray(self.img_idx)
        img_offsets = cp.asarray(self.img_offsets)

        naux = auxcell.nao
        vj_aux = cp.zeros(naux)
        err = libpbc.PBCcontract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.c_double(-self.omega),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(auxcell.nbas),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm failed')
        if hermi == 1:
            vj_aux *= 2
        if sort_output:
            vj_aux = auxcell.apply_CT_dot(vj_aux)
        return vj_aux

    def contract_auxvec(self, auxvec, kpts=None, sort_output=True):
        assert auxvec.dtype == np.float64
        assert auxvec.ndim == 1
        assert len(auxvec) == self.auxcell.nao
        auxvec = cp.asarray(auxvec)
        if self.bvkmesh_Ls is None:
            self.build()

        cell = self.cell
        auxcell = self.auxcell
        bvk_ncells = len(self.bvkmesh_Ls)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            cache_cart_idx=True, gout_width=29, gout_ndim='ij')
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        pair_per_block = _get_shl_pair_per_block(auxcell.l_ctr_counts, bvk_ncells)
        bas_ij_idx = cell.aggregate_shl_pairs(self.bas_ij_cache, pair_per_block)[0]

        l_ctr_aux_offsets = _counts_to_offsets(auxcell.l_ctr_counts)
        ksh_offsets = cp.asarray(l_ctr_aux_offsets, dtype=np.int32)

        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = self.int3c2e_envs
        img_idx = cp.asarray(self.img_idx)
        img_offsets = cp.asarray(self.img_offsets)

        nao = cell.nao
        vj = cp.zeros((nao, bvk_ncells, nao))
        err = libpbc.PBCcontract_int3c2e_auxvec(
            ctypes.cast(vj.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.c_double(-self.omega),
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

        if kpts is None or is_zero(kpts):
            vj = vj.sum(axis=1)
            vj = vj.reshape(1,nao,nao)
        else:
            nkpts = len(kpts)
            expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(asarray(kpts).T))
            expLkz = expLk.view(np.float64).reshape(bvk_ncells,nkpts,2)
            vj = contract('Lkz,pLq->kpqz', expLkz, vj)
            vj = vj.view(np.complex128)[:,:,:,0]
        vj = hermi_triu(vj)
        if sort_output:
            vj = cell.apply_CT_mat_C(vj)
        return vj

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

def int3c2e_scheme(*, shm_size=SHM_SIZE, gout_width=None, gout_ndim='ijk',
                   deriv=None, cache_cart_idx=False):
    return mol_int3c2e_scheme(
        short_range=True, shm_size=shm_size, gout_width=gout_width,
        gout_ndim=gout_ndim, deriv=deriv, cache_cart_idx=cache_cart_idx)

# This modified rcut estimation function will be available in pyscf-2.8 or newer
# TODO: improve the rcut estimation for PBCsr_int3c2e_latsum23 kernel
def estimate_rcut(cell, auxcell, omega):
    '''Estimate rcut for 3c2e SR-integrals'''
    if cell.nbas == 0 or auxcell.nbas == 0:
        return np.zeros(1)

    if omega == 0:
        # No SR integrals in int3c2e if omega=0
        assert cell.dimension == 0
        return np.zeros(1)

    precision = cell.precision
    ak, ck, lk = most_diffuse_pgto(auxcell)

    # the most diffuse orbital basis
    cell_exps, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,ANG_OF]
    r2_cell = np.log(cs**2 / precision * 10**ls + 1e-200) / cell_exps
    ai_idx = r2_cell.argmax()
    ai = cell_exps[ai_idx]
    aj = cell_exps
    li = ls[ai_idx]
    lj = ls
    ci = cs[ai_idx]
    cj = cs

    aij = ai + aj
    lij = li + lj
    l3 = lij + lk
    theta = 1./(omega**-2 + 1./aij + 1./ak)
    norm_ang = ((2*li+1)*(2*lj+1))**.5/(4*np.pi)
    c1 = ci * cj * ck * norm_ang
    sfac = aij*aj/(aij*aj + ai*theta)
    fl = 2
    fac = 2**li*np.pi**2.5*c1 * theta**(l3-.5)
    rad = cell.vol**(-1./3) * cell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*cell.rcut/(cell.vol*theta) + surface
    fac *= lattice_sum_factor
    fac /= aij**(li+1.5) * ak**(lk+1.5) * aj**lj
    fac *= fl / precision

    r0 = cell.rcut  # initial guess
    r0 = (np.log(fac * (sfac*r0+1e-200)**(l3-1) + 1.) / (sfac*theta))**.5
    r0 = (np.log(fac * (sfac*r0+1e-200)**(l3-1) + 1.) / (sfac*theta))**.5
    rcut = r0
    return rcut

def _get_shl_pair_per_block(nksh_per_block, bvk_ncells):
    '''Add the BvK-cell constraint to the maximum number of shell pairs.'''
    max_nksh = int(nksh_per_block.max())
    max_pairs = POOL_SIZE // (max_nksh * bvk_ncells)
    if max_pairs < 1:
        raise RuntimeError(
            f'CUDA task pool is too small: POOL_SIZE={POOL_SIZE}, '
            f'max_nksh={max_nksh}, bvk_ncells={bvk_ncells}')
    # Limit to 8 triplets per block for better load balance
    return min(8, nearest_power2(max_pairs))

class _GroupedBatch:
    batch_range = None
    shl_range = None
    ao_range = None
    sub_batch_offsets = None
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def _group_ksh_batches(l_ctr_offsets, uniq_l_ctr, aux_loc, aux_batch_size=None):
    if aux_batch_size is None:
        aux_groups = [
            _GroupedBatch(
                batch_range = (0, len(l_ctr_offsets)-1),
                shl_range = (0, len(aux_loc) - 1),
                ao_range = (0, aux_loc[-1])
            )
        ]
        aux_offsets = np.array(aux_groups[0].ao_range)
    else:
        aux_loc_for_group = aux_loc[l_ctr_offsets]
        # Regroup aux batch into batches. So each kernel launch can handle more aux batches
        #TODO: group_splits = splits_by_blocksize(aux_loc_for_group, aux_batch_size)
        group_splits = range(len(l_ctr_offsets))
        aux_offsets = aux_loc_for_group[group_splits]
        aux_groups = [
            _GroupedBatch(
                batch_range = (i0, i1),
                shl_range = (l_ctr_offsets[i0], l_ctr_offsets[i1]),
                ao_range = (aux_loc_for_group[i0], aux_loc_for_group[i1])
            ) for i0, i1 in zip(group_splits[:-1], group_splits[1:])
        ]
    l_ctr_offsets = asarray(l_ctr_offsets, dtype=np.int32)
    for group in aux_groups:
        batch0, batch1 = group.batch_range
        sub_batches = l_ctr_offsets[batch0:batch1+1]
        group.sub_batch_offsets = sub_batches
    return aux_groups, aux_offsets

def _counts_to_offsets(counts):
    if isinstance(counts, cp.ndarray):
        out = cp.empty(len(counts)+1, dtype=counts.dtype)
    else:
        out = np.empty(len(counts)+1, dtype=counts.dtype)
    counts.cumsum(out=out[1:])
    out[0] = 0
    return out

def _create_pair_recontractor(cell, bas_ij_batches, cart, bvk_ncells=1):
    '''Create an ordered partial CDERI tensor from a primitive shell triangle.

    Preserve (i, image, j) orientation and half-weight diagonal primitive shell
    blocks. Unpacking must add the exchanged tensor, not copy a triangle.
    '''
    recontract_bas = cp.asnumpy(cell.recontract_bas)
    recontract_coef = cp.asnumpy(cell.recontract_coef)
    recontraction_idx = cp.asnumpy(cell.recontraction_idx)

    nprims = recontract_bas[:,NPRIM_OF]
    prim_offsets = _counts_to_offsets(nprims)
    recon_shell_idx = np.empty(cell.nbas, dtype=np.int32)
    recon_shell_idx[recontraction_idx] = np.repeat(
        np.arange(len(recontract_bas)), nprims)
    prim_id_within_shell = np.empty(cell.nbas, dtype=np.int32)
    prim_id_within_shell[recontraction_idx] = (
        np.arange(cell.nbas) - np.repeat(prim_offsets[:-1], nprims))

    l_ctr = recontract_bas[:,ANG_OF]
    if cart:
        nf_ctr = (l_ctr + 1) * (l_ctr + 2) // 2
    else:
        nf_ctr = l_ctr * 2 + 1
    nf_ctr = nf_ctr * recontract_bas[:,NCTR_OF]

    ao_loc = np.asarray(_counts_to_offsets(nf_ctr), dtype=np.int32)
    nao = int(ao_loc[-1])

    NOT_INITIALIZED = -1
    output_lut = np.full(nao**2*bvk_ncells, NOT_INITIALIZED, dtype=np.int32)
    pair_addresses = np.empty(nao**2*bvk_ncells, dtype=np.int32)

    nctr_max = recontract_bas[:,NCTR_OF].max()
    ao_pair_counts = _count_ao_pairs(cell, bas_ij_batches, cart, bvk_ncells)
    size = ao_pair_counts.max(initial=0) * nctr_max**2
    out_idx = np.empty(size, dtype=np.int32)
    coef = np.empty(size, dtype=np.float64)
    out_offsets = np.empty(ao_pair_counts.max(initial=0)+1, dtype=np.int32)
    out_offsets[0] = 0

    recontraction_params = []

    cderi_npairs = ctypes.c_int(0)
    for batch_id, (bas_ij_idx, _) in enumerate(bas_ij_batches):
        bas_ij_idx = cp.asnumpy(bas_ij_idx)
        inp_count = ao_pair_counts[batch_id]

        out_count = libpbc.PBCpair_recontraction_info(
            out_idx.ctypes, out_offsets.ctypes, coef.ctypes,
            pair_addresses.ctypes, output_lut.ctypes,
            ctypes.byref(cderi_npairs),
            bas_ij_idx.ctypes, ctypes.c_int(len(bas_ij_idx)),
            recon_shell_idx.ctypes, prim_id_within_shell.ctypes,
            recontract_bas.ctypes, recontract_coef.ctypes,
            ao_loc.ctypes,
            ctypes.c_int(cell.nbas), ctypes.c_int(bvk_ncells),
            ctypes.c_int(nao), ctypes.c_int(cart))

        recontraction_params.append(
            (out_idx[:out_count].copy(), out_offsets[:inp_count+1].copy(),
             coef[:out_count].copy()))

    pair_addresses = pair_addresses[:cderi_npairs.value].copy()

    def recontract(batch_id, cderi, j3c):
        """Scatter [primitive_pair, aux] data into shared CDERI.
        """
        assert j3c.dtype == cderi.dtype
        npair, naux = j3c.shape
        out_idx, out_offsets, coef = recontraction_params[batch_id]
        assert len(out_offsets) == npair + 1
        if j3c.dtype == np.float64:
            kern = libpbc.PBCrecontract_cderi
        else:
            kern = libpbc.PBCzrecontract_cderi
        kern(cderi.ctypes, j3c.ctypes,
             out_idx.ctypes, out_offsets.ctypes, coef.ctypes,
             ctypes.c_int(naux), ctypes.c_int(cderi.shape[1]),
             ctypes.c_int(npair))
        return cderi

    return recontract, pair_addresses

def _count_ao_pairs(cell, bas_ij_batches, cart, ncells=1):
    l = cp.asarray(cell._bas[:,ANG_OF])
    if cart:
        nf = (l + 1) * (l + 2) // 2
    else:
        nf = l * 2 + 1
    bvk_nbas = cell.nbas * ncells
    sizes = []
    for bas_ij_idx, _ in bas_ij_batches:
        i, j = divmod(bas_ij_idx, bvk_nbas)
        j = j % cell.nbas
        sizes.append(nf[i].dot(nf[j]))
    return cp.array(sizes, dtype=np.int32).get()
