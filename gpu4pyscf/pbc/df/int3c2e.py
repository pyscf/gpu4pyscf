# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf.gto import ATOM_OF, ANG_OF, PTR_EXP, PTR_COORD, conc_env
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools.k2gamma import (
    translation_vectors_for_kmesh, double_translation_indices)
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, transpose_sum, ndarray
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.gto.mole import (
    groupby, PTR_BAS_COORD, extract_pgto_params, SortedCell,
    PBCIntEnvVars, _scale_sp_ctr_coeff)
from gpu4pyscf.scf.jk import _nearest_power2, SHM_SIZE
from gpu4pyscf.df.int3c2e_bdiv import get_ao_pair_loc, argsort_aux, _split_l_ctr_pattern
from gpu4pyscf.pbc.df.ft_ao import libpbc, most_diffuse_pgto, FTOpt
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.pbc.df.int2c2e import _estimate_sr_2c2e_rcut
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'sr_aux_e2', 'SRInt3c2eOpt'
]

libpbc.bvk_ovlp_img_counts.restype = ctypes.c_int
libpbc.bvk_ovlp_img_idx.restype = ctypes.c_int
libpbc.PBCsr_int3c2e_latsum23.restype = ctypes.c_int
libpbc.PBCcontract_int3c2e_dm.restype = ctypes.c_int
libpbc.PBCcontract_int3c2e_auxvec.restype = ctypes.c_int

LMAX = 4
L_AUX_MAX = 6
THREADS = 256
POOL_SIZE = 262144

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
    naux = auxcell.nao
    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega, bvk_kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)

    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator()[:2]
    pair_address = int3c2e_opt.pair_and_diag_indices()[0]
    aux_coeff = auxcell.ctr_coeff
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None
    j3c = eval_j3c()

    if is_gamma_point:
        j3c = j3c[:,:,0].dot(aux_coeff)
        out = cp.zeros((nao, nao, naux))
        i, j = divmod(pair_address, nao*bvk_ncells)
        out[j, i] = out[i, j] = j3c

    elif j_only:
        j3c = j3c.sum(axis=2).dot(aux_coeff)
        bvkmesh_Ls = cp.asarray(int3c2e_opt.bvkmesh_Ls)
        kpts = cp.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*bvkmesh_Ls.dot(kpts.T))
        conj_mapping = conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh)
        nkpts = len(kpts)
        out = _unpack_cderi_v2(j3c.T, pair_address, np.arange(nkpts),
                               conj_mapping, expLk, nao, axis=1)
        out = out.transpose(0,2,3,1)

    else:
        j3c = contract('tpL,pq->tqL', j3c, aux_coeff)
        bvkmesh_Ls = cp.asarray(int3c2e_opt.bvkmesh_Ls)
        kpts = cp.asarray(kpts).reshape(-1, 3)
        expLk = cp.exp(1j*bvkmesh_Ls.dot(kpts.T))
        nL, nkpts = expLk.shape
        conj_mapping = conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh)

        axis = 0 # Transform index i
        expLk_conjz = expLk.conj().view(np.float64).reshape(nL,nkpts,2)
        j3c = contract('tqL,LKz->Kqtz', j3c, expLk_conjz)
        j3c = j3c.view(np.complex128)[...,0]
        out = cp.empty((nkpts,nkpts,naux,nao,nao), dtype=np.complex128)
        conj_mapping = conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh)
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
            order = (ijk_conserv * nkpts + np.arange(nkpts)[:,None]).ravel()
        else:
            #for ki in range(nkpts):
            #    for kj in range(nkpts):
            #        out[ki,kj] = j3c[ijk_conserv[ki,kj],kj]
            #        => order_KJ = ijk_conserv[ki,kj] * nkpts + kj
            order = (ijk_conserv * nkpts + np.arange(nkpts)).ravel()
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
        assert omega < 0
        self.omega = -omega
        self.cell = SortedCell.from_cell(
            cell, allow_replica=True, allow_split_seg_contraction=False)
        assert self.cell.uniq_l_ctr[:,0].max() <= LMAX
        self.auxcell = SortedCell.from_cell(
            auxcell, allow_replica=True, allow_split_seg_contraction=False)
        assert self.auxcell.uniq_l_ctr[:,0].max() <= L_AUX_MAX
        self.cell.omega = omega
        self.auxcell.omega = omega
        # Adjust the rcut because the default cell.rcut is estimated based on
        # overlap integrals
        self.auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, cell.precision*1e-3)

        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh

        self.rcut = None
        self._int3c2e_envs = None
        self.bas_ij_cache = None
        self.bvkcell = None
        self.bvk_auxcell = None
        self.bvkmesh_Ls = None

    def build(self):
        cell = self.cell
        auxcell = self.auxcell
        assert all(self.cell.recontract_coef == 1.), \
                'int3c2e for general-contraction basis not supported'

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
            rcut = max(estimate_rcut(cell, auxcell, self.omega).max(), cell.rcut, auxcell.rcut)
            self.rcut = rcut
        Ls = asarray(bvkcell.get_lattice_Ls(rcut=self.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        logger.debug(cell, 'int3c2e_kernel rcut = %g, nimgs = %d', rcut, nimgs)

        _atm, _bas, _env = conc_env(
            bvkcell._atm, bvkcell._bas, _scale_sp_ctr_coeff(bvkcell),
            bvk_auxcell._atm, bvk_auxcell._bas, _scale_sp_ctr_coeff(bvk_auxcell))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas[bvkcell.nbas,PTR_EXP] - bvk_auxcell._bas[0,PTR_EXP]
        _bas[bvkcell.nbas:,PTR_BAS_COORD] += off
        ao_loc = bvkcell.ao_loc
        aux_loc = bvk_auxcell.ao_loc
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
        libpbc.bvk_ovlp_img_counts(
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            ctypes.byref(self._int3c2e_envs),
            ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff), ctypes.c_int(1))

        mask = img_counts.reshape(nbas, bvk_ncells, nbas) > 0
        self.bas_ij_cache = bas_ij_cache = {}
        groups = len(cell.uniq_l_ctr)
        l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
        ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        bas_ij_idx = []
        img = cp.arange(bvk_ncells, dtype=np.uint32) * nbas
        for i, j in ij_tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            ish = cp.arange(ish0, ish1, dtype=np.uint32)
            jsh = img[:,None] + cp.arange(jsh0, jsh1, dtype=np.uint32)
            bas_ij = ish[:,None,None] * (nbas*bvk_ncells) + jsh
            sub_mask = mask[ish0:ish1,:,jsh0:jsh1]
            bas_ij = bas_ij[sub_mask]
            bas_ij_cache[i, j] = bas_ij
            bas_ij_idx.append(bas_ij)

        bas_ij_idx = cp.hstack(bas_ij_idx, dtype=np.uint32)
        img_counts = img_counts[bas_ij_idx]
        img_offsets = cp.empty(img_counts.size+1, dtype=np.uint32)
        img_counts.cumsum(out=img_offsets[1:])
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
        self.img_idx = img_idx
        self.img_offsets = img_offsets
        return self

    @property
    def int3c2e_envs(self):
        _int3c2e_envs = self._int3c2e_envs
        if _int3c2e_envs is None or cp.cuda.device.get_device_id() == _int3c2e_envs.device:
            return self._int3c2e_envs
        return _int3c2e_envs.copy()

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

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          cart=None, bas_ij_aggregated=None):
        if self.bvkmesh_Ls is None:
            self.build()

        cell = self.cell
        auxcell = self.auxcell
        bvk_ncells = np.prod(self.bvk_kmesh)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(gout_width=54)
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        if bas_ij_aggregated is None:
            bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
                self.bas_ij_cache, nsp_per_block[0])
        else:
            bas_ij_idx, shl_pair_offsets = bas_ij_aggregated

        # For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
        # addresses of first element for the contracted pair-GTOs. In each
        # shell-pair, there are nfij elements. Note, the nfij elements are
        # sorted as [nfj,nfi] (in F-order).
        if cart is None:
            cart = cell.cell.cart
        ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], self.bas_ij_cache, cart)

        if ao_pair_batch_size is None:
            pair_splits = [0, len(shl_pair_offsets)-1]
            ao_pair_offsets = [0, ao_pair_loc[-1].get()]
        else:
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
            pair_splits = splits_by_blocksize(ao_pair_offsets, ao_pair_batch_size)
            ao_pair_offsets = ao_pair_offsets[pair_splits]
        shl_pair_offsets = cp.asnumpy(shl_pair_offsets)

        # Split auxbasis in the unit cell than the bvk-cell
        aux_loc = auxcell.ao_loc
        uniq_l_ctr_aux = auxcell.uniq_l_ctr
        l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
        if aux_batch_size is None:
            ksh_offsets_cpu = l_ctr_aux_offsets
            aux_splits = [0, len(ksh_offsets_cpu)-1]
        else:
            l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
                l_ctr_aux_offsets, uniq_l_ctr_aux, aux_batch_size)
            ksh_offsets_cpu = l_ctr_aux_offsets
            aux_splits = range(len(ksh_offsets_cpu))
        aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
        aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)

        ksh_idx = _aggregate_bas_idx(
            l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, auxcell.nbas)[1]
        ksh_idx += self.bvkcell.nbas
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)
        shl_pair_batches = len(ao_pair_offsets) - 1
        aux_batches = len(aux_offsets) - 1
        logger.debug1(self.cell, 'sp_batches = %d, ksh_batches = %d',
                      shl_pair_batches, aux_batches)
        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
        int3c2e_envs = self.int3c2e_envs
        img_idx = cp.asarray(self.img_idx)
        img_offsets = cp.asarray(self.img_offsets)
        kern = libpbc.PBCsr_int3c2e_latsum23

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            pair_split0 = pair_splits[shl_pair_batch_id]
            pair_split1 = pair_splits[shl_pair_batch_id+1]
            shl_pair0 = shl_pair_offsets[pair_split0]
            shl_pair1 = shl_pair_offsets[pair_split1]
            ao_pair_offset = ao_pair_offsets[shl_pair_batch_id]
            nao_pair = ao_pair_offsets[shl_pair_batch_id+1] - ao_pair_offset

            # Indexing the aux-basis within the first cell
            aux_split0 = aux_splits[aux_batch_id]
            aux_split1 = aux_splits[aux_batch_id+1]
            aux_ao_offset = aux_offsets[aux_batch_id]
            naux = aux_offsets[aux_batch_id+1] - aux_ao_offset
            out = ndarray((nao_pair, naux, bvk_ncells), buffer=out)
            # The output buffer must be initialized because integral screening
            # based on SR integrals is performed in the kernel, and certain ~0
            # shell-tritets are not evaluated, leaving the output buffer untouched
            out[:] = 0.
            if out.size == 0:
                return out
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(shl_pair1 - shl_pair0),
                ctypes.c_int(aux_split1 - aux_split0),
                ctypes.cast(bas_ij_idx[shl_pair0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets[shl_pair0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc[shl_pair0:].data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux * bvk_ncells),
                ctypes.c_int(not cart),
                ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
                ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel')
            return out
        return evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets

    pair_and_diag_indices = FTOpt.pair_and_diag_indices

    def contract_dm(self, dm, kpts=None, hermi=0):
        assert dm.shape[1] == self.cell.nao
        if self.bvkmesh_Ls is None:
            self.build()

        if hermi != 1:
            dm = transpose_sum(dm, inplace=False)
        if kpts is None or is_zero(kpts):
            assert dm.dtype == np.float64
        else:
            expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(asarray(kpts).T))
            dm = contract('Lk,kpq->Lpq', expLk, dm)
            dm = cp.asarray(dm.real, order='C')
            dm *= 1./len(kpts)
        assert dm.dtype == np.float64
        assert dm.flags.c_contiguous

        cell = self.cell
        auxcell = self.auxcell
        bvk_ncells = len(self.bvkmesh_Ls)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(self.bas_ij_cache, 256)

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, auxcell.uniq_l_ctr, 32)
        aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)

        ksh_idx = _aggregate_bas_idx(
            l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, auxcell.nbas)[1]
        ksh_idx += self.bvkcell.nbas
        ksh_offsets = cp.asarray(l_ctr_aux_offsets, dtype=np.int32)
        shl_pair_batches = len(shl_pair_offsets) - 1
        aux_batches = len(ksh_offsets) - 1

        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
        int3c2e_envs = self.int3c2e_envs

        naux = auxcell.nao
        vj_aux = cp.zeros((naux, bvk_ncells))
        err = libpbc.PBCcontract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(shl_pair_batches),
            ctypes.c_int(aux_batches),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm failed')
        vj_aux = vj_aux.sum(axis=1)[aux_sorting]
        if hermi == 1:
            vj_aux *= 2
        return auxcell.apply_CT_dot(vj_aux)

    def contract_auxvec(self, auxvec, kpts=None):
        assert auxvec.dtype == np.float64
        assert auxvec.ndim == 1
        assert len(auxvec) == self.auxcell.nao
        auxvec = cp.asarray(auxvec)
        if self.bvkmesh_Ls is None:
            self.build()

        cell = self.cell
        auxcell = self.auxcell
        bvk_ncells = len(self.bvkmesh_Ls)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(gout_width=30)
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx = cell.aggregate_shl_pairs(self.bas_ij_cache, 1000000)[0]

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
        ksh_idx = _aggregate_bas_idx(
            l_ctr_aux_offsets, auxcell.uniq_l_ctr, bvk_ncells, auxcell.nbas)[1]
        ksh_idx += self.bvkcell.nbas
        ksh_offsets = cp.asarray(l_ctr_aux_offsets, dtype=np.int32)
        aux_batches = len(ksh_offsets) - 1

        diffuse_exps = cp.asarray(self.diffuse_exps)
        diffuse_coefs = cp.asarray(self.diffuse_coefs)
        atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
        log_cutoff = math.log(self.cutoff)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
        int3c2e_envs = self.int3c2e_envs

        nao = cell.nao
        vj = cp.zeros((nao, bvk_ncells, nao))
        err = libpbc.PBCcontract_int3c2e_auxvec(
            ctypes.cast(vj.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(aux_batches),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('contract_int3c2e_auxvec failed')

        if kpts is None or is_zero(kpts):
            vj = vj[:,0]
        else:
            nkpts = len(kpts)
            expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(asarray(kpts).T))
            expLkz = expLk.view(np.float64).reshape(bvk_ncells,nkpts,2)
            vj = contract('Lkz,pLq->kpqz', expLkz, vj)
            vj = vj.view(np.complex128)[:,:,:,0]
        vj = transpose_sum(vj)
        vj = cell.apply_CT_mat_C(vj)
        return vj

def _conc_locs(ao_loc1, ao_loc2):
    comp_loc = np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)
    return cp.array(comp_loc, dtype=np.int32)

def int3c2e_scheme(gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk
    nroots = order//2 + 1
    nroots *= 2 # for short-range
    g_size = (li+1)*(lj+1)*(lk+1)
    unit = g_size*3 + nroots*2 + 7
    shm_size = shm_size - 1024
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    if gout_width is not None:
        gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size += (nfi + nfj + nfk) * 3 * 4
    return nsp_per_block, gout_stride, shm_size

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

def diffuse_exps_by_atom(cell):
    '''Find the most diffuse functions on each atom'''
    exps, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,ANG_OF]
    r2 = np.log(cs**2 / cell.precision * 10**ls + 1e-200) / exps
    idx = groupby(cell._bas[:,ATOM_OF], r2, 'argmax')
    return exps[idx]

def _aggregate_shl_pairs(img_idx_cache, nsp_per_block):
    sp_img_idx = []
    sp_img_offsets = []
    bas_ij_idx = []
    img_offset_cum = 0
    sp0 = sp1 = 0
    shl_pair_offsets = []
    for li, lj in img_idx_cache:
        img_idx, img_offsets, bas_ij = img_idx_cache[li, lj][:3]
        sp_img_idx.append(img_idx)
        sp_img_offsets.append(img_offset_cum + img_offsets[:-1])
        img_offset_cum += img_offsets[-1]
        bas_ij_idx.append(bas_ij)
        sp0, sp1 = sp1, sp1 + len(bas_ij)
        shl_pair_offsets.append(cp.arange(
            sp0, sp1, nsp_per_block[li,lj], dtype=np.int32))

    sp_img_idx = cp.asarray(cp.hstack(sp_img_idx), dtype=np.int32)
    sp_img_offsets.append(img_offset_cum)
    sp_img_offsets = cp.asarray(cp.hstack(sp_img_offsets), dtype=np.int32)
    bas_ij_idx = cp.asarray(cp.hstack(bas_ij_idx), dtype=np.int32)
    shl_pair_offsets.append(np.int32(sp1))
    shl_pair_offsets = cp.asarray(cp.hstack(shl_pair_offsets), dtype=np.int32)
    return shl_pair_offsets, bas_ij_idx, sp_img_idx, sp_img_offsets

def _aggregate_bas_idx(l_ctr_offsets, uniq_l_ctr, bvk_ncells, nbas, batch_size=256):
    ksh_offsets = []
    ksh_idx = []
    k0 = k1 = 0
    bvk_bas_offsets = cp.arange(bvk_ncells, dtype=np.int32) * nbas
    for ksh0, ksh1 in zip(l_ctr_offsets[:-1], l_ctr_offsets[1:]):
        idx = (bvk_bas_offsets + cp.arange(ksh0, ksh1, dtype=np.int32)[:,None]).ravel()
        ksh_idx.append(idx)
        k0, k1 = k1, k1 + len(idx)
        ksh_offsets.append(cp.arange(k0, k1, batch_size, dtype=np.int32))
    repeats = [len(x) for x in ksh_offsets]
    uniq_l_ctr = np.repeat(uniq_l_ctr, repeats, axis=0)

    ksh_offsets.append(np.int32(k1))
    ksh_offsets = cp.asarray(cp.hstack(ksh_offsets), dtype=np.int32)
    ksh_idx = cp.asarray(cp.hstack(ksh_idx), dtype=np.int32)
    return ksh_offsets, ksh_idx, uniq_l_ctr
