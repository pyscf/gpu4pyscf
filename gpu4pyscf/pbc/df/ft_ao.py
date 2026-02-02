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
Analytical Fourier transform for orbital-products
'''

import ctypes
import math
import itertools
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
from pyscf.gto.mole import ANG_OF, NPRIM_OF, NCTR_OF, ATOM_OF, PTR_COORD
from pyscf.scf import _vhf
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, get_avail_mem, dist_matrix, asarray, ndarray)
from gpu4pyscf.gto.mole import group_basis, SortedGTO, PTR_BAS_COORD
from gpu4pyscf.df.int3c2e_bdiv import get_ao_pair_loc
from gpu4pyscf.scf.jk import (
    _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE)
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.gto.mole import extract_pgto_params, RysIntEnvVars, PBCIntEnvVars
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'ft_aopair', 'ft_aopair_kpts', 'ft_ao'
]

libpbc = load_library('libpbc')
libpbc.build_ft_ao.restype = ctypes.c_int
libpbc.build_ft_aopair.restype = ctypes.c_int

LMAX = 4
GOUT_WIDTH = 29
THREADS = 256
POOL_SIZE = 65536

def ft_aopair(cell, Gv, kpti_kptj=None, q=None):
    if kpti_kptj is None:
        kptj = None
    else:
        kpti, kptj = kpti_kptj
        if q is None:
            q = kptj - kpti
        kptj = kptj.reshape(1,3)
    return ft_aopair_kpts(cell, Gv, q, kptj)[0]

def ft_aopair_kpts(cell, Gv, q=None, kptjs=None):
    '''Analytical Fourier transform orbital-pair on Gv grids'''
    if q is None:
        q = np.zeros(3)
    ft_kernel = gen_ft_kernel(cell, kptjs)
    return ft_kernel(Gv, q, kptjs)

def ft_ao(cell, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None,
          sort_cell=True):
    '''Analytical Fourier transform basis functions on Gv grids.

    If the sort_cell in the input is specified, the ao is evaluated on a sorted Cartesian basis,
    and then transformed back to original basis.
    '''
    assert shls_slice is None
    cell = SortedGTO.from_cell(cell)
    _env = _scale_sp_ctr_coeff(cell)
    ao_loc = cell.ao_loc
    envs = RysIntEnvVars.new(
        cell.natm, cell.nbas, cell._atm, cell._bas, _env, ao_loc)
    ngrids = len(Gv)
    assert ngrids < np.iinfo(np.int32).max, "possible int32 overflow"
    GvT = (asarray(Gv).T + asarray(kpt[:,None])).ravel()
    GvT = cp.append(GvT, cp.zeros(THREADS))
    nao = ao_loc[-1]
    out = cp.empty((nao, ngrids), dtype=np.complex128)
    err = libpbc.build_ft_ao(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_int(ngrids),
        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
        ctypes.c_int(cell.nbas))
    if err != 0:
        raise RuntimeError('build_ft_ao failed')
    if sort_cell:
        out = cell.apply_CT_dot(out, axis=0)
    return out.T

def ft_ao_ip1(cell, Gv, kpt=np.zeros(3), verbose=None, sort_cell=True):
    raise NotImplementedError

def gen_ft_kernel(cell, kpts=None, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The output tensor is saved in the shape [nGv, nao, nao] for single k-point
    case and [nkpts, nGv, nao, nao] for multiple k-points
    '''
    if kpts is None:
        kmesh = None
    else:
        kmesh = kpts_to_kmesh(cell, kpts)
    return FTOpt(cell, kmesh).gen_ft_kernel(verbose)

class FTOpt:
    def __init__(self, cell, bvk_kmesh=None):
        self.cell = SortedGTO.from_cell(cell)
        if bvk_kmesh is None:
            bvk_kmesh = np.ones(3, dtype=int)
        self.bvk_kmesh = bvk_kmesh

        self.rcut = None
        self._aft_envs = None
        self.bas_ij_cache = None
        self.bvkcell = None
        self.bvkmesh_Ls = None
        self.permutation_symmetry = True

    def build(self):
        log = logger.new_logger(self.cell)
        cell = self.cell
        bvk_kmesh = self.bvk_kmesh
        bvk_ncells = np.prod(bvk_kmesh)
        self.bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        Ls = cp.asarray(bvkcell.get_lattice_Ls(rcut=self.rcut))
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)
        log.debug('ft_ao bvk_ncells=%d, nimgs=%d', bvk_ncells, nimgs)

        _env = _scale_sp_ctr_coeff(bvkcell)
        ao_loc = bvkcell.ao_loc
        self._aft_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs,
            bvkcell._atm, bvkcell._bas, _env, ao_loc, Ls)

        exps, coef = extract_pgto_params(bvkcell, 'diffuse')
        self.diffuse_exps = cp.asarray(exps, dtype=np.float32)
        self.diffuse_coefs = cp.asarray(coef, dtype=np.float32)
        log_c = cp.log(self.diffuse_coefs)

        self.cutoff = cutoff = self.estimate_cutoff_with_penalty()
        log_cutoff = math.log(cutoff)

        nbas = cell.nbas
        img_counts = cp.zeros((nbas*bvk_ncells*nbas), dtype=np.uint32)
        libpbc.bvk_ovlp_img_counts(
            ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
            ctypes.byref(self._aft_envs),
            ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            ctypes.c_int(int(self.permutation_symmetry)))

        mask = img_counts.reshape(nbas, bvk_ncells, nbas) > 0
        self.bas_ij_cache = bas_ij_cache = {}
        groups = len(cell.uniq_l_ctr)
        l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
        if self.permutation_symmetry:
            ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
        else:
            ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
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
            ctypes.byref(self._aft_envs),
            ctypes.cast(self.diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(log_c.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        self.img_idx = img_idx
        self.img_offsets = img_offsets
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self._aft_envs = None
        self.bvkcell = None
        self.bas_ij_cache = {}
        return self

    @property
    def aft_envs(self):
        _aft_envs = self._aft_envs
        if _aft_envs is None:
            raise RuntimeError('FTOpt not initialized')
        if cp.cuda.device.get_device_id() == _aft_envs.device:
            return self._aft_envs
        return _aft_envs.copy()

    def estimate_cutoff_with_penalty(self):
        cell = self.cell
        rcut = cell.rcut
        vol = cell.vol
        cell_exp, _, cell_l = most_diffuse_pgto(cell)
        lsum = cell_l * 2 + 1
        rad = vol**(-1./3) * rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = 2*np.pi*rcut*lsum/(vol*cell_exp*2) + surface
        cutoff = cell.precision / lattice_sum_factor
        logger.debug1(cell, 'ft_ao min_exp=%g cutoff=%g', cell_exp, cutoff)
        return cutoff

    def pair_and_diag_indices(self, cart=None, original_ao_order=True):
        if self.bvkmesh_Ls is None:
            self.build()
        cell = self.cell
        if cart is None:
            cart = cell.cell.cart
        bvk_ncells = np.prod(self.bvk_kmesh)
        nbas = cell.nbas
        ao_loc = self.bvkcell.ao_loc_nr(cart=cart)
        nao = ao_loc[-1]
        if original_ao_order:
            dims = (ao_loc[1:] - ao_loc[:-1]).reshape(bvk_ncells, nbas)
            dims, tmp = np.empty_like(dims), dims
            dims[:,cell.sorted_idx] = tmp
            ao_loc = cp.asarray(np.append(0, np.cumsum(dims.ravel())))
            sorted_idx = (cp.arange(bvk_ncells)[:,None] * nbas +
                          cp.asarray(cell.sorted_idx)).ravel()

        ao_loc = cp.asarray(ao_loc)
        uniq_l = cell.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        carts = [cp.arange(n) for n in nf]
        # diag stores the indices for cderi_row that corresponds to
        # the diagonal blocks. Note this index array can contain some of the
        # off-diagonal elements which happen to be the off-diagonal elements
        # while within the diagonal blocks.
        offset = 0
        diag = []
        ao_pair_addresses = []
        for (i, j), bas_ij in self.bas_ij_cache.items():
            ish, jsh = divmod(bas_ij, bvk_ncells*nbas)
            if original_ao_order:
                ish = sorted_idx[ish]
                jsh = sorted_idx[jsh]
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            ao_pair_addresses.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
            if i == j: # the diagonal blocks
                jsh_cell0 = jsh % nbas
                nfi = nf[i]
                idx = cp.where(ish == jsh_cell0)[0]
                addr = offset + idx[:,None] * (nfi*nfi) + cp.arange(nfi*nfi)
                diag.append(addr.ravel())
            offset += len(bas_ij) * nf[i] * nf[j]
        ao_pair_addresses = cp.hstack(ao_pair_addresses, dtype=np.int32)
        diag = cp.hstack(diag, dtype=np.int32)
        return ao_pair_addresses, diag

    def ft_evaluator(self, batch_size=None, compressing=True, cart=None,
                     original_ao_order=True, bas_ij_aggregated=None):
        r'''
        Generate the analytical fourier transform kernel for AO products

        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        By default, the output tensor is saved in the shape [nGv, nao, nao] for
        single k-point case and [nkpts, nGv, nao, nao] for multiple k-points
        '''
        if self._aft_envs is None:
            self.build()

        cell = self.cell
        nsp_per_block, gout_stride, shm_size = ft_ao_scheme()
        lmax = cell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:lmax+1,:lmax+1].max()
        if bas_ij_aggregated is None:
            bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
                self.bas_ij_cache, nsp_per_block)
        else:
            bas_ij_idx, shl_pair_offsets = bas_ij_aggregated

        if cart is None:
            cart = cell.cell.cart
        ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], self.bas_ij_cache, cart)
        ao_loc = cell.ao_loc_nr(cart=cart)
        nao = ao_loc[-1]

        if not compressing and original_ao_order:
            dims = ao_loc[1:] - ao_loc[:-1]
            dims, tmp = np.empty_like(dims), dims
            dims[cell.sorted_idx] = tmp
            ao_loc = cp.asarray(np.append(0, np.cumsum(dims.ravel())))
            ao_loc = np.append(ao_loc[cell.sorted_idx], nao)
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)

        if batch_size is None:
            pair_splits = [0, len(shl_pair_offsets)-1]
            ao_pair_offsets = [0, ao_pair_loc[-1].get()]
        else:
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
            pair_splits = splits_by_blocksize(ao_pair_offsets, batch_size)
            ao_pair_offsets = ao_pair_offsets[pair_splits]

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, POOL_SIZE), dtype=np.float64)
        aft_envs = self.aft_envs
        img_idx = cp.asarray(self.img_idx)
        img_offsets = cp.asarray(self.img_offsets)
        bvk_ncells = len(self.bvkmesh_Ls)
        kern = libpbc.build_ft_aopair

        def evaluate_ft(Gv, batch_id=0, out=None):
            nGv = len(Gv)
            # Padding zeros, allowing idle threads to access these data
            GvT = cp.append(cp.asarray(Gv.T.ravel()), cp.zeros(THREADS))

            if compressing:
                pair_split0 = pair_splits[batch_id]
                pair_split1 = pair_splits[batch_id+1]
                pair_blocks = pair_split1 - pair_split0
                _shl_pair_offsets = shl_pair_offsets[pair_split0:]
                ao_pair_offset = ao_pair_offsets[batch_id]
                nao_pair = ao_pair_offsets[batch_id+1] - ao_pair_offset
                out = ndarray((nao_pair, nGv), dtype=np.complex128, buffer=out)
                if not cart:
                    out[:] = 0.
            else:
                pair_blocks = len(shl_pair_offsets) - 1
                _shl_pair_offsets = shl_pair_offsets
                ao_pair_offset = 0
                out = ndarray((nao, bvk_ncells, nao, nGv), dtype=np.complex128, buffer=out)
                out[:] = 0.
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(pair_blocks),
                ctypes.cast(_shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nGv),
                ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(compressing),
                ctypes.c_int(not cart))
            if err != 0:
                raise RuntimeError('build_ft_aopair kernel failed')
            return out

        return evaluate_ft, ao_pair_offsets

    def gen_ft_kernel(self, verbose=None, transform_ao=True):
        r'''
        Generate the analytical fourier transform kernel for AO products

        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        By default, the output tensor is saved in the shape [nGv, nao, nao] for
        single k-point case and [nkpts, nGv, nao, nao] for multiple k-points

        FT tensor is first computed in the basis of sorted_cell.
        transform_ao=True transforms AOs to their original order.
        '''
        from gpu4pyscf.pbc.df.int3c2e import fill_triu_bvk
        cart = None
        if not transform_ao:
            cart = True
        eval_ft = self.ft_evaluator(compressing=False, cart=cart,
                                    original_ao_order=transform_ao)[0]

        pair_address = self.pair_and_diag_indices(cart, original_ao_order=transform_ao)[0]
        pair_address = cp.asarray(pair_address, dtype=np.int32)
        bvk_ncells = len(self.bvkmesh_Ls)
        if bvk_ncells == 1:
            conj_mapping = cp.zeros(1, dtype=np.int32)
        else:
            conj_mapping = conj_images_in_bvk_cell(self.bvk_kmesh)
            conj_mapping = cp.asarray(conj_mapping, dtype=np.int32)

        cell = self.cell.cell
        nao = cell.nao_nr(cart=cart)
        # tril_idx in the reference cell associated to the pair_address.
        # Note indices within this array does not guarantee i>=j. It only indicates
        # the unique pairs for each unit cell.
        mask = cp.zeros(nao*bvk_ncells*nao, dtype=bool)
        mask[pair_address] = True
        mask = cp.any(mask.reshape(nao, bvk_ncells, nao), axis=1)
        tril_idx = cp.asarray(cp.where(mask.ravel())[0], dtype=np.int32)

        def ft_kernel(Gv, q=None, kpts=None, kj_idx=None):
            '''
            Analytical FT for orbital products. The output tensor has the shape
            [nk, nGv, nao, nao]

            If kj_idx is specified, it is used to sort the first dimension
            (kpts) of the output.
            '''
            if q is None:
                out = eval_ft(Gv)
            else:
                assert q.shape == (3,)
                out = eval_ft(Gv+q)

            nGv = len(Gv)
            symmetric_for_bvk_orbitals = (self.permutation_symmetry and
                                          (q is None or is_zero(q)))
            if symmetric_for_bvk_orbitals:
                logger.debug1(cell, 'symmetrize ft_aopair')
                fill_triu_bvk(out.view(np.float64), nao, self.bvk_kmesh,
                              pair_address, conj_mapping, bvk_axis=1)

            if kpts is None or is_zero(kpts):
                if bvk_ncells != 1:
                    out = out.sum(axis=1)[:,None]
                if self.permutation_symmetry and not symmetric_for_bvk_orbitals:
                    libpbc.fill_indexed_triu(
                        ctypes.cast(out.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tril_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(len(tril_idx)), ctypes.c_int(1),
                        ctypes.c_int(nao), ctypes.c_int(nGv*2))
                return out.transpose(1,3,0,2)
            else:
                logger.debug1(cell, 'transform BvK-cell to k-points')
                kpts = asarray(kpts, order='C')
                expLk = cp.exp(1j*asarray(self.bvkmesh_Ls).dot(kpts.T))
                out = contract('Lk,pLqG->kpqG', expLk, out)
                if (kj_idx is not None and
                    self.permutation_symmetry and not symmetric_for_bvk_orbitals):
                    nkpts = expLk.shape[1]
                    assert bvk_ncells == nkpts
                    conj_ki_order = cp.empty(nkpts, dtype=np.int32)
                    conj_ki_order[kj_idx] = conj_mapping
                    libpbc.fill_indexed_triu(
                        ctypes.cast(out.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tril_idx.data.ptr, ctypes.c_void_p),
                        ctypes.cast(conj_ki_order.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(len(tril_idx)), ctypes.c_int(nkpts),
                        ctypes.c_int(nao), ctypes.c_int(nGv*2))
                return out.transpose(0,3,1,2)
        return ft_kernel

def most_diffuse_pgto(cell):
    exps, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,ANG_OF]
    r2 = np.log(cs**2 / cell.precision * 10**ls + 1e-200) / exps
    idx = r2.argmax()
    return exps[idx], cs[idx], ls[idx]
most_diffused_pgto = most_diffuse_pgto # for backward compatibility

def ft_ao_scheme():
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gout_size = nfi * nfj
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    nGv_per_block = 32
    nsp_max = 8 // gout_stride
    assert np.all(nsp_max > 0)
    g_size = (li+1)*(lj+1)
    unit = g_size*3
    nsp_per_block = _nearest_power2((SHM_SIZE-256) // (nGv_per_block*(unit*16)))
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = cp.asarray(8 // nsp_per_block, dtype=np.int32)
    shm_size = nGv_per_block * nsp_per_block * (unit*16)
    shm_size += nsp_per_block * 3 * 8
    shm_size += (nfi + nfj) * 3 * 4
    return nsp_per_block, gout_stride, shm_size
