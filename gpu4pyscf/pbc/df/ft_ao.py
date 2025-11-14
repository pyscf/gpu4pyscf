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
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, get_avail_mem, dist_matrix, asarray)
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import (
    _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE)
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'ft_aopair', 'ft_aopair_kpts', 'ft_ao'
]

libpbc = load_library('libpbc')
libpbc.build_ft_ao.restype = ctypes.c_int
libpbc.build_ft_aopair.restype = ctypes.c_int
libpbc.init_constant(ctypes.c_int(SHM_SIZE))

LMAX = 4
GOUT_WIDTH = 19
THREADS = 256

def ft_aopair(cell, Gv, kpti_kptj=None, q=None):
    if kpti_kptj is None:
        kptj = np.zeros((1, 3))
    else:
        kpti, kptj = kpti_kptj
        if q is None:
            q = kptj - kpti
    return ft_aopair_kpts(cell, Gv, q, kptj.reshape(1,3))[0]

def ft_aopair_kpts(cell, Gv, q=None, kptjs=None):
    '''Analytical Fourier transform orbital-pair on Gv grids'''
    if q is None:
        q = np.zeros(3)
    if kptjs is None:
        kptjs = np.zeros((1, 3))
    ft_kernel = gen_ft_kernel(cell, kptjs)
    return ft_kernel(Gv, q, kptjs)

def ft_ao(cell, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None,
          sort_cell=True):
    '''Analytical Fourier transform basis functions on Gv grids.

    If the sorted_cell in the input is specified, the transform
    '''
    assert shls_slice is None
    if sort_cell:
        sorted_cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
    else:
        assert cell.cart
        assert all(cell._bas[:,NCTR_OF] == 1)
        sorted_cell = cell

    _atm = cp.array(sorted_cell._atm)
    _bas = cp.array(sorted_cell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(sorted_cell))
    ao_loc_cpu = sorted_cell.ao_loc
    ao_loc_gpu = cp.array(ao_loc_cpu)
    envs = PBCIntEnvVars(
        sorted_cell.natm, sorted_cell.nbas, _atm.data.ptr,
        _bas.data.ptr, _env.data.ptr, ao_loc_gpu.data.ptr, 1, 1, 0,
    )
    ngrids = len(Gv)
    GvT = (asarray(Gv).T + asarray(kpt[:,None])).ravel()
    GvT = cp.append(GvT, cp.zeros(THREADS))
    nao_cart = ao_loc_cpu[-1]
    out = cp.empty((nao_cart, ngrids), dtype=np.complex128)
    libpbc.build_ft_ao(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_int(ngrids),
        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
        sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
        sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
        sorted_cell._env.ctypes
    )
    if sort_cell:
        out = out.T.dot(asarray(coeff))
    else:
        out = out.T
    return out

def ft_ao_ip1(cell, Gv, kpt=np.zeros(3), verbose=None, sort_cell=True):
    raise NotImplementedError

def gen_ft_kernel(cell, kpts=None, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The output tensor is saved in the shape [nGv, nao, nao] for single k-point
    case and [nkpts, nGv, nao, nao] for multiple k-points
    '''
    return FTOpt(cell, kpts).gen_ft_kernel(verbose)


class FTOpt:
    def __init__(self, cell, kpts=None, bvk_kmesh=None):
        self.cell = cell
        sorted_cell, ao_idx, l_ctr_pad_counts, uniq_l_ctr, l_ctr_counts = group_basis(
            cell, tile=1, sparse_coeff=True)
        self.sorted_cell = sorted_cell
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.ao_idx = ao_idx
        self.l_ctr_pad_counts = l_ctr_pad_counts

        if bvk_kmesh is None:
            if kpts is None or is_zero(kpts):
                bvk_kmesh = np.ones(3, dtype=int)
            else:
                bvk_kmesh = kpts_to_kmesh(sorted_cell, kpts)
        self.bvk_kmesh = bvk_kmesh
        self.kpts = kpts

        self._aft_envs = None
        self.bvkcell = None
        self._img_idx_cache = {}

    def build(self, verbose=None):
        log = logger.new_logger(self.cell, verbose)
        cell = self.sorted_cell
        bvk_kmesh = self.bvk_kmesh
        self.bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell

        Ls = cp.asarray(bvkcell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]
        nimgs = len(Ls)

        bvk_ncells = np.prod(bvk_kmesh)
        nbas = cell.nbas
        log.debug('bvk_ncells=%d, nbas=%d, nimgs=%d', bvk_ncells, nbas, nimgs)

        _atm = cp.array(bvkcell._atm)
        _bas = cp.array(bvkcell._bas)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell))
        ao_loc = cp.array(bvkcell.ao_loc)
        self._aft_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, nimgs, _atm, _bas, _env, ao_loc, Ls)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self._aft_envs = None
        self.bvkcell = None
        self._img_idx_cache = {}
        return self

    @property
    def coeff(self):
        from pyscf import gto
        coeff = np.zeros((self.sorted_cell.nao, self.cell.nao))

        l_max = max([l_ctr[0] for l_ctr in self.uniq_l_ctr])
        if self.cell.cart:
            cart2sph_per_l = [np.eye((l+1)*(l+2)//2) for l in range(l_max + 1)]
        else:
            cart2sph_per_l = [gto.mole.cart2sph(l, normalized = "sp") for l in range(l_max + 1)]
        i_spherical_offset = 0
        i_cartesian_offset = 0
        for i, l in enumerate(self.uniq_l_ctr[:,0]):
            cart2sph = cart2sph_per_l[l]
            ncart, nsph = cart2sph.shape
            l_ctr_count = self.l_ctr_offsets[i + 1] - self.l_ctr_offsets[i]
            cart_offs = i_cartesian_offset + np.arange(l_ctr_count) * ncart
            sph_offs = i_spherical_offset + np.arange(l_ctr_count) * nsph
            cart_idx = cart_offs[:,None] + np.arange(ncart)
            sph_idx = sph_offs[:,None] + np.arange(nsph)
            coeff[cart_idx[:,:,None],sph_idx[:,None,:]] = cart2sph
            l_ctr_pad_count = self.l_ctr_pad_counts[i]
            i_cartesian_offset += (l_ctr_count + l_ctr_pad_count) * ncart
            i_spherical_offset += l_ctr_count * nsph
        assert len(self.ao_idx) == self.cell.nao
        out = cp.zeros_like(coeff)
        out[:,self.ao_idx] = coeff
        return asarray(out)

    @property
    def aft_envs(self):
        _aft_envs = self._aft_envs
        if _aft_envs is None:
            raise RuntimeError('FTOpt not initialized')
        if cp.cuda.device.get_device_id() == _aft_envs._device:
            return self._aft_envs
        return _aft_envs.copy()

    def estimate_cutoff_with_penalty(self):
        cell = self.sorted_cell
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

    def make_img_idx_cache(self, permutation_symmetry, verbose=None):
        '''Cache significant orbital-pairs and their lattice sum images'''
        if permutation_symmetry in self._img_idx_cache:
            return self._img_idx_cache[permutation_symmetry]

        log = logger.new_logger(self.cell, verbose)
        if self._aft_envs is None:
            self.build(verbose)

        cell = self.sorted_cell
        nbas = cell.nbas
        l_ctr_offsets = self.l_ctr_offsets
        uniq_l = self.uniq_l_ctr[:,0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        bvk_kmesh = self.bvk_kmesh
        if bvk_kmesh is None:
            bvk_ncells = 1
        else:
            bvk_ncells = np.prod(bvk_kmesh)
        cutoff = self.estimate_cutoff_with_penalty()
        log_cutoff = math.log(cutoff)

        exps, cs = extract_pgto_params(cell, 'diffuse')
        exps = cp.asarray(exps, dtype=np.float32)
        log_coeff = cp.log(abs(cp.asarray(cs, dtype=np.float32)))

        if permutation_symmetry:
            # symmetry between ish and jsh can be utilized. The triu part is excluded
            # from computation.
            ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        else:
            ij_tasks = itertools.product(range(n_groups), range(n_groups))

        aft_envs = self.aft_envs
        bas_ij_cache = {}
        for i, j in ij_tasks:
            ll_pattern = f'{l_symb[i]}{l_symb[j]}'
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
            nish = ish1 - ish0
            njsh = jsh1 - jsh0
            img_counts = cp.zeros((nish*bvk_ncells*njsh), dtype=np.int32)
            err = libpbc.overlap_img_counts(
                ctypes.cast(img_counts.data.ptr, ctypes.c_void_p),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(aft_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.c_int(int(permutation_symmetry)))
            if err != 0:
                raise RuntimeError(f'{ll_pattern} overlap_img_counts failed')
            bas_ij = cp.asarray(cp.where(img_counts > 0)[0], dtype=np.int32)
            n_pairs = len(bas_ij)
            if n_pairs == 0:
                bas_ij_cache[i, j] = (bas_ij, None, None)
                continue

            # Sort according to the number of images. In the CUDA kernel,
            # shell-pairs that have closed number of images are processed on
            # the same SM processor, ensuring the best parallel execution.
            counts_sorting = (-img_counts[bas_ij]).argsort()
            bas_ij = bas_ij[counts_sorting]
            img_counts = img_counts[bas_ij]
            img_offsets = cp.empty(n_pairs+1, dtype=np.int32)
            img_offsets[0] = 0
            cp.cumsum(img_counts, out=img_offsets[1:])
            tot_imgs = int(img_offsets[n_pairs])
            img_idx = cp.empty(tot_imgs, dtype=np.int32)
            err = libpbc.overlap_img_idx(
                ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_pairs),
                (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                ctypes.byref(aft_envs),
                ctypes.cast(exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError(f'{ll_pattern} overlap_img_idx failed')
            img_counts = counts_sorting = None

            # bas_ij stores the non-negligible primitive-pair indices.
            ish, J, jsh = cp.unravel_index(bas_ij, (nish, bvk_ncells, njsh))
            ish += ish0
            jsh += jsh0
            bas_ij = cp.ravel_multi_index((ish, J, jsh), (nbas, bvk_ncells, nbas))
            bas_ij = cp.asarray(bas_ij, dtype=np.int32)
            bas_ij_cache[i, j] = (bas_ij, img_offsets, img_idx)
            log.debug1('task (%d, %d), n_pairs=%d', i, j, n_pairs)

            self._img_idx_cache[permutation_symmetry] = bas_ij_cache
        return bas_ij_cache

    def gen_ft_kernel(self, verbose=None):
        r'''
        Generate the analytical fourier transform kernel for AO products

        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        By default, the output tensor is saved in the shape [nGv, nao, nao] for
        single k-point case and [nkpts, nGv, nao, nao] for multiple k-points
        '''
        log = logger.new_logger(self.cell, verbose)
        cput0 = log.init_timer()
        if self._aft_envs is None:
            self.build(verbose)

        cell = self.sorted_cell
        uniq_l = self.uniq_l_ctr[:,0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        l_ctr_offsets = self.l_ctr_offsets
        kern = libpbc.build_ft_aopair

        bvk_kmesh = self.bvk_kmesh
        kpts = self.kpts
        bvk_ncells = np.prod(bvk_kmesh)
        if bvk_ncells == 1:
            bvkmesh_Ls = cp.zeros((1, 3))
            conj_mapping = cp.zeros(1, dtype=np.int32)
        else:
            bvkmesh_Ls = cp.asarray(
                k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True))
            conj_mapping = cp.asarray(conj_images_in_bvk_cell(bvk_kmesh), dtype=np.int32)
        nao = self.sorted_cell.nao
        nao_orig = self.cell.nao
        coeff = cp.asarray(self.coeff, dtype=np.complex128)

        def _ft_sub(Gv, q, kptjs, img_idx_cache, transform_ao=True):
            t1 = log.init_timer()
            timing_collection = {}
            kern_counts = 0
            # Padding zeros, allowing idle threads to access these data
            GvT = cp.asarray(Gv.T) + cp.asarray(q)[:,None]
            GvT = cp.append(GvT.ravel(), cp.zeros(THREADS))
            aft_envs = self.aft_envs
            nGv = len(Gv)
            out = cp.zeros((bvk_ncells, nao, nao, nGv), dtype=np.complex128)

            for i, j in img_idx_cache:
                bas_ij, img_offsets, img_idx = img_idx_cache[i, j]
                npairs = len(bas_ij)
                if npairs == 0:
                    continue

                li = uniq_l[i]
                lj = uniq_l[j]
                ll_pattern = f'{l_symb[i]}{l_symb[j]}'
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                scheme = ft_ao_scheme(cell, li, lj, nGv)
                log.debug2('ft_ao_scheme for %s: %s', ll_pattern, scheme)
                err = kern(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(0), # Do not remove zero elements
                    ctypes.byref(aft_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                    ctypes.c_int(npairs), ctypes.c_int(nGv),
                    ctypes.cast(bas_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
                    cell._atm.ctypes, ctypes.c_int(cell.natm),
                    cell._bas.ctypes, ctypes.c_int(cell.nbas), cell._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'build_ft_aopair kernel for {ll_pattern} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {ll_pattern}', *t1), t1
                    if ll_pattern not in timing_collection:
                        timing_collection[ll_pattern] = 0
                    timing_collection[ll_pattern] += t1[1] - t1p[1]
                    kern_counts += 1

            if log.verbose >= logger.DEBUG1:
                log.debug1('kernel launches %d', kern_counts)
                for ll_pattern, t in timing_collection.items():
                    log.debug1('%s wall time %.2f', ll_pattern, t)

            if is_zero(q):
                log.debug1('symmetrize output')
                # For i<j, the real orbital product in BvK cell <i|j+L> is identical to <j|i-L>
                # conj_imgs stores the image indices of the corresponding +L and -L
                #ix, iy = cp.tril_indices(nao, -1)
                #for k, ck in enumerate(conj_mapping):
                #    out[iy,ix,ck] = out[ix,iy,k]
                err = libpbc.ft_aopair_fill_triu(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.cast(conj_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nao), ctypes.c_int(bvk_ncells), ctypes.c_int(nGv))
                if err != 0:
                    raise RuntimeError('ft_aopair_fill_triu kernel for {ll_pattern} failed')

            log.debug1('transform BvK-cell to k-points')
            gamma_point_only = kptjs is None or is_zero(kptjs)
            if not gamma_point_only:
                kptjs = cp.asarray(kptjs, order='C').reshape(-1,3)
                expLk = cp.exp(1j*cp.dot(bvkmesh_Ls, kptjs.T))
                out = contract('Lk,LpqG->kpqG', expLk, out)

            if transform_ao:
                log.debug1('transform basis')
                #:out = einsum('pqLG,pi,qj->LGij', out, coeff, coeff)
                out = contract('kpqG,pi->kiqG', out, coeff)
                out = contract('kiqG,qj->kijG', out, coeff)

            log.timer('ft_aopair', *cput0)
            return out

        def ft_kernel(Gv, q=np.zeros(3), kptjs=kpts, transform_ao=True):
            '''
            Analytical FT for orbital products. The output tensor has the shape
            [nk, nGv, nao, nao]

            FT tensor is first computed in the basis of sorted_cell.
            transform_ao=True transforms AOs to their original order.
            '''
            assert q.ndim == 1
            nGv = len(Gv)
            assert nGv > 0
            out_size = nao**2 * bvk_ncells*nGv * 16
            avail_mem = get_avail_mem()
            permutation_symmetry = is_zero(q)
            img_idx_cache = self.make_img_idx_cache(permutation_symmetry, log)

            if 2*out_size < avail_mem * .8:
                return _ft_sub(Gv, q, kptjs, img_idx_cache,
                               transform_ao).transpose(0,3,1,2)

            elif out_size < avail_mem * .8:
                if kptjs is None:
                    nkpts = 1
                else:
                    kptjs = kptjs.reshape(-1, 3)
                    nkpts = len(kptjs)
                if transform_ao:
                    out = cp.empty((nkpts, nao_orig, nao_orig, nGv), dtype=np.complex128)
                else:
                    out = cp.empty((nkpts, nao, nao, nGv), dtype=np.complex128)
                Gv_block = int((avail_mem * .95 - out_size) / (2*nao**2*bvk_ncells*16))
                Gv_block &= 0xfffffc
                if Gv_block >= 4:
                    logger.debug1(cell, 'Processing ft_kernel in sub-blocks, Gv_block = %d', Gv_block)
                    for p0, p1 in lib.prange(0, nGv, Gv_block):
                        out[:,:,:,p0:p1] = _ft_sub(Gv[p0:p1], q, kptjs,
                                                   img_idx_cache, transform_ao)
                    return out.transpose(0,3,1,2)

            raise RuntimeError('Not enough GPU memory. '
                               f'Available: {avail_mem*1e-9:.2f} GB. '
                               f'Required: {out_size*1.2e-9:.2f} GB')
        return ft_kernel

def most_diffuse_pgto(cell):
    exps, cs = extract_pgto_params(cell, 'diffuse')
    ls = cell._bas[:,ANG_OF]
    r2 = np.log(cs**2 / cell.precision * 10**ls + 1e-200) / exps
    idx = r2.argmax()
    return exps[idx], cs[idx], ls[idx]
most_diffused_pgto = most_diffuse_pgto # for backward compatibility

class PBCIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('bvk_ncells', ctypes.c_int),
        ('nimgs', ctypes.c_int),
        ('img_coords', ctypes.c_void_p),
    ]

    @classmethod
    def new(cls, natm, nbas, ncells, nimgs, atm, bas, env, ao_loc, Ls):
        obj = PBCIntEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr, env.data.ptr,
                            ao_loc.data.ptr, ncells, nimgs, Ls.data.ptr)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc, Ls)
        obj._device = cp.cuda.device.get_device_id()
        return obj

    def copy(self):
        atm, bas, env, ao_loc, Ls = self._env_ref_holder
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        Ls = cp.asarray(Ls)
        return PBCIntEnvVars.new(
            self.natm, self.nbas, self.bvk_ncells, self.nimgs,
            atm, bas, env, ao_loc, Ls)

    @property
    def device(self):
        return self._device

def ft_ao_scheme(cell, li, lj, nGv, shm_size=SHM_SIZE):
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gout_size = nfi * nfj
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    g_size = (li+1)*(lj+1)
    unit = g_size*3
    nGv_nsp_max = shm_size//(unit*16)
    nGv_nsp_max = _nearest_power2(nGv_nsp_max)
    nGv_max = min(nGv_nsp_max, THREADS//gout_stride, 64)

    # gout_stride*nGv_per_block >= 32 is a must due to syncthreads in CUDA kernel
    nGv_per_block = max(32//gout_stride, 1)

    # Test nGv_per_block in 1..nGv_max, find the case of minimal idle threads
    idle_min = nGv_max
    nGv_test = nGv_per_block
    while nGv_test <= nGv_max:
        idle = (-nGv) % nGv_test
        if idle <= idle_min:
            idle_min = idle
            nGv_per_block = nGv_test
        nGv_test *= 2

    sp_blocks = THREADS // (gout_stride * nGv_per_block)
    # the nGv * sp_blocks restrictrions due to shared memory size
    sp_blocks = min(sp_blocks, nGv_nsp_max // nGv_per_block)
    gout_stride = THREADS // (nGv_per_block * sp_blocks)
    return nGv_per_block, gout_stride, sp_blocks
