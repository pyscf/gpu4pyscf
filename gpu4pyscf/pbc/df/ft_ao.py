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
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COORD
from pyscf.scf import _vhf
from pyscf.pbc import tools as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, get_avail_mem, dist_matrix)
from gpu4pyscf.gto.mole import group_basis, PTR_BAS_COORD
from gpu4pyscf.scf.jk import (
    g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE)
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.gto.mole import extract_pgto_params
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'ft_aopair', 'ft_aopair_kpts', 'ft_ao'
]

libpbc = load_library('libpbc')
libpbc.build_ft_ao.restype = ctypes.c_int
libpbc.init_constant.restype = ctypes.c_int

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
    if q is None:
        q = np.zeros(3)
    if kptjs is None:
        kptjs = np.zeros((1, 3))
    ft_kernel = gen_ft_kernel(cell, kptjs)
    return ft_kernel(Gv, q, kptjs)

def ft_ao(cell, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None):
    from pyscf.pbc.df.ft_ao import ft_ao
    out = ft_ao(cell, Gv, shls_slice, b, gxyz, Gvbase, kpt, verbose)
    if out.flags.c_contiguous:
        return cp.asarray(out)
    else:
        return cp.asarray(out, order='F')

def _bas_overlap_mask(cell, bvkmesh_Ls, Ls):
    '''integral screening mask for basis product between cell and supmol'''
    # consider only the most diffused component of a basis
    exps, cs = extract_pgto_params(cell, 'diffused')
    ls = cell._bas[:,ANG_OF]
    bas_coords = cp.asarray(cell.atom_coords()[cell._bas[:,ATOM_OF]])

    ls = cp.asarray(ls)
    exps = cp.asarray(exps)
    norm = cp.asarray(cs) * ((2*ls+1)/(4*np.pi))**.5
    aij = exps[:,None] + exps
    fi = exps[:,None] / aij
    fj = exps[None,:] / aij
    theta = exps[:,None] * fj

    Ls = cp.asarray(Ls)
    # rj format: (bvk_cell_id, bas_id, lattice_img_id)
    rj = bvkmesh_Ls[:,None,None,:] + bas_coords[:,None,:] + Ls
    #:rirj = bas_coords[:,None,None,None,:] - rj
    #:dr = cp.linalg.norm(rirj, axis=4)
    dr = dist_matrix(bas_coords, rj.reshape(-1,3))
    dr = dr.reshape(len(bas_coords), *rj.shape[:3])

    dri = fj[:,None,:,None] * dr
    drj = fi[:,None,:,None] * dr
    li = ls[:,None,None,None]
    lj = ls[None,None,:,None]
    fac_dri = (li * .5/aij[:,None,:,None] + dri**2) ** (li*.5)
    fac_drj = (lj * .5/aij[:,None,:,None] + drj**2) ** (lj*.5)
    rad = cell.vol**(-1./3) * dr + 1
    surface = 4*np.pi * rad**2
    fl = cp.where(surface > 1, surface, 1)
    fac_norm = norm[:,None]*norm * (np.pi/aij)**1.5
    ovlp = fac_norm[:,None,:,None] * cp.exp(-theta[:,None,:,None]*dr**2) * fac_dri * fac_drj * fl
    return ovlp > cell.precision

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
        cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
        self.sorted_cell = cell
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.coeff = cp.asarray(coeff, dtype=np.complex128)

        if bvk_kmesh is None:
            if kpts is None or is_zero(kpts):
                bvk_kmesh = np.ones(3, dtype=int)
            else:
                bvk_kmesh = kpts_to_kmesh(cell, kpts)
        if np.prod(bvk_kmesh) == 1:
            bvkcell = cell
        else:
            bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
            # PTR_BAS_COORD was not initialized in pbctools.supe_rcell
            bvkcell._bas[:,PTR_BAS_COORD] = bvkcell._atm[bvkcell._bas[:,ATOM_OF],PTR_COORD]
        self.bvkcell = bvkcell
        self.bvk_kmesh = bvk_kmesh
        self.kpts = kpts

    def gen_ft_kernel(self, verbose=None):
        r'''
        Generate the analytical fourier transform kernel for AO products

        \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

        The output tensor is saved in the shape [nGv, nao, nao] for single k-point
        case and [nkpts, nGv, nao, nao] for multiple k-points
        '''
        cell = self.sorted_cell
        coeff = self.coeff
        uniq_l_ctr = self.uniq_l_ctr
        l_ctr_offsets = self.l_ctr_offsets
        bvk_kmesh = self.bvk_kmesh
        bvkcell = self.bvkcell
        kpts = self.kpts

        log = logger.new_logger(cell, verbose)
        cput0 = log.init_timer()
        Ls = cp.asarray(bvkcell.get_lattice_Ls())
        Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

        if bvk_kmesh is None:
            bvkmesh_Ls = cp.zeros((1, 3))
        else:
            bvkmesh_Ls = cp.asarray(
                k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True))

        # Generate img_idx based on the overlap between shells in cell and super-mol
        ovlp_mask = _bas_overlap_mask(cell, bvkmesh_Ls, Ls)
        bvk_ncells, nbas, nimgs = ovlp_mask.shape[1:]
        log.debug('bvk_ncells=%d, nbas=%d, nimgs=%d', bvk_ncells, nbas, nimgs)
        # Number of images for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
        img_counts = cp.count_nonzero(ovlp_mask, axis=3)
        img_offsets = cp.append(0, cp.cumsum(img_counts.ravel())).astype(np.int32)
        # The image Ids for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
        img_idx = cp.nonzero(ovlp_mask.reshape(-1, nimgs))[1].astype(np.int32)
        bvk_ovlp_mask = ovlp_mask.any(axis=3)

        _atm = cp.array(bvkcell._atm)
        _bas = cp.array(bvkcell._bas)
        _env = cp.array(_scale_sp_ctr_coeff(bvkcell))
        ao_loc = cp.array(bvkcell.ao_loc)
        aft_envs = AFTIntEnvVars(
            bvkcell.natm, bvkcell.nbas, bvk_ncells, 0,
            _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
            Ls.data.ptr, img_idx.data.ptr, img_offsets.data.ptr,
        )
        # Keep a reference to these arrays, prevent releasing them upon returning the closure
        aft_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls, img_idx, img_offsets)

        nao, nao_orig = coeff.shape
        ao_loc = bvkcell.ao_loc
        uniq_l = uniq_l_ctr[:,0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        if bvk_kmesh is None:
            conj_mapping = cp.zeros(1, dtype=np.int32)
        else:
            conj_mapping = cp.asarray(conj_images_in_bvk_cell(bvk_kmesh), dtype=np.int32)

        init_constant(cell)
        kern = libpbc.build_ft_ao
        cp.cuda.Stream.null.synchronize()
        log.timer_debug1('initialize ft_kern', *cput0)

        def _ft_sub(Gv, q, kptjs, transform_ao=True):
            '''
            FT tensor is first computed in the basis of sorted_cell, which
            transform_ao requires to transform AOs to their original order
            '''
            t1 = log.init_timer()
            timing_collection = {}
            kern_counts = 0
            nGv = len(Gv)
            # Padding zeros, allowing idle threads to access these data
            if isinstance(Gv, cp.ndarray) :
                GvT = cp.append((Gv.T + cp.asarray(q)[:,None]).ravel(), cp.zeros(THREADS))
            else:
                GvT = cp.append((Gv.T + q[:,None]).ravel(), cp.zeros(THREADS))
            out = cp.zeros((bvk_ncells, nao, nao, nGv), dtype=np.complex128)

            permutation_symmetry = is_zero(q)
            if permutation_symmetry:
                # symmetry between ish and jsh can be utilized. The triu part is excluded
                # from computation.
                ix, iy = cp.triu_indices(nbas, 1)
                _bvk_ovlp_mask = bvk_ovlp_mask.copy()
                _bvk_ovlp_mask[ix,:,iy] = False
                ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
            else:
                _bvk_ovlp_mask = bvk_ovlp_mask
                ij_tasks = itertools.product(range(n_groups), range(n_groups))

            for i, j in ij_tasks:
                li = uniq_l[i]
                lj = uniq_l[j]
                ll_pattern = f'{l_symb[i]}{l_symb[j]}'
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                mask = _bvk_ovlp_mask[ish0:ish1,:,jsh0:jsh1]
                sub_img_counts = img_counts[ish0:ish1,:,jsh0:jsh1]
                # Sort according to the number of images. In the CUDA kernel,
                # shell-pairs that have closed number of images are processed on
                # the same SM processor, ensuring the best parallel execution.
                idx = cp.argsort(sub_img_counts[mask])[::-1]
                i_in_pair, bvk_id, j_in_pair = cp.nonzero(mask)
                i_in_pair = (i_in_pair + ish0).astype(np.int32)[idx]
                j_in_pair = (j_in_pair + jsh0 + bvk_id*nbas).astype(np.int32)[idx]

                scheme = ft_ao_scheme(cell, li, lj, nGv)
                log.debug2('ft_ao_scheme for %s: %s', ll_pattern, scheme)
                err = kern(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aft_envs), (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                    ctypes.c_int(i_in_pair.size), ctypes.c_int(nGv),
                    ctypes.cast(i_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.cast(j_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    cell._atm.ctypes, ctypes.c_int(cell.natm),
                    cell._bas.ctypes, ctypes.c_int(cell.nbas), cell._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'build_ft_ao kernel for {ll_pattern} failed')
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

            if permutation_symmetry:
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
                    raise RuntimeError('ft_aopair_fill_triu kernel failed')

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
            Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
            '''
            assert q.ndim == 1
            nGv = len(Gv)
            assert nGv > 0
            out_size = nao**2 * bvk_ncells*nGv * 16
            avail_mem = get_avail_mem()

            if 2*out_size < avail_mem * .8:
                return _ft_sub(Gv, q, kptjs, transform_ao).transpose(0,3,1,2)

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
                        out[:,:,:,p0:p1] = _ft_sub(Gv[p0:p1], q, kptjs, transform_ao)
                    return out.transpose(0,3,1,2)

            raise RuntimeError('Not enough GPU memory. '
                               f'Available: {avail_mem*1e-9:.2f} GB. '
                               f'Required: {out_size*1.2e-9:.2f} GB')
        return ft_kernel

class AFTIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('bvk_ncells', ctypes.c_uint16),
        ('padding', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('img_coords', ctypes.c_void_p),
        ('img_idx', ctypes.c_void_p),
        ('img_offsets', ctypes.c_void_p),
    ]

def init_constant(cell):
    g_idx, offsets = g_pair_idx()
    err = libpbc.init_constant(
        g_idx.ctypes, offsets.ctypes, cell._env.ctypes, ctypes.c_int(cell._env.size),
        ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

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
    nGv_max = min(nGv_nsp_max, THREADS//gout_stride)

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
