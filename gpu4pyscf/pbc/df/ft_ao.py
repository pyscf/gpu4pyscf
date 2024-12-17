#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Compute analytical Fourier transform
'''

import ctypes
import math
import numpy as np
import cupy as cp
import scipy.linalg
from pyscf import lib
from pyscf.gto.mole import ANG_OF, ATOM_OF
from pyscf.scf import _vhf
from pyscf.pbc import tools as pbctools
from pyscf.pbc.gto.cell import _extract_pgto_params
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.lib.cupy_helper import load_library, contract
from gpu4pyscf.gto.mole import group_basis
from gpu4pyscf.scf.jk import (
    g_pair_idx, _nearest_power2, _scale_sp_ctr_coeff, SHM_SIZE)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger

__all__ = [
    'ft_aopair', 'ft_aopair_kpts', 'ft_ao'
]

libpbc = load_library('libpbc')
libpbc.PBC_build_ft_ao.restype = ctypes.c_int
libpbc.PBC_FT_init_constant.restype = ctypes.c_int

PTR_BAS_COORD = 7
LMAX = 4
GOUT_WIDTH = 19 # 15?
THREADS = 256

def ft_aopair(cell, Gv, kpti_kptj=None, q=None):
    if kpti_kptj is None:
        kptj = np.zeros((1, 3))
    else:
        kpti, kptj = kpti_kptj
        q = kptj - kpti
    return ft_aopair_kpts(cell, Gv, q, kptj.reshape(1,3))

def ft_aopair_kpts(cell, Gv, q=None, kptjs=None, bvk_kmesh=None):
    if q is None:
        q = np.zeros(3)
    if bvk_kmesh is None and kptjs is not None:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kptjs)
    ft_kernel = gen_ft_kernel(cell, bvk_kmesh)
    return ft_kernel(Gv, q, kptjs)

def ft_ao(cell, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None):
    from pyscf.pbc.df.ft_ao import ft_ao
    out = ft_ao(cell, Gv, shls_slice, b, gxyz, Gvbase, kpt, verbose)
    return cp.asarray(out)

def _bas_overlap_mask(cell, bvkmesh_Ls, Ls, cutoff=None):
    '''integral screening mask for basis product between cell and supmol'''
    # consider only the most diffused component of a basis
    exps, cs = _extract_pgto_params(cell, 'min')
    ls = cell._bas[:,ANG_OF]
    bas_coords = cp.asarray(cell.atom_coords()[cell._bas[:,ATOM_OF]])

    vol = cell.vol
    if cutoff is None:
        theta_ij = exps.min() / 2
        lattice_sum_factor = max(2*np.pi*cell.rcut/(vol*theta_ij), 1)
        cutoff = cell.precision/lattice_sum_factor * .1
        logger.debug(cell, 'Set ft_ao cutoff to %g', cutoff)

    ls = cp.asarray(ls)
    exps = cp.asarray(exps)
    norm = cp.asarray(cs) * ((2*ls+1)/(4*np.pi))**.5
    aij = exps[:,None] + exps
    theta = exps[:,None] * exps / aij

    Ls = cp.asarray(Ls)
    # rj is in the order of (bvk_cell_id, bas_id, lattice_img_id)
    rj = bvkmesh_Ls[:,None,None,:] + bas_coords[:,None,:] + Ls
    rirj = bas_coords[:,None,None,None,:] - rj

    dr = cp.linalg.norm(rirj, axis=4)

    dri = exps[None,None,:,None]/aij[:,None,:,None] * dr
    drj = exps[:,None,None,None]/aij[:,None,:,None] * dr
    li = ls[:,None,None,None]
    lj = ls[None,None,:,None]
    odd_l = ls % 2 == 1
    # li is even: ((li-1) * .5/aij[:,None,None,:] + dri**2) ** (li//2)
    # li is odd : (li * .5/aij[:,None,None,:] + dri**2) ** ((li+1)//2) * dri
    fac_dri = ((li-1) * .5/aij[:,None,:,None] + dri**2) ** (li//2)
    fac_drj = ((lj-1) * .5/aij[:,None,:,None] + drj**2) ** (lj//2)
    li_odd = li[odd_l,:,:,:]
    lj_odd = lj[:,:,odd_l,:]
    dri_odd = dri[odd_l,:,:,:]
    drj_odd = drj[:,:,odd_l,:]
    fac_dri[odd_l,:,:,:] = (li_odd*.5/aij[odd_l,None,:,None] + dri_odd**2)**((li_odd-1)//2) * dri_odd
    fac_drj[:,:,odd_l,:] = (lj_odd*.5/aij[:,None,odd_l,None] + drj_odd**2)**((lj_odd-1)//2) * drj_odd
    fl = 2*np.pi/vol * (dr/theta[:,None,:,None]) + 1.
    fac_norm = norm[:,None]*norm * (np.pi/aij)**1.5
    ovlp = fac_norm[:,None,:,None] * cp.exp(-theta[:,None,:,None]*dr**2) * fac_dri * fac_drj * fl
    return ovlp > cutoff

def gen_ft_kernel(cell, bvk_kmesh=None, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    The output tensor is saved in the shape [nGv, nao, nao] for single k-point
    case and [nkpts, nGv, nao, nao] for multiple k-points
    '''
    log = logger.new_logger(cell, verbose)
    cput0 = log.init_timer()

    cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    coeff = cp.asarray(coeff)

    # create BVK super-cell
    if bvk_kmesh is None:
        bvkmesh_Ls = cp.zeros(3)
        bvkcell = cell
    else:
        bvkmesh_Ls = cp.asarray(
                k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True))
        bvkcell = pbctools.super_cell(cell, bvk_kmesh, wrap_around=True)
    Ls = cp.asarray(bvkcell.get_lattice_Ls())
    Ls = Ls[cp.linalg.norm(Ls-.5, axis=1).argsort()]

    # Generate img_idx based on the overlap between shells in cell and super-mol
    ovlp_mask = _bas_overlap_mask(cell, bvkmesh_Ls, Ls)
    bvk_ncells, nbas, nimgs = ovlp_mask.shape[1:]
    # Number of images for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
    img_counts = cp.count_nonzero(ovlp_mask, axis=3)
    img_offsets = cp.append(0, cp.cumsum(img_counts.ravel())).astype(np.int32)
    # The image Ids for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
    img_idx = cp.nonzero(ovlp_mask.reshape(-1, nimgs))[1].astype(np.int32)

    bvk_ovlp_mask = ovlp_mask.any(axis=3)
    #TODO: symmetry between ish and jsh?
    #ix, iy = cp.triu(nbas, -1)
    #bvk_ovlp_mask[ix,:,iy] = False

    _atm = cp.array(bvkcell._atm)
    _bas = cp.array(bvkcell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(bvkcell))
    ao_loc = cp.array(bvkcell.ao_loc)
    aft_envs = AFTIntEnvVars(
        bvkcell.natm, bvkcell.nbas,
        _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
        Ls.data.ptr, img_idx.data.ptr, img_offsets.data.ptr,
    )
    # Keep a reference to these arrays, prevent releasing them upon returning the closure
    aft_envs._env_ref_holder = (_atm, _bas, _env, ao_loc, Ls, img_idx, img_offsets)

    nao = coeff.shape[0]
    ao_loc = bvkcell.ao_loc
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    aosym = 's1'

    init_constant(cell)
    kern = libpbc.PBC_build_ft_ao
    log.timer_debug1('initialize ft_kern', *cput0)

    def ft_kernel(Gv, q=np.zeros(3), kptjs=None, aosym=aosym):
        '''
        Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
        '''
        t1 = log.init_timer()
        assert q.ndim == 1
        nGv = len(Gv)
        assert nGv > 0
        # Padding zeros, allowing idle threads to access these data
        GvT = cp.append(Gv.T.ravel(), cp.zeros(THREADS))
        out = cp.zeros((nao, nao, bvk_ncells, nGv), dtype=np.complex128)

        timing_collection = {}
        kern_counts = 0

        for i in range(n_groups):
            for j in range(i+1):
                li = uniq_l[i]
                lj = uniq_l[j]
                ll_pattern = f'{l_symb[i]}{l_symb[j]}'
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                mask = bvk_ovlp_mask[ish0:ish1,:,jsh0:jsh1]
                sub_img_counts = img_counts[ish0:ish1,:,jsh0:jsh1]
                # Sort according to the number of images. In the CUDA kernel,
                # shell-pairs that have closed number of images are processed on
                # the same SM processor, ensuring the best parallel execution.
                idx = cp.argsort(sub_img_counts[mask])[::-1]
                i_in_pair, j_in_pair = cp.nonzero(mask.reshape(ish1-ish0, -1))
                i_in_pair = i_in_pair.astype(np.int32)[idx]
                j_in_pair = j_in_pair.astype(np.int32)[idx]

                scheme = ft_ao_scheme(cell, li, lj, nGv)
                log.debug2('ft_ao_scheme %s', scheme)
                err = kern(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    aft_envs, (ctypes.c_int*3)(*scheme),
                    (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                    ctypes.c_int(i_in_pair.size), ctypes.c_int(nGv),
                    ctypes.cast(i_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.cast(j_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    cell._atm.ctypes, ctypes.c_int(cell.natm),
                    cell._bas.ctypes, ctypes.c_int(cell.nbas), cell._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'PBC_build_ft_ao kernel for {ll_pattern} failed')
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

        #:out = einsum('pqLG,pi,qj->LGij', out, coeff, coeff)
        out = contract('pqLG,pi->qLGi', out, coeff)
        out = contract('qLGi,qj->LGij', out, coeff)
        if kptjs is None:
            out = out.sum(axis=0)
        else:
            kptjs = cp.asarray(kptjs, order='C').reshape(-1,3)
            expLk = cp.exp(1j*cp.dot(bvkmesh_Ls, kptjs.T))
            out = contract('Lk,LGij->kGij', expLk, out)

        #TODO:
        #if aosym == 's1hermi':
        #    # Gamma point only
        #    assert is_zero(q) and is_zero(kptjs) and ni == nj
        #    # Theoretically, hermitian symmetry can be also found for kpti == kptj != 0:
        #    #       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
        #    # hermi operation needs to reorder axis-0.  It is inefficient.
        #if aosym == 's1hermi':
        #    for i in range(1, ni):
        #        out[:,:,:i,i] = out[:,:,i,:i]

        log.timer('ft_aopair', *cput0)
        return out

    return ft_kernel

class AFTIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
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
    err = libpbc.PBC_FT_init_constant(
        g_idx.ctypes, offsets.ctypes, cell._env.ctypes, ctypes.c_int(cell._env.size),
        ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

def ft_ao_scheme(cell, li, lj, nGv, shm_size=SHM_SIZE):
    order = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gout_size = nfi * nfj
    gout_stride = (gout_size + GOUT_WIDTH-1) // GOUT_WIDTH
    # Round up to the next 2^n
    gout_stride = _nearest_power2(gout_stride, return_leq=False)

    g_size = (li+1)*(lj+1)
    unit = g_size*3
    nGv_max = min(shm_size//(unit*16), THREADS//gout_stride)

    # Test nGv_per_block in 8..nGv_max, find the case of minimal idle threads
    idle_min = nGv_max
    nGv_test = nGv_per_block = 8
    while nGv_test <= nGv_max:
        idle = (-nGv) % nGv_test
        if idle <= idle_min:
            idle_min = idle
            nGv_per_block = nGv_test
        nGv_test *= 2

    sp_blocks = THREADS // (gout_stride * nGv_per_block)
    return nGv_per_block, gout_stride, sp_blocks
