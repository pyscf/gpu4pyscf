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
from pyscf.pbc.lib import k2gamma
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

PTR_BAS_COORD = 7
LMAX = 4
#UNROLL_ORDER = ctypes.c_int.in_dll(libpbc, 'rys_jk_unrolled_max_order').value
#UNROLL_LMAX = ctypes.c_int.in_dll(libpbc, 'rys_jk_unrolled_lmax').value
#UNROLL_NFMAX = ctypes.c_int.in_dll(libpbc, 'rys_jk_unrolled_max_nf').value
#UNROLL_J_LMAX = ctypes.c_int.in_dll(libpbc, 'rys_j_unrolled_lmax').value
#UNROLL_J_MAX_ORDER = ctypes.c_int.in_dll(libpbc, 'rys_j_unrolled_max_order').value
GOUT_WIDTH = 19 # 15?
THREADS = 256

def ft_aopair(cell, Gv, shls_slice=None, aosym='s1',
              b=None, gxyz=None, Gvbase=None, kpti_kptj=np.zeros((2,3)),
              q=None, intor='GTO_ft_ovlp', comp=1, verbose=None):
    pass


def ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                   b=None, gxyz=None, Gvbase=None, q=np.zeros(3),
                   kptjs=np.zeros((1,3)), intor='GTO_ft_ovlp', comp=1,
                   bvk_kmesh=None, out=None):
    pass


def ft_ao(cell, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=np.zeros(3), verbose=None):
    pass

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
    norm = cp.asarray(cs * ((2*ls+1)/(4*np.pi))**.5)
    aij = exps[:,None] + exps
    theta = exps[:,None] * exps / aij

    Ls = cp.asarray(Ls)
    # rj is in the order of (bvk_cell_id, bas_id, lattice_img_id)
    rj = bvkmesh_Ls[:,None,None,:] + bas_coords[:,None,:] + Ls
    rirj = bas_coords[:,None,None,None,:] - rj

    dr = cp.linalg.norm(rirj, axis=4)

    dri = exps[None,None,None,:]/aij[:,None,None,:] * dr
    drj = exps[:,None,None,None]/aij[:,None,None,:] * dr
    li = ls[:,None,None,None]
    lj = ls[None,None,None,:]
    fac_dri = ((li-1) * .5/aij[:,None,None,:] + dri**2) ** (li//2)
    fac_drj = ((lj-1) * .5/aij[:,None,None,:] + drj**2) ** (lj//2)
    odd_l = ls % 2 == 1
    fac_dri[odd_l,:,:,:] *= dri
    fac_drj[:,:,:,odd_l] *= drj
    fl = 2*np.pi/vol * (dr/theta[:,None,None,:]) + 1.
    ovlp = (norm[:,None]*norm * np.pi**1.5 * aij[:,None,None,:]**-1.5 *
            cp.exp(-theta[:,None,None,:]*dr**2) * fac_dri * fac_drj * fl)
    return ovlp > cutoff

def gen_ft_kernel(cell, kpts=None, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
    '''
    log = logger.new_logger(cell, verbose)
    cput0 = log.init_timer()

    cell, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell, tile=1)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    coeff = cp.asarray(coeff)

    # create BVK super-cell
    if kpts is None:
        bvkmesh_Ls = cp.zeros(3)
        bvkcell = cell
    else:
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        bvkmesh_Ls = cp.asarray(k2gamma.translation_vectors_for_kmesh(cell, kmesh, True))
        bvkcell = pbctools.super_cell(cell, kmesh, wrap_around=True)
    Ls = bvkcell.get_lattice_Ls()
    Ls = Ls[np.linalg.norm(Ls-.5, axis=1).argsort()]

    # Generate img_idx based on the overlap between shells in cell and super-mol
    ovlp_mask = _bas_overlap_mask(cell, bvkmesh_Ls, Ls)
    bvk_ncells, nbas, nimgs = ovlp_mask.shape[1:]
    # Number of images for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
    img_counts = cp.count_nonzero(ovlp_mask, axis=3)
    img_offsets = cp.append(0, cp.cumsum(img_counts.ravel())).astype(np.int32)
    # The image Ids for each pair of (bas_i_in_cell0, bas_j_in_bvkcell)
    img_idx = cp.nonzero(ovlp_mask.reshape(-1, nimgs))[1].astype(np.int32)

    bvk_ovlp_mask = ovlp_mask.any(axis=3)
    #if permutation symmetry?
    #ix, iy = cp.triu(nbas, -1)
    #bvk_ovlp_mask[ix,:,iy] = False

    _atm = cp.array(bvkcell._atm)
    _bas = cp.array(bvkcell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(bvkcell))
    ao_loc = cp.array(bvkcell.ao_loc)
    bvkcell._env_on_gpu = (_atm, _bas, _env, ao_loc, Ls, img_idx, img_offsets)
    aft_envs = AFTIntEnvVars(
        bvkcell.natm, bvkcell.nbas,
        _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
        Ls.data.ptr, img_idx.data.ptr, img_offsets.data.ptr,
    )

    nao = coeff.shape[0]
    ao_loc = bvkcell.ao_loc
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    aosym = 's1'

    init_constant(cell)
    kern = libpbc.PBC_build_ft_ao
    log.timer_debug1('initialize ft_kern', *cput0)

    def ft_kernel(Gv, q=np.zeros(3), kptjs=None, aosym=aosym, out=None):
        '''
        Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
        '''
        t1 = log.init_timer()
        assert q.ndim == 1
        nGv = len(Gv)
        assert nGv > 0
        GvT = cp.asarray(Gv.T, order='C')
        out = cp.zeros((nao, nao, bvk_ncells, nGv), dtype=np.complex128)

        timing_collection = {}
        kern_counts = 0

        for i in range(n_groups):
            for j in range(i+1):
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

                scheme = ft_ao_scheme(cell, uniq_l_ctr[[i, j]], nGv)
                err = kern(
                    ctypes.cast(out.data.ptr, ctypes.c_void_p),
                    aft_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                    ctypes.c_int(i_in_pair.size),
                    ctypes.cast(i_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.cast(j_in_pair.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nGv), ctypes.c_int(nGv),
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
        if kptjs is not None:
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
    libpbc.PBC_FT_init_constant(
        g_idx.ctypes, offsets.ctypes, cell._env.ctypes, ctypes.c_int(cell._env.size),
        ctypes.c_int(SHM_SIZE))

def ft_ao_scheme(cell, l_ctr_pattern, nGv, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj = ls
    order = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gout_size = nfi * nfj
#    if (gout_size <= UNROLL_NFMAX or order <= UNROLL_ORDER) and all(ls <= UNROLL_LMAX):
#        if (CUDA_VERSION >= 12040 and
#            order <= 3 and (li,lj,lk,ll) != (1,1,1,0) and (li,lj,lk,ll) != (1,0,1,1)):
#            return 512, 1
#        return 256, 1

    g_size = (li+1)*(lj+1)
    unit = g_size*3
    counts = shm_size // (unit*16)
    counts = _nearest_power2(counts)
    raise
#    if counts < nGv:
#        n = min(THREADS, counts)
#    else:
#        n = _nearest_power2(nGv*2-1)
#    gout_stride = THREADS // n
#    while gout_stride < 16 and gout_size / (gout_stride*GOUT_WIDTH) > 1:
#        n //= 2
#        gout_stride *= 2
#    return n, gout_stride
