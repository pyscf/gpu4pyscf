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
from pyscf.gto import (ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF,
                       PTR_EXP, gto_norm)
from pyscf import lib
from pyscf.scf import _vhf
from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import load_library, condense, sandwich_dot, transpose_sum
from gpu4pyscf.scf.jk import (
    g_pair_idx, _nearest_power2, _split_l_ctr_groups, basis_seg_contraction)
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
SHM_SIZE = getattr(__config__, 'GPU_SHM_SIZE',
                   int(gpu_specs['sharedMemPerBlockOptin']//9)*8)
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

def _overlap_mask(cell1, cell2):
    mask = []
    mask

def gen_ft_kernel(cell, kpts=None, verbose=None):
    r'''
    Generate the analytical fourier transform kernel for AO products

    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
    '''
    log = logger.new_logger(cell, verbose)
    cput0 = log.init_timer()

    sorted_cell, coeff = basis_seg_contraction(cell)

    # Sort basis according to angular momentum and contraction patterns so
    # as to group the basis functions to blocks in GPU kernel.
    l_ctrs = sorted_cell._bas[:,[ANG_OF, NPRIM_OF]]
    # Ensure the more contracted Gaussians being accessed first
    l_ctrs_descend = l_ctrs.copy()
    l_ctrs_descend[:,1] = -l_ctrs[:,1]
    uniq_l_ctr, where, inv_idx, l_ctr_counts = np.unique(
        l_ctrs_descend, return_index=True, return_inverse=True, return_counts=True, axis=0)
    uniq_l_ctr[:,1] = -uniq_l_ctr[:,1]
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    if cell.verbose >= logger.DEBUG1:
        log.debug1('Number of shells for each [l, nprim] group')
        for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
            log.debug1('    %s : %s', l_ctr, n)

    nao_orig = coeff.shape[1]
    ao_loc = sorted_cell.ao_loc
    coeff = np.split(coeff, ao_loc[1:-1], axis=0)

    pad_inv_idx = []
    pad_bas = []
    inv_idx = np.hstack([inv_idx.ravel(), pad_inv_idx])
    sorted_idx = np.argsort(inv_idx, kind='stable').astype(np.int32)
    coeff = cp.asarray(np.vstack([coeff[i] for i in sorted_idx]))
    assert coeff.shape[0] < 32768

    max_nprims = uniq_l_ctr[:,1].max()
    sorted_cell._env = np.append(sorted_cell._env, np.zeros(max_nprims))
    if pad_bas:
        sorted_cell._bas = np.vstack([sorted_cell._bas, pad_bas])[sorted_idx]
    else:
        sorted_cell._bas = sorted_cell._bas[sorted_idx]
    assert sorted_cell._bas.dtype == np.int32

    # PTR_BAS_COORD is required by CUDA kernels
    sorted_cell._bas[:,PTR_BAS_COORD] = sorted_cell._atm[sorted_cell._bas[:,ATOM_OF],PTR_COORD]

    # very high angular momentum basis are processed on CPU
    lmax = uniq_l_ctr[:,0].max()
    assert lmax <= LMAX

    if kpts is None:
        bvk_cell = sorted_cell
    else:
        bvk_cell = supercell??

    _atm = cp.array(bvk_cell._atm)
    _bas = cp.array(bvk_cell._bas)
    _env = cp.array(_scale_sp_ctr_coeff(bvk_cell))
    ao_loc = cp.array(bvk_cell.ao_loc)
    bvk_cell._env_on_gpu = (_atm, _bas, _env, ao_loc)
    aft_envs = AFTIntEnvVars(
        bvk_cell.natm, bvk_cell.nbas,
        _atm.data.ptr, _bas.data.ptr, _env.data.ptr, ao_loc.data.ptr,
        img_coords, img_idx, shl_pair_img_offsets,
    )

    # create bvk_cell
    # ovlp_mask
    # generate img_idx

    nbasp = rs_cell.ref_cell.nbas
    cell0_ao_loc = rs_cell.ref_cell.ao_loc
    bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape

    ovlp_mask = supmol.get_ovlp_mask()

    img_count = cp.count_nonzero(ovlp_mask, axis=1)
    img_offsets = cp.append(0, cp.cumsum(img_count))
    img_idx = cp.nonzero(ovlp_mask)[1]


    bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, rs_cell.sh_loc, supmol.sh_loc)
    cell0_ovlp_mask = bvk_ovlp_mask.reshape(nbasp, bvk_ncells, nbasp).any(axis=1)
    ovlp_mask = ovlp_mask.astype(np.int8)
    cell0_ovlp_mask = cell0_ovlp_mask.astype(np.int8)
    log.timer_debug1('initialize ft_kern', *cput0)

    kern = libpbc.PBC_build_ft_ao
    log.timer_debug1('ft_ao kernel initialization', *cput0)

    def ft_kernel(Gv, q=np.zeros(3), kptjs=None, aosym=aosym, out=None):
        '''
        Analytical FT for orbital products. The output tensor has the shape [nGv, nao, nao]
        '''
        cput0 = log.init_timer()
        assert q.ndim == 1
        assert shls_slice is None:
        nGv = len(Gv)
        assert nGv > 0
        GvT = cp.asarray(Gv.T, order='C')

        nao, nao_orig = vhfopt.coeff.shape

        out = cp.zeros((nao_cell0, nao_bvk, nGv))
        out_ptr = ctypes.cast(out.data.ptr, ctypes.c_void_p)

        init_constant(cell)
        ao_loc = cell.ao_loc

        uniq_l_ctr = vhfopt.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = vhfopt.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        timing_collection = {}
        kern_counts = 0
        kern = libpbc.PBC_build_ft_ao

        #if aosym == 's1hermi':
        #    # Gamma point only
        #    assert is_zero(q) and is_zero(kptjs) and ni == nj
        #    # Theoretically, hermitian symmetry can be also found for kpti == kptj != 0:
        #    #       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
        #    # hermi operation needs to reorder axis-0.  It is inefficient.

        out = cp.zeros((nao, nao, bvk_ncells, nGv), dtype=np.complex128)

        for i in range(n_groups):
            for j in range(i+1):
                llll = f'{l_symb[i]}{l_symb[j]}'
                ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
                jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
                mask = ovlp_mask > cutoff
                if i == j:
                    mask = cp.tril(mask)
                t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                        cp.arange(j0, j1, dtype=np.int32))
                idx = cp.argsort(sub_tile_q[mask])[::-1]
                pair_mapping = t_ij[mask][idx]
                pair_mapping = sort(pair_mapping, img_counts[i,j])

                scheme = ft_ao_scheme(cell, uniq_l_ctr[[i, j]], nGv)
                err = kern(
                    out_ptr, ctypes.c_int(nao),
                    vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*4)(ish0, ish1, jsh0, jsh1),
                    ctypes.c_int(tile_ij_mapping.size),
                    cell._atm.ctypes, ctypes.c_int(cell.natm),
                    cell._bas.ctypes, ctypes.c_int(cell.nbas), cell._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                if log.verbose >= logger.DEBUG1:
                    t1, t1p = log.timer_debug1(f'processing {llll}, tasks = {info[1]}', *t1), t1
                    if llll not in timing_collection:
                        timing_collection[llll] = 0
                    timing_collection[llll] += t1[1] - t1p[1]
                    kern_counts += 1

        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_collection.items():
                log.debug1('%s wall time %.2f', llll, t)

        #:out = einsum('pqLG,pi,qj->LGij', out, coeff, coeff)
        out = contract('pqLG,pi->qLGi', out, coeff)
        out = contract('qLGi,qj->LGij', out, coeff)
        if kptjs is not None:
            kptjs = cp.asarray(kptjs, order='C').reshape(-1,3)
            expLk = cp.exp(1j*cp.dot(bvkmesh_Ls, kptjs.T))
            out = contract('Lk,LGij->kGij', expLk, out)

        #??? symmetrize?
        if aosym == 's1hermi':
            for i in range(1, ni):
                out[:,:,:i,i] = out[:,:,i,:i]

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
        ('shl_pair_img_offsets', ctypes.c_void_p),
    ]

def init_constant(cell):
    g_idx, offsets = g_pair_idx()
    libvhf_pbc.PBC_FT_init_constant(
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
    nps = l_ctr_pattern[:,1]
    unit = g_size*3
    counts = shm_size // (unit*16)
    counts = _nearest_power2(counts)
    FIXME
    if counts < nGv:
        n = min(THREADS, counts)
    else:
        n = _nearest_power2(nGv*2-1)
    gout_stride = THREADS // n
    while gout_stride < 16 and gout_size / (gout_stride*GOUT_WIDTH) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride
