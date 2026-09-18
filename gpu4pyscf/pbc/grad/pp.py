#!/usr/bin/env python
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

import ctypes
import numpy as np
import cupy as cp
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from gpu4pyscf.lib.cupy_helper import contract, asarray
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu, _sorted_fake_cell_vnl
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.int1e import libpbc
from gpu4pyscf.pbc.tools import k2gamma

def vppnl_nuc_grad(cell, dm, kpts=None):
    '''Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    return ppnl_derivatives(cell, dm, kpts)[:-3]

def ppnl_derivatives(cell, dm, kpts=None):
    '''Nonlocal GTH atomic and strain derivatives, averaged over k-points.

    Returns a real (natm+3, 3) array: nuclear gradients followed by dE/d(strain_xy)
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = np.reshape(kpts, (-1, 3))
    is_gamma_point = gamma_point(kpts)
    nkpts = len(kpts)

    nao = cell.nao
    dm = cp.asarray(dm).reshape(-1, nao, nao)
    if len(dm) != nkpts:
        raise ValueError('Expected one density matrix per k-point')
    if is_gamma_point:
        dm = dm.real
    dm_dmH = dm + dm.conj().transpose(0, 2, 1)

    grad_sigma = np.zeros((cell.natm+3, 3))

    fakecell, hl_blocks, pattern, splits = _sorted_fake_cell_vnl(cell)
    if not hl_blocks:
        return grad_sigma

    sorted_cell = SortedGTO.from_cell(cell, decontract=True)
    ppnl_half = _int_vnl_gpu(sorted_cell, fakecell, hl_blocks, kpts)

    derivative_kernels = (
        ('PBCovlp_cross_derivatives', (0, 1)),
        ('PBCint1e_r2_origi_derivatives', (2, 1)),
        ('PBCint1e_r4_origi_derivatives', (4, 1)),
    )

    bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
    expLk = cp.exp(-1j*asarray(bvkmesh_Ls).dot(asarray(kpts).T))
    dtype = np.float64 if is_gamma_point else np.complex128

    hl_offset = [0] * 3
    for (hl_dim, l), i0, i1 in zip(pattern, splits[:-1], splits[1:]):
        if hl_dim == 0:
            continue
        nd = 2 * l + 1
        n_hl = i1 - i0
        hl_block = asarray(np.stack(hl_blocks[i0:i1]))
        ilp = cp.empty((hl_dim, nkpts, n_hl, nd, nao), dtype=dtype)
        for rn in range(hl_dim):
            p0 = hl_offset[rn]
            p1 = p0 + n_hl * nd
            ilp[rn] = ppnl_half[rn][:,p0:p1].reshape(nkpts, n_hl, nd, nao)
            hl_offset[rn] = p1
        tmp = contract('nij,jknlq->iknlq', hl_block, ilp)
        weights = contract('iknlq,kqp->iknlp', tmp, dm_dmH, out=ilp)

        if not is_gamma_point:
            weights = contract('Lk,iknlp->iLnlp', expLk, weights).real

        pcell = fakecell.copy(deep=False)
        pcell._bas = fakecell._bas[i0:i1]
        opt = int1e.CrossInt1e(pcell, sorted_cell, bvk_kmesh)
        for rn in range(hl_dim):
            kern, deriv = derivative_kernels[rn]
            as_dm = weights[rn].reshape(-1, n_hl*nd, nao)
            grad_sigma += _derivatives_intor(opt, as_dm, kern, deriv) / nkpts
    return grad_sigma

def _derivatives_intor(cross_int1e, dm, kern, deriv):
    '''Contract Re[dm * conjugate(dI)] for a rectangular cross integral.
    '''
    cell = cross_int1e.cell
    assert dm.ndim == 3
    assert dm.dtype == np.float64
    nkpts = len(dm)

    tmp = cross_int1e.cell2.apply_C_dot(dm, axis=2)
    dm = cross_int1e.cell1.apply_C_dot(tmp, axis=1)
    dm = cp.asarray(dm, order='C')

    gout_stride_lookup, shm_size = int1e._gout_stride_lookup_table(cell, deriv)
    nbatches_shl_pair = len(cross_int1e.shl_pair_offsets) - 1

    if nbatches_shl_pair == 0:
        return np.zeros([cell.natm+3, 3])

    grad = cp.zeros((cell.natm, 3))
    sigma = cp.zeros((3, 3))
    drv = getattr(libpbc, kern)
    err = drv(
        ctypes.cast(grad.data.ptr, ctypes.c_void_p),
        ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.byref(cross_int1e.int1e_envs),
        ctypes.c_int(shm_size),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(cross_int1e.shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(cross_int1e.bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride_lookup.data.ptr, ctypes.c_void_p),
        ctypes.c_int(cross_int1e.cell1.nao),
        ctypes.c_int(cross_int1e.cell2.nao))
    if err != 0:
        raise RuntimeError(f'{kern} failed')

    # CrossInt1e concatenates the projector and AO atom lists.
    natm = cross_int1e.cell1.natm
    if natm != cross_int1e.cell2.natm:
        raise ValueError(
            'fakecell for ppnl must have the same number of atoms as the AO cell')
    grad = (grad[:natm] + grad[natm:]).get()
    grad_sigma = np.vstack([grad, sigma.get()])
    return grad_sigma
