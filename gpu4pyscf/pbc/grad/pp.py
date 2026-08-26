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

import numpy as np
import cupy as cp
from pyscf.gto.mole import ATOM_OF
from pyscf.pbc.lib.kpts_helper import gamma_point
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu, _sorted_fake_cell_vnl

def vppnl_nuc_grad(cell, dm, kpts=None):
    '''Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.

    Uses GPU CUDA kernels for the r^2/r^4 moment integrals at gamma point,
    with CPU fallback via pyscf _int_vnl for multi-k-point calculations.
    '''
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
    nkpts = len(kpts_lst)

    # pattern stores the unique [hl_dim, l] combinations
    fakecell, hl_blocks, pattern, splits = _sorted_fake_cell_vnl(cell)

    intors_d = ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    ppnl_half = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst)
    ppnl_half_ip2 = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst, intors_d, comp=3)
    if len(ppnl_half_ip2[0]) > 0:
        ppnl_half_ip2[0] *= -1

    nao = cell.nao
    dm = cp.asarray(dm).reshape(-1, nao, nao)
    if gamma_point(kpts_lst):
        dm = dm.real
    dm_dmH = dm + dm.transpose(0, 2, 1).conj()

    grad = np.zeros([cell.natm, 3], dtype=cp.complex128)
    dppnl = cp.zeros((nao, 3), dtype=cp.complex128)

    hl_offset = [0] * 3
    for ii, (i0, i1) in enumerate(zip(splits[:-1], splits[1:])):
        hl_dim, l = pattern[ii]
        nd = 2 * l + 1
        hl_block = cp.asarray(np.stack(hl_blocks[i0:i1]))
        n_hl = len(hl_block)

        ilp = cp.empty((nkpts, n_hl, hl_dim, nd, nao), dtype=cp.complex128)
        dilp = cp.empty((nkpts, 3, n_hl, hl_dim, nd, nao), dtype=cp.complex128)
        for i in range(hl_dim):
            p0 = hl_offset[i]
            p1 = p0 + n_hl * nd
            ilp[:,:,i] = ppnl_half[i][:,p0:p1].reshape(nkpts, n_hl, nd, nao)
            dilp[:,:,:,i] = ppnl_half_ip2[i][:,:,p0:p1].reshape(nkpts, 3, n_hl, nd, nao).conj()
            hl_offset[i] = p1

        tmp = contract('nij,knjlq->knilq', hl_block, ilp)
        tmp = contract('knilq,kqp->knilp', tmp, dm_dmH)

        value = contract('kdnilp,knilp->nd', dilp, tmp)
        np.add.at(grad, fakecell._bas[i0:i1, ATOM_OF], value.get())

        dppnl -= contract('kdnilp,knilp->pd', dilp, tmp)

    grad -= groupby(cell._bas[:,ATOM_OF], dppnl.get(), 'sum')

    grad_max_imag = np.max(np.abs(grad.imag))
    if grad_max_imag >= 1e-8:
        logger.warn(cell, f"Large imaginary part ({grad_max_imag:e}) from pseudopotential non-local term gradient.")
    grad = grad.real

    return grad
