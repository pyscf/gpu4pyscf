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

import numpy
import cupy as cp
from gpu4pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl

# The following function is derived from pyscf/pbc/gto/pseudo/pp_int.py.
# It's updated to support k-point sampling after pyscf>2.11.0,
# however we want gpu4pyscf to be compatible with older versions of pyscf,
# particularly pyscf==2.8.0, the version used by github CI.
# The integral computation uses GPU CUDA kernels for the r^2/r^4 moment
# integrals, and the contraction is done with cupy.

def vppnl_nuc_grad(cell, dm, kpts=None):
    '''Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix. Fully GPU-accelerated.
    '''
    if kpts is None:
        kpts_lst = numpy.zeros((1, 3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1, 3))

    dm = cp.asarray(dm)
    fakecell, hl_blocks = fake_cell_vnl(cell)

    intors_d = ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    if gamma_point(kpts_lst):
        from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu
        ppnl_half = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst)
        ppnl_half_ip2 = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts_lst, intors_d, comp=3)
    else:
        from pyscf.pbc.gto.pseudo.pp_int import _int_vnl
        ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
        ppnl_half_ip2 = _int_vnl(cell, fakecell, hl_blocks, kpts_lst, intors_d, comp=3)
    if len(ppnl_half_ip2[0]) > 0:
        for k in range(len(kpts_lst)):
            ppnl_half_ip2[0][k] *= -1

    # Move integral arrays to GPU
    def _to_gpu(arrs):
        return [cp.asarray(a) if len(a) > 0 else a for a in arrs]
    ppnl_half = _to_gpu(ppnl_half)
    ppnl_half_ip2 = _to_gpu(ppnl_half_ip2)

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()

    dm = dm.reshape(-1, nao, nao)
    if gamma_point(kpts_lst):
        dm = dm.real
    dm_dmH = dm + dm.transpose(0, 2, 1).conj()

    grad = cp.zeros([cell.natm, 3], dtype=cp.complex128)
    dppnl = cp.zeros((nkpts, 3, nao, nao), dtype=cp.complex128)

    for k in range(nkpts):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            hl_gpu = cp.asarray(hl)

            ilp = cp.zeros((hl_dim, nd, nao), dtype=cp.complex128)
            dilp = cp.zeros((hl_dim, 3, nd, nao), dtype=cp.complex128)
            for i in range(hl_dim):
                p0 = offset[i]
                if len(ppnl_half[i]) > 0:
                    ilp[i] = ppnl_half[i][k, p0:p0+nd]
                if len(ppnl_half_ip2[i]) > 0:
                    dilp[i] = ppnl_half_ip2[i][k, :, p0:p0+nd]
                offset[i] = p0 + nd

            # dppnl_k[d,p,q] = sum_{i,j,l} dilp[i,d,l,p].conj() * hl[i,j] * ilp[j,l,q]
            dppnl_k = cp.einsum('idlp,ij,jlq->dpq', dilp.conj(), hl_gpu, ilp)
            dppnl[k] += dppnl_k

            i_pp_atom = fakecell._bas[ib, 0]
            grad[i_pp_atom] += cp.einsum('dpq,qp->d', dppnl_k, dm_dmH[k])

    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia][2:]
        grad[ia] -= cp.einsum('kdpq,kqp->d', dppnl[:, :, p0:p1, :],
                              dm_dmH[:, :, p0:p1])

    grad = grad.get()

    grad_max_imag = numpy.max(numpy.abs(grad.imag))
    if grad_max_imag >= 1e-8:
        logger.warn(cell, f"Large imaginary part ({grad_max_imag:e}) from pseudopotential non-local term gradient.")
    grad = grad.real

    return grad
