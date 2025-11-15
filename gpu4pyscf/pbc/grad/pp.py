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
from gpu4pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl, _int_vnl, _contract_ppnl_nuc_grad

# The following function is copied from pyscf/pbc/gto/pseudo/pp_int.py
# It's updated to support k-point sampling after pyscf>2.11.0,
# however we want gpu4pyscf to be compatable with older version of pyscf,
# particularly pyscf==2.8.0, the version used by github CI.
# So, we made a copy.

def vppnl_nuc_grad(cell, dm, kpts=None):
    '''
    Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    fakecell, hl_blocks = fake_cell_vnl(cell)
    intors = ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    ppnl_half_ip2 = _int_vnl(cell, fakecell, hl_blocks, kpts_lst, intors, comp=3)
    # int1e_ipovlp computes ip1 so multiply -1 to get ip2
    if len(ppnl_half_ip2[0]) > 0:
        for k, kpt in enumerate(kpts_lst):
            ppnl_half_ip2[0][k] *= -1

    if gamma_point(kpts_lst):
        grad = _contract_ppnl_nuc_grad(cell, fakecell, dm, hl_blocks,
                                       ppnl_half, ppnl_half_ip2, kpts=kpts)
        grad *= -2
        return grad

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    assert dm.shape == (nkpts, nao, nao)
    dm_dmH = dm + dm.transpose(0,2,1).conj() # bra and ket

    grad = numpy.zeros([cell.natm, 3], order='C', dtype=numpy.complex128)

    buf1 = numpy.empty((3*9*nao), dtype=numpy.complex128)
    buf2 = numpy.empty((3*3*9*nao), dtype=numpy.complex128)

    dppnl = numpy.zeros((nkpts,3,nao,nao), dtype=numpy.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3

        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf1)
            dilp = numpy.ndarray((hl_dim,3,nd,nao), dtype=numpy.complex128, buffer=buf2)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                dilp[i] = ppnl_half_ip2[i][k][:, p0:p0+nd]
                offset[i] = p0 + nd
            dppnl_k = numpy.einsum('idlp,ij,jlq->dpq', dilp.conj(), hl, ilp)
            dppnl[k] += dppnl_k

            i_pp_atom = fakecell._bas[ib,0]
            grad[i_pp_atom] += numpy.einsum('dpq,qp->d', dppnl_k, dm_dmH[k])

    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia][2:]
        grad[ia] -= numpy.einsum('kdpq,kqp->d', dppnl[:,:,p0:p1,:], dm_dmH[:,:,p0:p1])

    grad_max_imag = numpy.max(numpy.abs(grad.imag))
    if grad_max_imag >= 1e-8:
        logger.warn(cell, f"Large imaginary part ({grad_max_imag:e}) from pseudopotential non-local term gradient.")
    grad = grad.real

    return grad
