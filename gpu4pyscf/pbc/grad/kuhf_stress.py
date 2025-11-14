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

'''
Stress tensor
'''

import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import kuhf as kuhf_grad
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells, ewald
from gpu4pyscf.pbc.grad.krhf_stress import get_nuc, get_veff

def kernel(mf_grad):
    '''Compute the energy derivatives for strain tensor (e_ij)

                1  d E
    sigma_ij = --- ------
                V  d e_ij

    sigma is a asymmetric 3x3 matrix. The symmetric stress tensor in the 6 Voigt
    notation can be transformed from the asymmetric stress tensor

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

    See K. Doll, Mol Phys (2010), 108, 223
    '''
    assert isinstance(mf_grad, kuhf_grad.Gradients)
    mf = mf_grad.base

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    sigma = ewald(cell)

    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]
    kpts = mf.kpts
    int1e_opt_v2 = int1e._Int1eOptV2(cell)
    sigma -= int1e_opt_v2.get_ovlp_strain_deriv(dme0_sf, kpts)

    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)
    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            t1 = int1e.int1e_kin(cell1, kpts1)
            t2 = int1e.int1e_kin(cell2, kpts2)
            t1 = cp.einsum('kij,kji->', t1, dm0_sf).real
            t2 = cp.einsum('kij,kji->', t2, dm0_sf).real
            sigma[x,y] += (t1 - t2).get() / (2*disp) / nkpts

    sigma += get_nuc(mf_grad, cell, dm0_sf, kpts)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_veff(mf_grad, cell, dm0, kpts)
    t0 = log.timer_debug1('vhf derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma
