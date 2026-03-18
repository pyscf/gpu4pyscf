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
from gpu4pyscf.pbc.grad import uhf as uhf_grad
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells, ewald
from gpu4pyscf.pbc.grad.rhf_stress import get_nuc, get_veff

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
    assert isinstance(mf_grad, uhf_grad.Gradients)
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
    sigma -= int1e.ovlp_strain_deriv(cell, dme0_sf)
    sigma += int1e.kin_strain_deriv(cell, dm0)
    sigma += get_nuc(mf_grad, cell, dm0_sf)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_veff(mf_grad, cell, dm0)
    t0 = log.timer_debug1('vhf derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma
