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
from gpu4pyscf.pbc.grad import rhf as rhf_grad
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells, ewald
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.df import aft, aft_jk

ALIGNED = 256

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
    assert isinstance(mf_grad, rhf_grad.Gradients)
    mf = mf_grad.base

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    sigma = ewald(cell)

    int1e_opt_v2 = int1e._Int1eOptV2(cell)
    sigma -= int1e_opt_v2.get_ovlp_strain_deriv(dme0)

    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = int1e.int1e_kin(cell1)[0]
            t2 = int1e.int1e_kin(cell2)[0]
            t1 = cp.einsum('ij,ji->', t1, dm0)
            t2 = cp.einsum('ij,ji->', t2, dm0)
            sigma[x,y] += (t1 - t2) / (2*disp)

    sigma += get_nuc(mf_grad, cell, dm0)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_veff(mf_grad, cell, dm0)
    t0 = log.timer_debug1('vhf derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma

def get_veff(mf_grad, cell, dm):
    '''Strain derivatives for Coulomb and exchange energy with k-point samples
    '''
    mf = mf_grad.base
    with_rsjk = mf.rsjk
    if with_rsjk is not None:
        assert isinstance(with_rsjk, PBCJKMatrixOpt)
        if with_rsjk.supmol is None:
            with_rsjk.build()
        sigma = with_rsjk._get_ejk_sr_strain_deriv(dm, exxdiv=mf.exxdiv)
        sigma+= with_rsjk._get_ejk_lr_strain_deriv(dm, exxdiv=mf.exxdiv)
    elif isinstance(mf.with_df, aft.AFTDF):
        sigma = aft_jk.get_ej_strain_deriv(mf.with_df, dm)
        sigma -= aft_jk.get_ek_strain_deriv(mf.with_df, dm, exxdiv=mf.exxdiv) * .5
    else:
        raise NotImplementedError(f'Stress tensor for KHF for {mf.with_df}')
    return sigma

def get_nuc(mf_grad, cell, dm):
    '''Strain derivatives for Coulomb and Exc at gamma point
    '''
    from gpu4pyscf.pbc.grad import krhf_stress
    kpts = np.zeros((1, 3))
    return krhf_stress.get_nuc(mf_grad, cell, dm[None], kpts)
