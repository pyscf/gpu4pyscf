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
from gpu4pyscf.pbc.dft import multigrid, BeckeGrids
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.pbc.grad import rhf

__all__ = ['Gradients']

class Gradients(rhf.Gradients):
    grids = None
    grid_response = False

    _keys = {'grid_response', 'grids'}

    def reset(self, cell=None):
        if self.grids is not None:
            self.grids.reset(cell)
        return rhf.Gradients.reset(self, cell)

    def energy_ee(self, dm):
        '''
        The contribution of electron-electron interactions per cell to the
        nuclear gradients.
        '''
        mf = self.base
        with_df = mf.with_df
        # When the MultiGridNumInt integrator is used, the J term can be
        # evaluated together with the XC term. However, if J is computed using
        # GDF approximate integrals, J from MultiGridNumInt is inconsistent with
        # the GDF-based J. In this case, j_in_xc must be disabled, and the J
        # contribution must be evaluated using the GDF _get_ejk_derivatives function.
        #
        # J matrix is accurately computed when rsjk or j_engine is enabled.
        # In the two cases, J from MultiGridNumInt is theoretically
        # identical to the the J computed using these real-space integral
        # techniques.
        ni = mf._numint
        j_in_xc = not isinstance(with_df, GDF)
        j_factor = 1
        if j_in_xc:
            j_factor = 0
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            omega, k_lr, k_sr = 0, 1, 1
        else:
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = 0
        spin = 0 if dm.ndim == 2 else 1
        if isinstance(ni, multigrid.MultiGridNumIntBase):
            de = ni.energy_derivatives(
                xc, dm, spin=spin, with_j=j_in_xc, with_nuc=True)
        elif xc.upper() != 'HF':
            from gpu4pyscf.pbc.grad.krks import get_vxc, get_vxc_full_response
            if self.grid_response:
                assert isinstance(mf.grids, BeckeGrids), "Only Becke grid requires grid response"
                fn = get_vxc_full_response
            else:
                fn = get_vxc
            cell = self.cell
            de = np.empty([cell.natm+3, 3])
            de[:-3] = fn(ni, mf.cell, mf.grids, xc, dm[None], np.zeros((1, 3)))
            if isinstance(grids, BeckeGrids):
                de[-3:] = np.nan
            else:
                de[-3:] = ni.energy_strain_gradient(
                    xc, dm, spin=spin, with_j=False, with_nuc=True)

        if j_factor != 0 or k_sr != 0 or k_lr != 0:
            de += rhf._get_ejk_derivatives(mf, dm, None, j_factor, omega, k_lr, k_sr)
        return de
