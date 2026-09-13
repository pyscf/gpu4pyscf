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
        ni = mf._numint
        xc = getattr(mf, 'xc', 'HF')
        if xc.upper() == 'HF':
            omega, k_lr, k_sr = 0, 1, 1
        else:
            omega, k_lr, k_sr = ni.rsh_and_hybrid_coeff(mf.xc)
        j_factor = 1

        # TODO: handle all-electron+GGA and pseudo+GGA differently
        # pseudo+GGA does not need to evaluate the gradients with PBCJKMatrixOpt
        de = np.zeros([self.cell.natm+3, 3])
        spin = 0 if dm.ndim == 2 else 1
        if isinstance(ni, multigrid.MultiGridNumIntBase):
            # Match pbc/dft/rks.py: multigrid supplies J even when exchange uses GDF.
            de = ni.energy_derivatives(
                xc, dm, spin=spin, with_j=True, with_nuc=True)
            j_factor = 0
        elif xc.upper() != 'HF':
            from gpu4pyscf.pbc.grad.krks import get_vxc, get_vxc_full_response
            grids = self.grids or mf.grids
            if grids.coords is None:
                grids.build()
            if self.grid_response:
                assert isinstance(grids, BeckeGrids), "Only Becke grid requires grid response"
                fn = get_vxc_full_response
            else:
                fn = get_vxc
            cell = self.cell
            de[:-3] = fn(ni, cell, grids, xc, dm[None], np.zeros((1, 3)))
            if isinstance(grids, BeckeGrids):
                de[-3:] = np.nan
            else:
                de[-3:] = ni.energy_strain_gradient(
                    xc, dm, spin=0, with_j=False, with_nuc=False)

        if j_factor != 0 or k_sr != 0 or k_lr != 0:
            de += rhf._get_ejk_derivatives(mf, dm, None, j_factor, omega, k_lr, k_sr)
        return de
