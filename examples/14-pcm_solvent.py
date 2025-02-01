#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

############################################################
#  Example of DFT with PCM solvent model
############################################################

import numpy as np
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='HYB_GGA_XC_B3LYP').density_fit()
mf = mf.PCM()
mf.grids.atom_grid = (99,590)
mf.with_solvent.lebedev_order = 29  # 302 Lebedev grids
mf.with_solvent.method = 'IEF-PCM'   # Can be C-PCM, SS(V)PE, COSMO
mf.with_solvent.eps = 78.3553        # Dielectric constant
mf.kernel()

gradobj = mf.nuc_grad_method()
f = gradobj.kernel()

hessobj = mf.Hessian()
hess = hessobj.kernel()

# mass weighted hessian
mass = [15.99491, 1.00783, 1.00783]
for i in range(3):
    for j in range(3):
        hess[i,j] = hess[i,j]/np.sqrt(mass[i]*mass[j])


