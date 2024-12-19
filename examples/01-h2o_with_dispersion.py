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

###########################################################
#  Example of DFT with Dispersion correction (dftd3/dftd4)
###########################################################

import numpy as np
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.disp = 'd3bj' # d3zero,  d3bjm,  d4
mf_GPU.grids.level = 5
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50
mf_GPU.conv_tol_cpscf = 1e-6

# Compute Energy
print('------------------- Energy -----------------------------')
e_dft = mf_GPU.kernel()
print('DFT energy by GPU4PySCF')

# Compute Gradient
print('------------------ Gradient ----------------------------')
g = mf_GPU.nuc_grad_method()
g.auxbasis_response = True
g_dft = g.kernel()
print('Gradient by GPU4PySCF')
print(g_dft)

# Compute Hessian
print('------------------- Hessian -----------------------------')
h = mf_GPU.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
mass = [15.99491, 1.00783, 1.00783]
for i in range(3):
    for j in range(3):
        h_dft[i,j] = h_dft[i,j]/np.sqrt(mass[i]*mass[j])
n = h_dft.shape[0]
h_dft = h_dft.transpose([0,2,1,3]).reshape(3*n,3*n)
print('Hessian by GPU4PySCF')
print(h_dft)
