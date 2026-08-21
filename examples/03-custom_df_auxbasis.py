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

####################################################
#  Example of DFT with different customized auxbasis
####################################################

import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='dzvp')
auxbasis = {}
from pyscf import df
etb_basis = df.aug_etb(mol, beta=2.0)
for i in range(mol.natm):
    sym = mol.atom_pure_symbol(i)
    if sym not in auxbasis:
        # use even-tempered basis for I and other heavy atoms (nuc_charge > 36)
        nuc_charge = gto.charge(sym)
        if nuc_charge > 36:
            auxbasis[sym] = etb_basis[sym]
        else:
            auxbasis[sym] = 'def2-tzvpp-jkfit'

mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.disp = 'd3bj'
mf_GPU.grids.level = 3
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50

# Compute Energy
print('------------------- Energy -----------------------------')
e_dft = mf_GPU.kernel()
print('DFT energy by GPU4PySCF')
print(e_dft)

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
print('Diagonal entries of Mass-weighted Hessian by GPU4PySCF')
mass = [15.99491, 1.00783, 1.00783]
for i in range(3):
    for j in range(3):
        h_dft[i,j] = h_dft[i,j]/np.sqrt(mass[i]*mass[j])
n = h_dft.shape[0]
h_dft = h_dft.transpose([0,2,1,3]).reshape(3*n,3*n)
print(np.diag(h_dft))
