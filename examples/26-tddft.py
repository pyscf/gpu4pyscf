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

###################################
#  Example of TDDFT
###################################

import pyscf
import gpu4pyscf
from gpu4pyscf import tdscf


atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')

xc = 'b3lyp'
mf = gpu4pyscf.dft.RKS(mol, xc=xc)
mf.grids.level = 5
mf.kernel() # -76.4666495331835

# Compute TDDFT and TDA excitation energy
print('------------------- TDDFT -----------------------------')
td = mf.TDDFT().set(nstates=5)
assert td.device == 'gpu'
e_tddft = td.kernel()[0] # [ 7.51061679  9.42244054  9.76601486 11.7438481  13.59746126]
# print('5 TDDFT excitation energy by GPU4PySCF')
# print(e_tddft)

print('------------------- TDA -----------------------------')
td = mf.TDA().set(nstates=5)
assert td.device == 'gpu'
e_tda = td.kernel()[0] # [ 7.53380573  9.42804505  9.81320363 11.78176645 13.62814473]
# print('5 TDA excitation energy by GPU4PySCF')
# print(e_tda)

print('The gradient of first TDA excitation energy by GPU4PySCF')
g = td.nuc_grad_method()
g.kernel()
"""
--------- TDA gradients for state 1 ----------
         x                y                z
0 O    -0.0000000000     0.0000000000    -0.0901093713
1 H     0.0598455827    -0.0000000000     0.0450549086
2 H    -0.0598455827    -0.0000000000     0.0450549086
----------------------------------------------
"""