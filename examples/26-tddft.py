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
e_tddft = td.kernel()[0] # [ 7.51061148  9.42243504  9.76601005 11.74384344 13.5974535 ]
print('5 TDDFT excitation energy by GPU4PySCF')

print('------------------- TDA -----------------------------')
td = mf.TDA().set(nstates=5)
assert td.device == 'gpu'
e_tda = td.kernel()[0] # [ 7.53380573  9.42804505  9.81320363 11.78176645 13.62814473]
print('5 TDA excitation energy by GPU4PySCF')

