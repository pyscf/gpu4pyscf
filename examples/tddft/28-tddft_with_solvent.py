#!/usr/bin/env python
# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
#  Example of TDDFT with solvent
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
mf = gpu4pyscf.dft.RKS(mol, xc=xc).PCM()
mf.with_solvent.method = 'IEFPCM'
mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
mf.with_solvent.eps = 78
mf.grids.level = 5
mf.kernel() # -76.476456106979

# Compute TDDFT and TDA excitation energy
print('------------------- vertical exitation TDA (non-equilibrium solvent)-----------------------------')
td = mf.TDDFT(equilibrium_solvation=False).set(nstates=5)
e_tddft = td.kernel()[0] # [ 8.03553836 10.073618   10.20203543 12.36009825 13.83374465]
# print('5 TDDFT excitation energy by GPU4PySCF')
# print(e_tddft)

print('------------------- vertical excitation TDA (equilibrium solvent)-----------------------------')
td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
e_tda = td.kernel()[0] # [ 7.99456768 10.06329607 10.08523514 12.30675317 13.64298135]
# print('5 TDA excitation energy by GPU4PySCF')
# print(e_tda)

print('The gradient of first TDA excitation energy by GPU4PySCF with equilibrium solvent')
g = td.nuc_grad_method()
g.kernel()
"""
--------- PCMTDA gradients for state 1 ----------
         x                y                z
0 O    -0.0000000000     0.0000000000    -0.0836461563
1 H     0.0601539524    -0.0000000000     0.0418233032
2 H    -0.0601539524    -0.0000000000     0.0418233032
----------------------------------------------
"""

# ValueError: equilibrium_solvation must be specified
# td = mf.TDA().set(nstates=5)
# e_tda = td.kernel()[0]