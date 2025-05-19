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

'''
TDDFT excited state geometry optimization
'''

import pyscf
from gpu4pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='631g')

mf = rks.RKS(mol, xc='b3lyp')
mf.kernel() 

td = mf.TDA().set(nstates=5)
assert td.device == 'gpu'
e_tda = td.kernel()[0]

print('The gradient of first TDA excitation energy by GPU4PySCF before optimization')
g = td.nuc_grad_method()
g.kernel()

excited_grad = td.nuc_grad_method().as_scanner(state=1)
mol1 = excited_grad.optimizer().kernel()
mol2 = pyscf.geomopt.geometric_solver.optimize(td)

mff = rks.RKS(mol1, xc='b3lyp')
mff.kernel() #  -76.2224050802565
tdf = mff.TDA().set(nstates=5)
output = tdf.kernel()
print('The gradient of first TDA excitation energy by GPU4PySCF after optimization')
excited_gradf = tdf.nuc_grad_method()
excited_gradf.kernel() # [ 1.8664593   1.86646751  6.0627608   6.06276617 10.92296501]

mff = rks.RKS(mol2, xc='b3lyp')
mff.kernel() #  -76.2224050802565
tdf = mff.TDA().set(nstates=5)
output = tdf.kernel()
print('The gradient of first TDA excitation energy by GPU4PySCF after optimization')
excited_gradf = tdf.nuc_grad_method()
excited_gradf.kernel() # [ 1.8664593   1.86646751  6.0627608   6.06276617 10.92296501]
"""
--------- TDA gradients for state 1 ----------
         x                y                z
0 O    -0.0000000000     0.0000000000    -0.0000441423
1 H     0.0001631345    -0.0000000000     0.0000220852
2 H    -0.0001631345    -0.0000000000     0.0000220852
----------------------------------------------
"""

