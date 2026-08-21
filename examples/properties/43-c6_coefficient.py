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
C6 coefficient
'''

import pyscf
from gpu4pyscf import dft
from gpu4pyscf.properties.c6 import calc_c6

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

# Basis set must conatins diffuse functions and polarization functions
bas='def2svpd'

mol = pyscf.M(atom=atom, basis=bas)

mf = dft.RKS(mol, xc='b3lyp').density_fit()
e_gpu = mf.kernel() # -76.380311497689

td_a = mf.TDDFT()
td_b = mf.TDDFT()
c6_val = calc_c6(td_a, td_b, n_grid=20) 
print("Calculated C6 coefficient:", c6_val)