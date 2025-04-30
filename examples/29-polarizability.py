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
Static polarizability (unit Bohr^3)
'''

import pyscf
from gpu4pyscf.properties import polarizability

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='631g'

mol = pyscf.M(atom=atom, basis=bas)

mf = mol.RKS(xc='b3lyp').to_gpu()
e_gpu = mf.kernel() #  -76.3849465432042
polar_gpu = polarizability.eval_polarizability(mf)
print('------------------- Polarizability -----------------------------')
print(polar_gpu)
"""
[[ 6.96413065e+00  9.60315894e-18 -2.25792304e-13]
 [ 9.60315894e-18  1.48264155e+00 -6.84920815e-15]
 [-2.25792304e-13 -6.84920815e-15  4.81230498e+00]]
"""
