#!/usr/bin/env python
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
#  Example of DFT with Custom Dispersion correction (dftd3/dftd4)
###########################################################

import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')

print('Dispersion convention examples')
print('disp accepts d3/d4 names, or d4:method for method-specific parameters')
print('-----------------------------------------------')

cases = [
    ('B3LYP + D3BJ', 'b3lyp', 'd3bj', None),
    ('B3LYP + D3BJ', 'b3lyp', 'd3bj:b3lyp', None),
    ('wB97X-V', 'wb97x-v', None, 'vv10'),
    ('wB97X-3c', 'wb97x-v', 'd4:wb97x-3c', 0),
    ('wB97X-D4', 'wb97x-v', 'd4:wb97x', 0),
    ('wB97X-D4rev', 'wb97x-v', 'd4:wb97x-rev', 0),
]

for label, xc, disp, nlc in cases:
    print(f'Case: {label}')
    mf = rks.RKS(mol, xc=xc)
    if nlc is not None:
        mf.nlc = nlc
        print(f'  xc={xc} disp={disp} nlc={nlc}')
    else:
        print(f'  xc={xc} disp={disp}')
    mf.disp = disp
    mf.grids.level = 5
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-12
    mf.max_cycle = 50
    e_tot = mf.kernel()
    print(f'  e_tot={e_tot}')
