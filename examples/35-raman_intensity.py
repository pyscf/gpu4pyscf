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
Raman scattering activity and depolarization ratio
'''

import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import raman

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

basis = 'def2-SVP'

mol = pyscf.M(atom = atom, basis = basis)

mf = mol.RKS(xc='wB97M-V').to_gpu()
energy = mf.kernel()
print(f"SCF energy = {energy}")
# Reference energy = -76.3254651015359

frequencies, raman_intensities, depolarization_ratio = raman.eval_raman_intensity(mf)

print(frequencies)
print(raman_intensities)
print(depolarization_ratio)

print('------------------- Raman frequncy, intensity and depolarization ratio ---------------------------')
for i in range(frequencies.shape[0]):
    print(f"{i}-th mode: frequency = {frequencies[i]:7.2f} (cm^-1), "
          f"Raman scattering activity = {raman_intensities[i]:7.3f} (A^4/AMU), "
          f"depolarization ratio = {depolarization_ratio[i]:6.4f}")
### Reference output:
# 0-th mode: frequency = 1602.59 (cm^-1), Raman scattering activity =   6.563 (A^4/AMU), depolarization ratio = 0.5394
# 1-th mode: frequency = 3927.37 (cm^-1), Raman scattering activity =  74.157 (A^4/AMU), depolarization ratio = 0.1564
# 2-th mode: frequency = 4033.36 (cm^-1), Raman scattering activity =  35.239 (A^4/AMU), depolarization ratio = 0.7500
