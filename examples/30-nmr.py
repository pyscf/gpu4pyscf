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
#  Example of NMR shielding constant
###################################

import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.properties import shielding

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='631g', max_memory=32000)

mf = rks.RKS(mol, xc='b3lyp')
e_gpu = mf.kernel() # -76.3849465432042
msc_d, msc_p = shielding.eval_shielding(mf)
msc = (msc_d + msc_p).get()
print('------------------- NMR shielding constant -----------------------------')
for i in range(mol.natm):
    print(f"Isotropic NMR shielding constant for {i}-th atom is {msc[i].trace()/3:.4f}")
"""
Isotropic NMR shielding constant for 0-th atom is 318.1915
Isotropic NMR shielding constant for 1-th atom is 33.0777
Isotropic NMR shielding constant for 2-th atom is 33.0777
"""