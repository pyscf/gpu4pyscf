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

'''
Compute J and K matrices separately. The J matrix is evaluated using J-engine.
'''

import pyscf
from gpu4pyscf import scf
from gpu4pyscf.scf import jk

mol = pyscf.M(
atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
basis='def2-tzvp',
verbose=5
)

def get_veff(self, mol, dm, *args, **kwargs):
    vj = jk.get_j(mol, dm[0] + dm[1], hermi=1)
    _, vk = jk.get_jk(mol, dm, hermi=1, with_j=False)
    return vj - vk

scf.uhf.UHF.get_veff = get_veff

mf = mol.UHF().to_gpu().run()
