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

import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import rks


atom = '''
I 0 0 0
I 1 0 0
'''
bas='def2-tzvpp'
grids_level = 7

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, ecp=bas)
    mol.output='/dev/null'
    mol.verbose = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.grids.prune = None
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        assert np.allclose(e_tot, -582.7625143308, rtol=1e-8)

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()
