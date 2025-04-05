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

import unittest
import numpy as np
import pyscf
from pyscf import lib, gto
from gpu4pyscf import scf

def setUpModule():
    global mol
    atom = '''
I 0 0 0
I 1 0 0
'''
    bas='def2-svp'
    mol = pyscf.M(atom=atom, basis=bas, ecp=bas)
    mol.output = '/dev/null'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf(self):
        mf = scf.RHF(mol)
        mf.max_cycle = 10
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.abs(e_tot - -578.9674228876) < 1e-5

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()
