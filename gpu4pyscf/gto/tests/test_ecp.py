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
import cupy as cp
import pyscf
from pyscf import lib, gto, df
from gpu4pyscf.gto.ecp import get_ecp

def setUpModule():
    global mol, mol1, mol2, cu1_basis
    cu1_basis = gto.basis.parse('''
     H    S
           1.8000000              1.0000000
     H    S
           2.8000000              0.0210870             -0.0045400              0.0000000
           1.3190000              0.3461290             -0.1703520              0.0000000
           0.9059000              0.0393780              0.1403820              1.0000000
     H    P
           2.1330000              0.0868660              0.0000000
           1.2000000              0.0000000              0.5000000
           0.3827000              0.5010080              1.0000000
     H    D
           0.3827000              1.0000000
     H    F
           2.1330000              0.1868660              0.0000000
           0.3827000              0.2010080              1.0000000
                                   ''')

    mol = gto.M(
        atom='''
    Cu1 0. 0. 0.
    Cu 0. 1. 0.
    He 1. 0. 0.
    ''',
    basis={'Cu':'lanl2dz', 'Cu1': cu1_basis, 'He':'sto3g'},
    ecp = {'cu':'lanl2dz'})

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_ecp(self):
        h1 = mol.intor('ECPscalar_cart')
        print(h1.shape)

        h1 = mol.intor('ECPscalar_sph')
        print(h1.shape)

        get_ecp(mol)
if __name__ == "__main__":
    print("Full Tests for ECP Integrals")
    unittest.main()
