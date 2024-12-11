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
from pyscf import gto
from pyscf.data import radii
from gpu4pyscf import scf
from gpu4pyscf.pop import esp

def setUpModule():
    global mol0, mol1
    mol0 = gto.Mole()
    mol0.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol0.basis = 'sto3g'
    mol0.output = '/dev/null'
    mol0.build(verbose=0)

    mol1 = gto.Mole()
    mol1.atom = """ C   1.45051389  -0.06628932   0.00000000
     H   1.75521613  -0.62865986  -0.87500146
     H   1.75521613  -0.62865986   0.87500146
     H   1.92173244   0.90485897   0.00000000
     C  -0.04233122   0.09849378   0.00000000
     O  -0.67064817  -1.07620915   0.00000000
     H  -1.60837259  -0.91016601   0.00000000
     O  -0.62675864   1.13160510   0.00000000"""
    mol1.basis = '6-31gs'
    mol1.cart = True
    mol1.output = '/dev/null'
    mol1.build(verbose=0)

def tearDownModule():
    global mol0, mol1
    mol0.stdout.close()
    mol1.stdout.close()
    del mol0, mol1

class KnownValues(unittest.TestCase):
    def test_esp_charge(self):
        mf = scf.RHF(mol0)
        mf.kernel()
        dm = mf.make_rdm1()
        q = esp.esp_solve(mol0, dm)
        q_ref = np.asarray([-0.61468718,  0.30747624,  0.30721094])
        assert np.linalg.norm(q - q_ref) < 1e-4

    def test_resp_charge(self):
        ''' The reference data is taken from 
        https://github.com/cdsgroup/resp/blob/master/resp/tests/test_resp.py
        '''
        mf = scf.RHF(mol1)
        mf.kernel()

        dm = mf.make_rdm1()

        q = esp.resp_solve(mol1, dm, grid_density=1.0*radii.BOHR**2, hfree=True)
        q_ref = np.array([-0.294974,  0.107114,  0.107114,  0.084795,
                        0.803999, -0.661279,  0.453270, -0.600039])
        assert np.linalg.norm(q - q_ref) < 1e-5

        sum_constraints = []
        for i in range(4,8):
            sum_constraints.append([q[i], [i]])
        equal_constraints = [[1,2,3]]

        q = esp.resp_solve(mol1, dm, resp_a = 1e-3,
                    grid_density=1.0*radii.BOHR**2, hfree=True,
                    sum_constraints=sum_constraints,
                    equal_constraints=equal_constraints)
        q_ref = np.array([-0.290893,  0.098314,  0.098314,  0.098314,
                        0.803999, -0.661279,  0.453270, -0.600039])
        assert np.linalg.norm(q - q_ref) < 1e-5

if __name__ == "__main__":
    print("Full Tests for population analysis")
    unittest.main()
    
