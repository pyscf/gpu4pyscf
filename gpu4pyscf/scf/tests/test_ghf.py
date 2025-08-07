# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import pyscf

class KnownValues(unittest.TestCase):
    def test_ghf_scf(self):
        mol = pyscf.M(atom='''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''', basis = 'cc-pvdz')
        mf = mol.GHF().to_gpu()
        assert mf.device == 'gpu'
        e_tot = mf.kernel()
        e_ref = mf.to_cpu().kernel()
        assert abs(e_tot - e_ref) < 1e-5

    #def test_ghf_x2c(self):
    #    pass

if __name__ == "__main__":
    print("Full Tests for ghf")
    unittest.main()
