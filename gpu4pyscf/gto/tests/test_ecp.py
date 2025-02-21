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
     H    D
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
     H    G
            6.491000E-01           1.0000000   
                                ''')

    mol1 = gto.M(
        atom="""
        Cu 0.0 0.0 0.0
        """,
        basis="sto3g",  # A basis set that includes an ECP for Cu
        ecp="crenbl",    # Assign the corresponding ECP
        spin=1,           # Copper (Cu) has an unpaired electron
        charge=0,
        cart=1,
        output = '/dev/null'
    )

    mol2 = gto.M(
        atom='''
            Na 0.5 0.5 0.
            H  0.  1.  1.
            ''',
        output = '/dev/null',
        basis={'Na': cu1_basis, 'H': cu1_basis},
        ecp = {'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                    0.0
Na S
2      13.652203             732.2692
2       6.826101              26.484721
Na P
2      10.279868             299.489474
2       5.139934              26.466234
Na D
2       7.349859             124.457595
2       3.674929              14.035995
Na F
2       3.034072              21.531031
Na G
2       4.808857             -21.607597
                                         ''')})
def tearDownModule():
    global mol1, mol2
    mol1.stdout.close()
    mol2.stdout.close()
    del mol1, mol2

class KnownValues(unittest.TestCase):
    def test_ecp_cart(self):
        h1_cpu = mol1.intor('ECPscalar_cart')
        h1_gpu = get_ecp(mol1)
        assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-8
    
    def test_ecp_sph(self):
        h1_cpu = mol2.intor('ECPscalar_sph')
        h1_gpu = get_ecp(mol2)
        print(h1_cpu[:3,:3])
        print(h1_gpu[:3,:3])
        assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-8
    
if __name__ == "__main__":
    print("Full Tests for ECP Integrals")
    unittest.main()
