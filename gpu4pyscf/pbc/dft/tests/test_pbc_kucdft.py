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
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import dft as pbcdft
from gpu4pyscf.pbc.dft.kucdft import CDFT_KUKS


def setUpModule():
    global cell
    L = 4.
    cell = pbcgto.Cell()
    cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
    cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
    cell.basis='gth-szv'
    cell.pseudo='gth-pade'
    cell.unit = 'B'
    cell.verbose = 4
    cell.output = '/dev/null'
    cell.build()

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):

    def test_atom_cons(self):
        kpts = cell.make_kpts([2, 2, 2])
        charge_constraints = ([ [0] ], [ 4.1 ])
        mf = CDFT_KUKS(cell, kpts, charge_constraints=charge_constraints)
        mf.minao_ref = 'gth-szv'
        mf.xc = 'pbe'
        mf.kernel()

        print(f"Lagrange Multipliers: {mf.v_lagrange}")

        ref_energy = -11.2549563001339
        ref_lagrange = -0.04310747
        ref_mo_energy_00 = np.array([-0.30546736,  0.45683201,  0.45683201,  0.45683201,  0.68759527,
            0.68759527,  0.68759527,  1.03539854])
        self.assertAlmostEqual(mf.e_tot, ref_energy, 6)
        self.assertAlmostEqual(mf.v_lagrange[0], ref_lagrange, 6)
        np.testing.assert_allclose(mf.mo_energy[0][0].get(), ref_mo_energy_00, atol=1e-6)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.kucdft")
    unittest.main()
