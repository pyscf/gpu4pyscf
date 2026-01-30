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

import cupy as cp
import numpy as np
import unittest
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import dft as pbcdft
from gpu4pyscf.pbc.dft.kucdft import CDFT_KUKS


def setUpModule():
    global cell
    cell = pbcgto.Cell()
    cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
    cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
    cell.basis=('gth-szv', [[2, [0.5, 1]]])
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
        kpts = cell.make_kpts([3, 2, 1])
        charge_constraints = ([ [0] ], [ 4.1 ])
        mf = CDFT_KUKS(cell, kpts, charge_constraints=charge_constraints)
        mf.minao_ref = 'gth-szv'
        mf.xc = 'pbe'
        mf.kernel()

        dm_kpts = mf.make_rdm1()
        proj = mf.constraint_projectors[0]
        k_weights = cp.full(len(kpts), 1.0/len(kpts))
        val_a = cp.einsum('kij,kji->k', dm_kpts[0], proj)
        val_b = cp.einsum('kij,kji->k', dm_kpts[1], proj)
        pop = cp.sum((val_a + val_b) * k_weights)

        ref_energy = -11.176954803067169
        ref_lagrange = -0.03582391
        ref_mo_energy_00 = np.array([-0.29843439,  0.4561228,  0.45623718,  0.45707317,  0.66672637,  0.6673692,
            0.66765121,  1.04564717,  1.47638186,  1.47863224,  1.89201068,  1.89239729,
            1.89337587,  2.32346589,  2.32774572,  2.32968555,  3.22073326,  3.22145838])
        ref_mo_energy_00_cononical = np.array([-0.28055544,  0.47357079,  0.47367915,  0.47453062,  0.68461715,  0.68523977,
                0.68553677,  1.0635916,  1.4763818,  1.47863232,  1.89215314,  1.89254046,
                1.89351736,  2.32381348,  2.32809319,  2.33003982,  3.22073322,  3.22145843])
        mo_energy_00_cononical = mf.get_canonical_mo()[0][0,0]
        self.assertAlmostEqual(mf.e_tot, ref_energy, 6)
        self.assertAlmostEqual(mf.v_lagrange[0], ref_lagrange, 6)
        self.assertAlmostEqual(float(pop.real), 4.1, 6)
        np.testing.assert_allclose(mf.mo_energy[0][0].get(), ref_mo_energy_00, atol=1e-6)
        np.testing.assert_allclose(mo_energy_00_cononical, ref_mo_energy_00_cononical, atol=1e-6)

        mf.analyze()

if __name__ == '__main__':
    print("Full Tests for pbc.dft.kucdft")
    unittest.main()
