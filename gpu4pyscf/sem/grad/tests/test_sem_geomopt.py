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
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.sem.gto.mole import Mole
from gpu4pyscf.sem.scf import hf

atom = 'O 0.0000 0.0000 0.0000; H 0.7570 0.5860 0.0000; H -0.7570 0.5860 0.0000'


def setUpModule():
    global mol
    mol = Mole(
        atom=atom, max_memory=32000, output="/dev/null", verbose=1)
    mol.build()


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):

    def test_scanner(self):
        """scanner should return the same energy/gradient."""
        mf = hf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        g = mf.nuc_grad_method()
        g.kernel()

        scanner = mf.nuc_grad_method().as_scanner()
        e_tot, de = scanner(mol)

        self.assertAlmostEqual(e_tot, mf.e_tot, delta=1e-9)
        assert np.abs(de - g.de).max() < 1e-8

    def test_geomopt(self):
        """produce vanishing gradients."""
        mf = hf.RHF(mol)
        mf.conv_tol = 1e-12
        e0 = mf.kernel()

        mol_eq = optimize(mf, maxsteps=50)

        mf_eq = hf.RHF(mol_eq)
        mf_eq.conv_tol = 1e-12
        e1 = mf_eq.kernel()

        # The optimized structure must not have a higher energy.
        self.assertLessEqual(e1, e0 + 1e-9)

        g_eq = mf_eq.nuc_grad_method().kernel()
        assert np.abs(g_eq).max() < 1e-4

        coords = mol_eq.atom_coords(unit='Angstrom')
        r_oh1 = np.linalg.norm(coords[1] - coords[0])
        r_oh2 = np.linalg.norm(coords[2] - coords[0])
        self.assertAlmostEqual(r_oh1, r_oh2, delta=1e-3)
        assert 0.8 < r_oh1 < 1.2


if __name__ == "__main__":
    print("Running tests for restricted PM6 geometry optimization...")
    unittest.main()
