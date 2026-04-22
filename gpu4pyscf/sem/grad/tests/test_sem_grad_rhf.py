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
from pyscf.lib import fp
from pyscf.data.nist import BOHR
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

    def test_grad(self):
        mf = hf.RHF(mol)
        mf.conv_tol = 1e-16
        etot = mf.kernel()
        # reference energy from mopac
        # with 1e-16 / 23.060547830619029 tolerance
        e_ref = -319.073057543435
        self.assertAlmostEqual(etot * mol.HARTREE2EV, e_ref, delta=1e-11)
        ref_mo_energy = np.array([-30.35188827, -18.69310543, -14.24656862, -11.8336073 ,
         4.04723741,   5.89311133])
        assert np.abs(mf.mo_energy.get()*27.211386245988 - ref_mo_energy).max() < 1e-7

        g = mf.nuc_grad_method()
        g.kernel()

        g_ref_num = np.array([[-1.77635684e-11, -5.86363543e-03,  0.00000000e+00],
                              [-1.52821806e-03,  2.93181755e-03,  0.00000000e+00],
                              [ 1.52821808e-03,  2.93181755e-03,  0.00000000e+00]])
        # /(23.060547830619029*27.211386245988/0.529177210903) mopac unit is kcal/mol/A
        g_ref_mopac = np.array([[ 0.        , -0.00586363,  0.        ],
                                [-0.00152822, 0.00293182,  0.        ],
                                [ 0.00152822, 0.00293182,  0.        ]])

        assert np.abs(g_ref_num - g.de).max() < 1e-8
        assert np.abs(g_ref_mopac - g.de).max() < 1e-8


if __name__ == "__main__":
    print("Running tests for resctricted PM6 gradient...")
    unittest.main()
