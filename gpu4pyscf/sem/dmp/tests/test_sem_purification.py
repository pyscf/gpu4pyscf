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
from gpu4pyscf.sem.integral.fock import (get_hcore)
from gpu4pyscf.sem.integral import eri_2c2e
from gpu4pyscf.sem.gto.params import load_sem_params
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

    def test_scf(self):
        mf = hf.RHF(mol).purification()
        mf.conv_tol = 1e-16
        etot = mf.kernel()
        # reference energy from mopac
        # with 1e-16 / 23.060547830619029 tolerance
        e_ref = -319.073057543435
        self.assertAlmostEqual(etot * mol.HARTREE2EV, e_ref, delta=1e-11)


if __name__ == "__main__":
    print("Running tests for resctricted PM6 SCF with sp2 purification...")
    unittest.main()
