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

    def test_init_guess(self):
        mf = hf.RHF(mol)
        dm = mf.init_guess_by_1e()
        ref_dm = cp.array([[ 1.915763772229531e+00,  2.340088389234758e-16,
            -1.318436988772523e-01,  0.000000000000000e+00,
            2.683225228510512e-01,  2.683225228510510e-01],
            [ 2.340088389234758e-16,  1.745216742172839e+00,
            7.573541492284053e-16,  1.262555451165763e-15,
            4.715145847083093e-01, -4.715145847083110e-01],
            [-1.318436988772523e-01,  7.573541492284053e-16,
            1.793642695147734e+00, -1.311509821171923e-30,
            4.199693509680212e-01,  4.199693509680223e-01],
            [ 0.000000000000000e+00,  1.262555451165763e-15,
            -1.311509821171923e-30,  2.000000000000000e+00,
            -1.768742454744762e-15,  1.768742454744762e-15],
            [ 2.683225228510512e-01,  4.715145847083093e-01,
            4.199693509680212e-01, -1.768742454744762e-15,
            2.726883952249469e-01,  1.790513739778678e-02],
            [ 2.683225228510510e-01, -4.715145847083110e-01,
            4.199693509680223e-01,  1.768742454744762e-15,
            1.790513739778678e-02,  2.726883952249486e-01]])
        assert cp.abs(dm - ref_dm).max() < 1e-12

    def test_scf(self):
        mf = hf.RHF(mol)
        etot = mf.kernel()
        # reference energy from mopac
        # with 1e-12 / 23.060547830619029 tolerance
        e_ref = -319.07305754343497
        self.assertAlmostEqual(etot * mol.HARTREE2EV, e_ref, delta=1e-12)


if __name__ == "__main__":
    print("Running tests for resctricted PM6 SCF...")
    unittest.main()
