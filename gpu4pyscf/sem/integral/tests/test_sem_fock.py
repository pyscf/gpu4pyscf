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
from pyscf.data.nist import BOHR
from gpu4pyscf.sem.integral.fock import (get_hcore)
from gpu4pyscf.sem.integral import eri_2c2e
from gpu4pyscf.sem.gto.params import load_sem_params
from gpu4pyscf.sem.gto.mole import Mole

class KnownValues(unittest.TestCase):
    def test_multipole_eval(self):
        mol = Mole('O 0.0000 0.0000 0.0000; H 0.7570 0.5860 0.0000; H -0.7570 0.5860 0.0000')
        mol.build()

        hcore = get_hcore(mol)
        hcore_ref = np.array([[-111.07239162768941 ,    0.               ,   -1.359178647940259,
                0.               ,   -5.876475600590749,   -5.876475600590749],
            [   0.               ,  -90.03044144464374 ,    0.               ,
                0.               ,   -4.636920222083057,    4.636920222083057],
            [  -1.359178647940259,    0.               ,  -89.50428424109921 ,
                0.               ,   -3.589478533871429,   -3.589478533871429],
            [   0.               ,    0.               ,    0.               ,
                -88.71753083892997 ,    0.               ,    0.               ],
            [  -5.876475600590749,   -4.636920222083057,   -3.589478533871429,
                0.               ,  -77.37216383317403 ,   -1.998790032593725],
            [  -5.876475600590749,    4.636920222083057,   -3.589478533871429,
            0.               ,   -1.998790032593725,  -77.37216383317403 ]])

        assert np.abs(hcore.get() - hcore_ref).max() < 1e-12

    def test_multipole_eval2(self):
        mol = Mole('S 0.0000 0.0000 0.0000; Cl 0.7570 0.5860 0.0000; Cl -0.7570 0.5860 0.0000')
        mol.build()

        hcore = get_hcore(mol)
        hcore_sum_ref = -4517.609394917911

        assert np.abs(hcore.get().sum() - hcore_sum_ref).max() < 1e-11


if __name__ == "__main__":
    print("Running tests for fock...")
    unittest.main()
