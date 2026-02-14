# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf
import pytest
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet(self):
        """
        density fitting only
        """
        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        print(nac1.de)
        print(nac_test.results[(1,2)]['de'])
        print(abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max())
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-7

        nac1 = td.nac_method()
        nac1.states=(1,3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1,3)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-7

        nac1 = td.nac_method()
        nac1.states=(2,3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(2,3)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(2,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-7



if __name__ == "__main__":
    print("Full Tests for batched TD-RHF nonadiabatic coupling vectors between excited states.")
    unittest.main()
