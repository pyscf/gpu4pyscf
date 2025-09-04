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
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""
atom1 = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def setUpModule():
    global mol, mol1
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)
    mol1 = pyscf.M(
        atom=atom1, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1


class KnownValues(unittest.TestCase):
    def test_nac_scanner_ge(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)

        assert (new_nac[1]*nac1.de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac1.de) > 0.99

    def test_nac_scanner_ee(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)

        assert (new_nac[1]*nac1.de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac1.de) > 0.99


if __name__ == "__main__":
    print("Full Tests for TD-RHF nonadiabatic coupling vectors scanner.")
    unittest.main()
