# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.gw import rpa


def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_rpa(self):
        myrpa = rpa.RPA(mf)
        myrpa.kernel()
        self.assertAlmostEqual(myrpa.e_corr, -0.30783004035780076, delta=1e-6)
        self.assertAlmostEqual(myrpa.e_tot, -76.26428191794182, delta=1e-6)

    def test_rpa_outcore(self):
        myrpa = rpa.RPA(mf)
        myrpa.outcore = True
        myrpa.segsize = 2
        myrpa.kernel()
        self.assertAlmostEqual(myrpa.e_corr, -0.30783004035780076, delta=1e-6)
        self.assertAlmostEqual(myrpa.e_tot, -76.26428191794182, delta=1e-6)

    def test_rpa_frozen(self):
        myrpa = rpa.RPA(mf)
        myrpa.frozen = 1
        myrpa.kernel()
        self.assertAlmostEqual(myrpa.e_corr, -0.3046092046689639, delta=1e-6)
        self.assertAlmostEqual(myrpa.e_tot, -76.26106101984152, delta=1e-6)

    def test_rpa_frozen_outcore(self):
        myrpa = rpa.RPA(mf)
        myrpa.frozen = 1
        myrpa.outcore = True
        myrpa.segsize = 2
        myrpa.kernel()
        self.assertAlmostEqual(myrpa.e_corr, -0.3046092046689639, delta=1e-6)
        self.assertAlmostEqual(myrpa.e_tot, -76.26106101984152, delta=1e-6)


if __name__ == "__main__":
    print("Full Tests for GPU RPA")
    unittest.main()
