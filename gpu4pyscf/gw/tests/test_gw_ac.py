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
import numpy as np
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.gw import gw_ac


def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_gwac_diag(self):
        gw = gw_ac.GWAC(mf)
        gw.orbs = range(4, 6)
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)

    def test_gwac_full(self):
        gw = gw_ac.GWAC(mf)
        gw.fullsigma = True
        gw.orbs = range(4, 6)
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)

    def test_gwac_diag_outcore(self):
        gw = gw_ac.GWAC(mf)
        gw.orbs = range(4, 6)
        gw.outcore = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)

    def test_gwac_full_outcore(self):
        gw = gw_ac.GWAC(mf)
        gw.orbs = range(4, 6)
        gw.fullsigma = True
        gw.outcore = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)

    def test_gwac_frozen_core(self):
        gw = gw_ac.GWAC(mf)
        gw.orbs = range(4, 6)
        gw.frozen = 1
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42667346, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16490656, delta=1e-5)

    def test_gwac_frozen_list(self):
        gw = gw_ac.GWAC(mf)
        gw.orbs = [4, 7]
        gw.frozen = [0, 5]
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.43309464, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[7], 0.73675504, delta=1e-5)

    def test_gwac_rdm(self):
        gw = gw_ac.GWAC(mf)
        gw.fullsigma = True
        gw.rdm = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)
        rdm1 = gw.make_rdm1()
        self.assertEqual(rdm1.shape, (gw.nmo, gw.nmo))
        self.assertAlmostEqual(np.trace(rdm1), 9.996786358429022, delta=1e-6)

    def test_gwac_gf(self):
        gw = gw_ac.GWAC(mf)
        gw.fullsigma = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[4], -0.42657296, delta=1e-5)
        self.assertAlmostEqual(gw.mo_energy[5], 0.16495549, delta=1e-5)
        omega = np.linspace(-0.5, 0.5, 101)
        gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01, mode='dyson')
        self.assertEqual(gf.shape, (gw.nmo, gw.nmo, len(omega)))
        self.assertTrue(np.isfinite(gf).all())


if __name__ == "__main__":
    print("Full Tests for GPU GW-AC")
    unittest.main()
