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
from packaging.version import Version
import pyscf
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft

# gpu4pyscf.pbc.gw uses the pyscf.gw and pyscf.pbc.gw helpers of pyscf >= 2.14
if Version(pyscf.__version__) < Version('2.14'):
    raise unittest.SkipTest('gpu4pyscf.pbc.gw requires pyscf >= 2.14')

from gpu4pyscf.pbc.gw import krgw_ac


def setUpModule():
    global cell, kmf, cell_metal, kmf_metal
    # diamond
    cell = gto.Cell()
    cell.build(
        unit='angstrom',
        a='''
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        ''',
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        dimension=3,
        verbose=1,
        output='/dev/null',
        pseudo='gth-pade',
        basis='gth-szv',
        precision=1e-10,
    )
    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    kmf = dft.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf.conv_tol = 1e-12
    kmf.kernel()

    # Na (metallic)
    cell_metal = gto.Cell()
    cell_metal.build(
        unit='angstrom',
        a='''
            -2.11250000000000   2.11250000000000   2.11250000000000
            2.11250000000000  -2.11250000000000   2.11250000000000
            2.11250000000000   2.11250000000000  -2.11250000000000
        ''',
        atom='Na 0.00000 0.00000 0.00000',
        dimension=3,
        verbose=1,
        output='/dev/null',
        pseudo='gth-pade',
        basis='gth-dzvp-molopt-sr',
        precision=1e-10,
    )
    kpts = cell_metal.make_kpts([2, 2, 1], scaled_center=[0, 0, 0])
    kmf_metal = dft.KRKS(cell_metal, kpts).density_fit()
    kmf_metal = kmf_metal.smearing(sigma=5e-3, method='fermi')
    kmf_metal.xc = 'lda'
    kmf_metal.conv_tol = 1e-12
    kmf_metal.kernel()


def tearDownModule():
    global cell, kmf, cell_metal, kmf_metal
    cell.stdout.close()
    cell_metal.stdout.close()
    del cell, kmf, cell_metal, kmf_metal


class KnownValues(unittest.TestCase):
    def test_krgwac(self):
        gw = krgw_ac.KRGWAC(kmf)
        gw.fc = False
        gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
        self.assertAlmostEqual(gw.mo_energy[0][3], 0.62047817, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.96570185, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][3], 0.52637737, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][4], 1.0751805, delta=1e-4)

    def test_krgwac_outcore(self):
        gw = krgw_ac.KRGWAC(kmf)
        gw.fc = False
        gw.outcore = True
        gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
        self.assertAlmostEqual(gw.mo_energy[0][3], 0.62047817, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.96570185, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][3], 0.52637737, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][4], 1.0751805, delta=1e-4)

    def test_krgwac_fc(self):
        gw = krgw_ac.KRGWAC(kmf)
        gw.fc = True
        gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))
        self.assertAlmostEqual(gw.mo_energy[0][3], 0.44029997, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.80144906, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][3], 0.35195222, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[1][4], 0.92909667, delta=1e-4)

    def test_krgwac_fc_frozen_core(self):
        gw = krgw_ac.KRGWAC(kmf)
        gw.fc = True
        gw.frozen = 1
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[0][3], 0.44098086, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.79816375, delta=1e-4)

    def test_krgwac_fc_full_rdm_gf(self):
        gw = krgw_ac.KRGWAC(kmf)
        gw.eta = 1e-2
        gw.fullsigma = True
        gw.fc = True
        gw.rdm = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[0][3], 0.44029997, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.80144906, delta=1e-4)
        rdm1 = gw.make_rdm1()
        self.assertEqual(rdm1.shape, (gw.nkpts, gw.nmo, gw.nmo))
        nelec = sum(np.trace(dm) for dm in rdm1) / gw.nkpts
        self.assertAlmostEqual(nelec, 7.999947846930319, delta=1e-6)
        # AO representation is C dm C^dagger, so Tr(dm_ao S) gives the same particle number
        rdm1_ao = gw.make_rdm1(ao_repr=True)
        mo_coeff = kmf.mo_coeff.get()
        np.testing.assert_allclose(rdm1_ao, mo_coeff @ rdm1 @ mo_coeff.conj().transpose(0, 2, 1), atol=1e-10)
        nelec = np.einsum('kij,kji->', rdm1_ao, kmf.get_ovlp().get()).real / gw.nkpts
        self.assertAlmostEqual(nelec, 7.999947846930319, delta=1e-6)
        omega = np.linspace(0.2, 1.2, 101)
        gf, gf0, _ = gw.make_gf(omega, gw.eta)
        self.assertEqual(gf.shape, (gw.nkpts, gw.nmo, gw.nmo, len(omega)))
        self.assertTrue(np.isfinite(gf).all())

    def test_krgwac_metal(self):
        gw = krgw_ac.KRGWAC(kmf_metal)
        gw.kernel(kptlist=[0, 1, 2], orbs=range(1, 7))
        self.assertAlmostEqual(gw.mo_energy[0][3], -0.94493882, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.04603593, delta=1e-4)

    def test_krgwac_metal_outcore(self):
        gw = krgw_ac.KRGWAC(kmf_metal)
        gw.outcore = True
        gw.kernel(kptlist=[0, 1, 2], orbs=range(1, 7))
        self.assertAlmostEqual(gw.mo_energy[0][3], -0.94493882, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.04603593, delta=1e-4)

    def test_krgwac_metal_frozen_core(self):
        gw = krgw_ac.KRGWAC(kmf_metal)
        gw.frozen = 1
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[0][3], -0.94740522, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.04504422, delta=1e-4)

    def test_krgwac_metal_full_gf(self):
        gw = krgw_ac.KRGWAC(kmf_metal)
        gw.eta = 1e-2
        gw.fullsigma = True
        gw.kernel()
        self.assertAlmostEqual(gw.mo_energy[0][3], -0.94493882, delta=1e-4)
        self.assertAlmostEqual(gw.mo_energy[0][4], 0.04603593, delta=1e-4)
        omega = np.linspace(-0.5, 0.5, 101)
        gf, gf0, _ = gw.make_gf(omega, eta=0.01)
        self.assertEqual(gf.shape, (gw.nkpts, gw.nmo, gw.nmo, len(omega)))
        self.assertTrue(np.isfinite(gf).all())


if __name__ == "__main__":
    print("Full Tests for GPU periodic GW-AC")
    unittest.main()
