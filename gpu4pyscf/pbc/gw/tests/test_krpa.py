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
from packaging.version import Version
import pyscf
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft, scf

# gpu4pyscf.pbc.gw uses the pyscf.gw and pyscf.pbc.gw helpers of pyscf >= 2.14
if Version(pyscf.__version__) < Version('2.14'):
    raise unittest.SkipTest('gpu4pyscf.pbc.gw requires pyscf >= 2.14')

from gpu4pyscf.pbc.gw import krpa


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
        pseudo='gth-pbe',
        basis='gth-dzv',
        precision=1e-12,
    )
    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    kmf = scf.KRHF(cell, kpts).density_fit()
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
    def test_krpa(self):
        rpa = krpa.KRPA(kmf)
        rpa.fc = False
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.185274629765, delta=1e-6)
        self.assertAlmostEqual(rpa.e_tot, -10.694342003554, delta=1e-6)

    def test_krpa_outcore(self):
        rpa = krpa.KRPA(kmf)
        rpa.outcore = True
        rpa.segsize = 2
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.185274629765, delta=1e-6)
        self.assertAlmostEqual(rpa.e_tot, -10.694342003554, delta=1e-6)

    def test_krpa_fc(self):
        rpa = krpa.KRPA(kmf)
        rpa.fc = True
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.207229608059, delta=1e-6)
        self.assertAlmostEqual(rpa.e_tot, -10.716296981848, delta=1e-6)

    def test_krpa_fc_outcore(self):
        rpa = krpa.KRPA(kmf)
        rpa.fc = True
        rpa.outcore = True
        rpa.segsize = 2
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.207229608059, delta=1e-6)
        self.assertAlmostEqual(rpa.e_tot, -10.716296981848, delta=1e-6)

    def test_krpa_metal(self):
        rpa = krpa.KRPA(kmf_metal)
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.04031792880477294, delta=1e-5)
        self.assertAlmostEqual(rpa.e_tot, -47.60693294927457, delta=1e-5)

    def test_krpa_metal_outcore(self):
        rpa = krpa.KRPA(kmf_metal)
        rpa.outcore = True
        rpa.segsize = 3
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.04031792880477294, delta=1e-5)
        self.assertAlmostEqual(rpa.e_tot, -47.60693294927457, delta=1e-5)

    def test_krpa_metal_acfd_exx(self):
        rpa = krpa.KRPA(kmf_metal)
        rpa.acfd_exx = True
        rpa.kernel()
        self.assertAlmostEqual(rpa.e_corr, -0.04031792880477294, delta=1e-5)
        self.assertAlmostEqual(rpa.e_tot, -47.660250115879, delta=1e-5)


if __name__ == "__main__":
    print("Full Tests for GPU periodic RPA")
    unittest.main()
