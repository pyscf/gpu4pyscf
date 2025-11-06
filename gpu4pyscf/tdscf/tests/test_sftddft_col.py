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
from pyscf import lib, gto, scf
from gpu4pyscf import tdscf
try:
    import mcfun
except ImportError:
    mcfun = None


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e, xy = np.linalg.eig(np.asarray(a))
    sorted_indices = np.argsort(e)

    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]

    return e_sorted[:nroots], xy_sorted[:, :nroots]


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_lda_tda(self):
        mf = self.mol.UKS(xc='svwn').to_gpu().run()
        na, nb = mf.mol.nelec

        td = mf.SFTDA()
        td.extype = 0
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[0][na] - mf.mo_energy[1][nb-1]) < 1e-6

        td = mf.SFTDA()
        td.extype = 1
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        e = diagonalize_tda(a[1], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[1][nb] - mf.mo_energy[0][na-1]) < 1e-6

    def test_b3lyp_tda(self):
        mf = self.mol.UKS(xc='b3lyp').to_gpu().run()
        na, nb = mf.mol.nelec

        td = mf.SFTDA()
        td.extype = 0
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[0][na] - mf.mo_energy[1][nb-1]) < 1e-6

        td = mf.SFTDA()
        td.extype = 1
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        e = diagonalize_tda(a[1], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[1][nb] - mf.mo_energy[0][na-1]) < 1e-6

    def test_tpss_tda(self):
        mf = self.mol.UKS(xc='tpss').to_gpu().run()
        na, nb = mf.mol.nelec

        td = mf.SFTDA()
        td.extype = 0
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[0][na] - mf.mo_energy[1][nb-1]) < 1e-6

        td = mf.SFTDA()
        td.extype = 1
        td.conv_tol = 1e-5
        td.nroots = 3
        td.collinear = 'col'
        td.run()
        e = diagonalize_tda(a[1], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        assert td.e[0] - (mf.mo_energy[1][nb] - mf.mo_energy[0][na-1]) < 1e-6


if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA using collinear functional.")
    unittest.main()
