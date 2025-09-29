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

    def test_hf_tda(self):
        mf = self.mol.UHF().to_gpu().run()
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [ 0.46644071, 0.55755649, 1.05310518]
        td = mf.SFTDA().run(extype=0, conv_tol=1e-5)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = [-0.21574567, 0.00270390, 0.03143914]
        td = mf.SFTDA().run(extype=1, conv_tol=1e-5)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(a[1], nroots=3)[0]
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

    def test_mcol_svwn_tda(self):
        mf = self.mol.UKS(xc='svwn').to_gpu().run()
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [0.45022394, 0.57917576, 1.04475443]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 0
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 5)

        ref = [-0.32642984,  0.0003752 ,  0.02156706]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 1
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        e = diagonalize_tda(a[1], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='b3lyp').to_gpu().run()
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [0.45941163, 0.57799537, 1.06629197]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 0
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [-0.29629126,  0.00067001,  0.0195629 ]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 1
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        e = diagonalize_tda(a[1], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

    def test_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='tpss').to_gpu().run()
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [0.4498647 , 0.57071842, 1.0544106 ]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 0
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        a, b = td.get_ab()
        e = diagonalize_tda(a[0], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [-0.28699899,  0.00063662,  0.0232923 ]
        td = mf.SFTDA()
        td.collinear = 'mcol'
        td.extype = 1
        td.collinear_samples=200
        td.conv_tol = 1e-5
        td.kernel()
        e = diagonalize_tda(a[1], nroots=3)[0]

        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

    @unittest.skip('Numerical issues encountered in non-hermitian diagonalization')
    def test_tdhf(self):
        mf = self.mol.UHF().to_gpu().run()
        ref = [1.74385401, 9.38227395, 14.90168875]
        td = mf.SFTDHF().run(extype=0, conv_tol=1e-5)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [0.41701647, 9.59644331, 22.99972711]
        td = mf.SFTDHF().run(extype=1, conv_tol=1e-5)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA and spin-flip-TDDFT using multi-collinear functionals")
    unittest.main()
