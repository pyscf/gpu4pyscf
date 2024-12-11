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
        cls.mf = mol.UHF().to_gpu().run()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_tda(self):
        mf = self.mf
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [ 0.46644071, 0.55755649, 1.05310518]
        td = mf.SFTDA().run(extype=0, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [-0.21574567, 0.00270390, 0.03143914]
        td = mf.SFTDA().run(extype=1, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

    @unittest.skipIf(mcfun is None, 'MCfun not available')
    def test_mcol_b3lyp_tda(self):
        mf = self.mf
        # sftddft not available in pyscf main branch. References are created
        # using the sftda module from pyscf-forge
        ref = [ 0.45941171, 0.57799552, 1.06629265]
        td = mf.SFTDA().run(collinear='mcol', extype=0, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [-0.29629139, 0.00067017, 0.01956306]
        td = mf.SFTDA().run(collinear='mcol', extype=1, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

    @unittest.skip('Numerical issues encountered in non-hermitian diagonalization')
    def test_tdhf(self):
        mf = self.mf
        ref = [1.74385401, 9.38227395, 14.90168875]
        td = mf.SFTDHF().run(extype=0, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

        ref = [0.41701647, 9.59644331, 22.99972711]
        td = mf.SFTDHF().run(extype=1, conv_tol=1e-7)
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA and spin-flip-TDDFT")
    unittest.main()
