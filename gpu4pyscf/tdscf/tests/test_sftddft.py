#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

    @unittest.skipIf(mcfun is None)
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
