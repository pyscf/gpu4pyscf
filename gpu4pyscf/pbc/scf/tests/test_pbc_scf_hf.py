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
from pyscf import lib
from pyscf.pbc.scf import hf as pbchf_cpu
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = 4
        n = 21
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   verbose = 7,
                   output = '/dev/null',
                   a = ((L,0,0),(0,L,0),(0,0,L)),
                   mesh = [n,n,n],
                   atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                           ['He', (L/2.   ,L/2.,L/2.+.5)]],
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    [0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]]})
        cls.cell = cell

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()

    def test_rhf_exx_ewald(self):
        cell = self.cell
        mf = scf.RHF(cell, exxdiv='ewald').run()
        self.assertAlmostEqual(mf.e_tot, -4.3511582284698633, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.double)
        #kmf = scf.KRHF(cell, [[0,0,0]], exxdiv='ewald').run()
        #self.assertAlmostEqual(mf.e_tot, kmf.e_tot, 8)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e1, c1 = mf.get_bands(kpts_band)
        #e0, c0 = kmf.get_bands(kpts_band)
        #self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
        #self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1[0].get()), -6.2986775452228283, 6)
        self.assertAlmostEqual(lib.fp(e1[1].get()), -7.6616273746782362, 6)

    def test_rhf_exx_ewald_with_kpt(self):
        np.random.seed(1)
        k = np.random.random(3)
        cell = self.cell
        mf = scf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.complex128)

        #kmf = scf.KRHF(cell, k, exxdiv='ewald')
        #e0 = kmf.kernel()
        #self.assertTrue(np.allclose(e0,e1))

        # test bands
        np.random.seed(1)
        kpt_band = np.random.random(3)
        e1, c1 = mf.get_bands(kpt_band)
        #e0, c0 = kmf.get_bands(kpt_band)
        #self.assertAlmostEqual(abs(e0-e1).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1.get()), -6.8312867098806249, 6)

    def test_rhf_exx_None(self):
        cell = self.cell
        mf = scf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.double)

        #mf = scf.KRHF(cell, [[0,0,0]], exxdiv=None)
        #e0 = mf.kernel()
        #self.assertTrue(np.allclose(e0,e1))

        np.random.seed(1)
        k = np.random.random(3)
        mf = scf.RHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.complex128)

        #mf = scf.KRHF(cell, k, exxdiv=None)
        #mf.init_guess = 'hcore'
        #e0 = mf.kernel()
        #self.assertTrue(np.allclose(e0,e1))

    def test_jk(self):
        cell = self.cell
        nao = cell.nao
        np.random.seed(2)
        dm = np.random.random((2,nao,nao)) + .5j*np.random.random((2,nao,nao))
        dm = dm + dm.conj().transpose(0,2,1)
        ref = pbchf_cpu.RHF(cell).get_jk(cell, dm)

        dm = cp.asarray(dm)
        vj, vk = scf.RHF(cell).get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vk.get() - ref[1]).max(), 0, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
