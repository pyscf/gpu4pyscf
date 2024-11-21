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
import tempfile
import numpy as np
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import dft as pbcdft


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global cell
        L = 4
        n = 21
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
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
        global cell
        del cell

    def test_lda_fft(self):
        mf = pbcdft.RKS(cell, xc='lda,vwn').run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_gga_fft(self):
        mf = pbcdft.RKS(cell, xc='pbe0').run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_rsh_fft(self):
        mf = pbcdft.RKS(cell, xc='camb3lyp').run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_lda_fft_with_kpt(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pbcdft.RKS(cell, xc='lda,vwn', kpt=k).run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_gga_fft_with_kpt(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pbcdft.RKS(cell, xc='pbe0', kpt=k).run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_rsh_fft_with_kpt(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pbcdft.RKS(cell, xc='camb3lyp', kpt=k).run(conv_tol=1e-8)
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
