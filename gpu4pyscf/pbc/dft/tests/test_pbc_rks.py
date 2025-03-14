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
        mf = pbcdft.RKS(cell, xc='pbe0', kpt=k).run(conv_tol=1e-10)
        mf_ref = mf.to_cpu().run(conv_tol=1e-10)
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
        mf = pbcdft.RKS(cell, xc='camb3lyp', kpt=k).run(conv_tol=1e-10)
        mf_ref = mf.to_cpu().run(conv_tol=1e-10)
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_kpts_lda_fft(self):
        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='lda,vwn', kpts=kpts).run(conv_tol=1e-10)
        mf_ref = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = kmf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_kpts_gga_fft(self):
        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='pbe0', kpts=kpts).run(conv_tol=1e-10)
        mf_ref = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

    def test_kpts_rsh_fft(self):
        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='camb3lyp', kpts=kpts).run(conv_tol=1e-9)
        mf_ref = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

    def test_kpts_gga_gdf(self):
        from gpu4pyscf.pbc.df.df import GDF
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.build()

        mf = cell.RKS(xc='pbe0').to_gpu().density_fit().run()
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(mf.e_tot, -0.44834992009430463, 7)
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='pbe0', kpts=kpts).density_fit().run()
        self.assertTrue(isinstance(kmf.with_df, GDF))
        self.assertAlmostEqual(kmf.e_tot, -0.44429306, 7)
        mf_ref = kmf.to_cpu()
        mf_ref.run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
