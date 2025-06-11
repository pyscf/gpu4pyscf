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
import cupy as cp
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.dft import UniformGrids
from pyscf.lib import unpack_tril
from gpu4pyscf.pbc import dft as pbcdft


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_lda_fft(self):
        cell = self.cell
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
        cell = self.cell
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
        cell = self.cell
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

    def test_kpts_mgga(self):
        cell = self.cell
        mf = pbcdft.RKS(cell, xc='tpss').run()
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e0, c0 = mf_ref.get_bands(kpts_band)
        e1, c1 = mf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e1[0].get() - e0[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e1[1].get() - e0[1]).max(), 0, 7)

    def test_lda_gdf(self):
        from pyscf.pbc.df.df import _load3c
        cell = self.cell
        xc = 'svwn'
        mf = pbcdft.RKS(cell, xc=xc).density_fit().run()
        pcell = cell.copy()
        pcell.precision = 1e-10
        mf_ref = pcell.RKS(xc=xc).density_fit()
        # Becke grids were used in PySCF, while the GPU4PySCF takes uniform grids
        mf_ref.grids = UniformGrids(pcell)
        mf_ref.run()
        assert abs(mf.e_tot - mf_ref.e_tot) < 5e-7

        with_df = mf.with_df
        auxcell = with_df.auxcell
        i, j, diag = with_df._cderi_idx
        nao = cell.nao
        naux = auxcell.nao
        out = cp.zeros((naux,nao,nao))
        out[:,j,i] = out[:,i,j] = with_df._cderi[0]
        with _load3c(mf_ref.with_df._cderi, 'j3c', np.zeros((2,3))) as cderi:
            ref = unpack_tril(cderi[:])
        assert abs(out.get() - ref).max() < 1e-8

    def test_gga_gdf(self):
        cell = self.cell
        xc = 'pbe0'
        mf = pbcdft.RKS(cell, xc=xc).density_fit().run()
        pcell = cell.copy()
        pcell.precision = 1e-10
        mf_ref = pcell.RKS(xc=xc).density_fit()
        # Becke grids were used in PySCF, while the GPU4PySCF takes uniform grids
        mf_ref.grids = UniformGrids(pcell)
        mf_ref.run()
        assert abs(mf.e_tot - mf_ref.e_tot) < 5e-7

    def test_rsh_gdf(self):
        cell = self.cell
        xc = 'camb3lyp'
        mf = pbcdft.RKS(cell, xc=xc).density_fit().run()
        pcell = cell.copy()
        pcell.precision = 1e-10
        mf_ref = pcell.RKS(xc=xc).density_fit()
        # Becke grids were used in PySCF, while the GPU4PySCF takes uniform grids
        mf_ref.grids = UniformGrids(pcell)
        mf_ref.run()
        assert abs(mf.e_tot - mf_ref.e_tot) < 5e-7

    def test_lda_fft_with_kpt(self):
        cell = self.cell
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
        cell = self.cell
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
        cell = self.cell
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
        cell = self.cell
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
        cell = self.cell
        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='pbe0', kpts=kpts).run(conv_tol=1e-10)
        mf_ref = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

    def test_kpts_rsh_fft(self):
        cell = self.cell
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
        self.assertAlmostEqual(mf.e_tot, -0.4483496502, 7)
        mf_ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='pbe0', kpts=kpts).density_fit().run()
        self.assertTrue(isinstance(kmf.with_df, GDF))
        self.assertAlmostEqual(kmf.e_tot, -0.44429306, 6)
        mf_ref = kmf.to_cpu()
        mf_ref.run()
        self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
