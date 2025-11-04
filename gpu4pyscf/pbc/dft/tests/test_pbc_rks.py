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
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

def setUpModule():
    global cell
    L = 4.
    cell = pbcgto.Cell()
    cell.a = np.eye(3)*L
    cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
    cell.basis = [[0, (3.0, 1.0)], [0, (1.0, 1.0)]]
    cell.verbose = 6
    cell.output = '/dev/null'
    cell.build()

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

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
        ij, diag = with_df._cderi_idx
        nao = cell.nao
        i, j = divmod(ij, nao)
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
        k = np.random.random((1, 3))
        mf = pbcdft.KRKS(cell, xc='lda,vwn', kpts=k).run()
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
        k = np.random.random((1, 3))
        mf = pbcdft.KRKS(cell, xc='pbe0', kpts=k).run(conv_tol=1e-10)
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
        k = np.random.random((1, 3))
        mf = pbcdft.KRKS(cell, xc='camb3lyp', kpts=k).run(conv_tol=1e-10)
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
        #mf_ref = mf.to_cpu().run()
        #self.assertAlmostEqual(mf.e_tot, mf_ref.e_tot, 7)

        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kmf = pbcdft.KRKS(cell, xc='pbe0', kpts=kpts).density_fit().run()
        self.assertTrue(isinstance(kmf.with_df, GDF))
        self.assertAlmostEqual(kmf.e_tot, -0.44429306, 6)
        #mf_ref = kmf.to_cpu()
        #mf_ref.run()
        #self.assertAlmostEqual(kmf.e_tot, mf_ref.e_tot, 7)

    def test_reset(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 7
        cell.output = '/dev/null'
        cell.build()
        np.random.seed(1)
        kpt0 = np.random.rand(3)
        mf = cell.RKS(kpt=kpt0).to_gpu()

        cell1 = pbcgto.Cell()
        cell1.atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95'
        cell1.a = '''0.   1.9  1.9
                     1.9  0.   1.9
                     1.9  1.9  0.    '''
        cell1.basis = 'gth-dzvp'
        cell1.pseudo = 'gth-pade'
        cell1.verbose = 7
        cell1.output = '/dev/null'
        cell1.build()
        mf.reset(cell1)
        assert abs(mf.kpt - kpt0).sum() > 0.01

    def test_lda_rsjk(self):
        mf = cell.RKS().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.40889872799664, 8)
        #ref = cell.RKS().run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_pbe0_rsjk(self):
        mf = cell.RKS(xc='pbe0').to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.453945026971869, 8)
        #ref = cell.RKS(xc='pbe0').run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_wb97_rsjk(self):
        mf = cell.RKS(xc='wb97', exxdiv=None).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.145749552940759, 8)
        #ref = cell.RKS(xc='wb97', exxdiv=None).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.RKS(xc='wb97').to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.475660452630126, 8)
        #ref = cell.RKS(xc='wb97').run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_hse06_rsjk(self):
        mf = cell.RKS(xc='hse06', exxdiv=None).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.390562199148231, 8)
        #ref = cell.RKS(xc='hse06', exxdiv=None).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.RKS(xc='hse06').to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.453371384843629, 8)
        #ref = cell.RKS(xc='hse06').run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_camb3lyp_rsjk(self):
        mf = cell.RKS(xc='camb3lyp', exxdiv=None).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.228873263786209, 8)
        #ref = cell.RKS(xc='camb3lyp', exxdiv=None).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.RKS(xc='camb3lyp').to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.442283740471709, 8)
        #ref = cell.RKS(xc='camb3lyp').run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_lda_krks_rsjk(self):
        kpts = cell.make_kpts([2,1,1])
        mf = cell.KRKS(kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.408895805671884, 8)
        #ref = cell.KRKS(kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_pbe0_krks_rsjk(self):
        kpts = cell.make_kpts([2,1,1])
        mf = cell.KRKS(xc='pbe0', kpts=kpts).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.449887356407533, 8)
        #ref = cell.KRKS(xc='pbe0', kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_wb97_krks_rsjk(self):
        kpts = cell.make_kpts([2,1,1])
        mf = cell.KRKS(xc='wb97', exxdiv=None, kpts=kpts).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.243471468502624, 8)
        #ref = cell.KRKS(xc='wb97', exxdiv=None, kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.KRKS(xc='wb97', kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.459652873123233, 8)
        #ref = cell.KRKS(xc='wb97', kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_hse06_krks_rsjk(self):
        kpts = cell.make_kpts([2,1,1])
        mf = cell.KRKS(xc='hse06', exxdiv=None, kpts=kpts).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.419625683338891, 8)
        #ref = cell.KRKS(xc='hse06', exxdiv=None, kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.KRKS(xc='hse06', kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.449507864648219, 8)
        #ref = cell.KRKS(xc='hse06', kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_cambl3yp_krks_rsjk(self):
        kpts = cell.make_kpts([2,1,1])
        mf = cell.KRKS(xc='camb3lyp', exxdiv=None, kpts=kpts).to_gpu()
        mf._numint = MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.292124263676038, 8)
        #ref = cell.KRKS(xc='camb3lyp', exxdiv=None, kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        mf = cell.KRKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.432150196659050, 8)
        #ref = cell.KRKS(xc='camb3lyp', kpts=kpts).run()
        #self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.rks")
    unittest.main()
