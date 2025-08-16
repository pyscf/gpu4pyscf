#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import multigrid as multigrid_cpu
if hasattr(multigrid_cpu, 'MultiGridNumInt'):
    MultiGridNumInt_cpu = multigrid_cpu.MultiGridNumInt
else:
    MultiGridNumInt_cpu = multigrid_cpu.MultiGridFFTDF
from gpu4pyscf.pbc.dft import multigrid_v2 as multigrid
from gpu4pyscf.pbc.tools import ifft, fft

def setUpModule():
    global cell_orth, cell_nonorth
    global kpts, dm, dm1
    np.random.seed(2)
    cell_orth = gto.M(
        verbose = 7,
        output = '/dev/null',
        a = np.diag([3.6, 3.2, 4.5]),
        atom = '''C     0.      0.      0.
                  C     1.8     1.8     1.8   ''',
        basis = ('gth-dzv', [[3, [2., 1.]], [4, [1., 1.]]]),
        pseudo = 'gth-pade',
        precision = 1e-9,
    )

    kptsa = np.random.random((2,3))
    kpts = kptsa.copy()
    kpts[1] = -kpts[0]
    nao = cell_orth.nao_nr()
    dm = np.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + np.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)

    cell_nonorth = pyscf.M(
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]],
        a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000''',
        basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
        pseudo = 'gth-pade',
        unit = 'bohr',
        mesh = [18] * 3) # GGA needs dense mesh for derivative in reciprocal space

def tearDownModule():
    global cell_orth, cell_nonorth
    del cell_orth, cell_nonorth

class KnownValues(unittest.TestCase):
    def test_get_pp(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_pp()
        out = multigrid.MultiGridNumInt(cell_orth).get_pp().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc_nonorth(self):
        ref = MultiGridNumInt_cpu(cell_nonorth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_nonorth).get_nuc().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc_kpts(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    @unittest.skip('kpts not supported')
    def test_get_nuc_kpts_nonorth(self):
        ref = MultiGridNumInt_cpu(cell_nonorth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_nonorth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_rho(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        ref = MultiGridNumInt_cpu(cell_orth).get_rho(dm)
        out = multigrid.MultiGridNumInt(cell_orth).get_rho(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_j(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        ref = MultiGridNumInt_cpu(cell_orth).get_jk(dm[None], with_k=False)[0]
        out = multigrid.MultiGridNumInt(cell_orth).get_j(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_j_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        ref = MultiGridNumInt_cpu(cell_nonorth).get_jk(dm[None], with_k=False)[0]
        out = multigrid.MultiGridNumInt(cell_nonorth).get_j(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_vxc_lda(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

        xc = 'lda,'
        dm = np.array([dm, dm])
        mf = pcell.RKS(xc=xc)
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    @unittest.skip('kpts not supported')
    def test_get_vxc_lda_kpts(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        nkpts = len(kpts)
        dm = np.random.random((nkpts,nao,nao)) - .5
        dm = dm + dm.transpose(0,2,1)
        dm = np.array([dm+1e-3, dm])
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        if hasattr(multigrid_cpu, 'nr_uks'):
            n0, exc0, ref = multigrid_cpu.nr_uks(
                MultiGridNumInt_cpu(pcell), xc, dm, with_j=True, kpts=kpts)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_uks(
                pcell, None, xc, dm, kpts=kpts)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(
            cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    def test_get_vxc_gga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

        xc = 'pbe,'
        dm = np.array([dm, dm])
        mf = pcell.RKS(xc=xc)
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    def test_get_vxc_gga_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(cell_nonorth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    @unittest.skip('kpts not supported')
    def test_get_vxc_gga_kpts(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'pbe,'
        nkpts = len(kpts)
        dm = np.random.random((nkpts,nao,nao)) - .5
        dm = dm + dm.transpose(0,2,1)
        dm = np.array([dm+1e-3, dm])
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        mf = pcell.KRKS(xc=xc)
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(
            cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    @unittest.skip('kpts not supported')
    def test_get_vxc_gga_kpts_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(
                MultiGridNumInt_cpu(pcell), xc, dm, with_j=True, kpts=kpts)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(
                pcell, None, xc, dm, kpts=kpts)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(
            cell_nonorth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    def test_get_vxc_mgga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'r2scan'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        mf = pcell.RKS(xc=xc)

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc.get()).max() < 1e-7

        dm = np.array([dm, dm])
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc.get()).max() < 1e-7

    def test_get_vxc_mgga_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'r2scan'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10
        mf = pcell.RKS(xc=xc)

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(cell_nonorth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc.get()).max() < 1e-7

    def test_rks_lda(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.777337612, 8)

    def test_rks_gga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.87059063524272, 8)

    def test_rks_mgga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='scan').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.7542917283246, 8)

    def test_compact_basis_functions(self):
        cell = gto.M(
            a = np.diag([4., 8., 7.]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = [[0, [2e4, 1.]], [0, [1e2, 1.]], [0, [2., 1.]],
                     [1, [2e2, 1.]], [1, [1., 1.]]],
            mesh = [7, 7, 7],
        )
        np.random.seed(2)
        nao = cell.nao
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        ref = cell.RKS().get_rho(dm)
        out = multigrid.MultiGridNumInt(cell).get_rho(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)

if __name__ == '__main__':
    print("Full Tests for multigrid")
    unittest.main()
