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
from gpu4pyscf.pbc.dft import KRKS as KRKS_gpu
from gpu4pyscf.pbc.dft import KUKS as KUKS_gpu
import pytest

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
        # self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc().get()
        # self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc_nonorth(self):
        ref = MultiGridNumInt_cpu(cell_nonorth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_nonorth).get_nuc().get()
        # self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc_kpts(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

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
        ref = multigrid_cpu.multigrid.get_rho(MultiGridNumInt_cpu(cell_orth), dm)
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

    def test_get_vxc_lda_kpts(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        nkpts = len(kpts)
        dm = np.empty((2,nkpts,nao,nao), dtype = np.complex128)
        dm.real = np.random.random((2,nkpts,nao,nao)) - .5
        dm.imag = np.random.random((2,nkpts,nao,nao)) - .5
        dm = np.einsum('ukpr,ukqr->ukpq', dm, dm.conj()) # Make sure dm is Hermitian positive definite, so rho > 0 and tau > 0
        pcell = cell_orth.copy()
        pcell.precision = 1e-10

        mf = pcell.KUKS(xc=xc)
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]

        ### Henry 20250909: The CPU multigrid reference result for UKS is wrong both in value and in format in pyscf==2.8.0.
        # if hasattr(multigrid_cpu, 'nr_uks'):
        #     n0, exc0, ref = multigrid_cpu.nr_uks(
        #         MultiGridNumInt_cpu(pcell), xc, dm, with_j=True, kpts=kpts)
        # else:
        #     n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_uks(
        #         pcell, None, xc, dm, kpts=kpts)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
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

    def test_get_vxc_gga_kpts(self):
        nao = cell_orth.nao
        np.random.seed(20)
        xc = 'pbe,'
        nkpts = len(kpts)
        dm = np.empty((2,nkpts,nao,nao), dtype = np.complex128)
        dm.real = np.random.random((2,nkpts,nao,nao)) - .5
        dm.imag = np.random.random((2,nkpts,nao,nao)) - .5
        dm = np.einsum('ukpr,ukqr->ukpq', dm, dm.conj()) # Make sure dm is Hermitian positive definite, so rho > 0 and tau > 0
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

    def test_get_vxc_gga_kpts_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'pbe,'
        nkpts = len(kpts)
        dm = np.empty((nkpts,nao,nao), dtype = np.complex128)
        dm.real = np.random.random((nkpts,nao,nao)) - .5
        dm.imag = np.random.random((nkpts,nao,nao)) - .5
        dm = np.einsum('kpr,kqr->kpq', dm, dm.conj()) # Make sure dm is Hermitian positive definite, so rho > 0 and tau > 0
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

    def test_get_vxc_mgga_kpts(self):
        nao = cell_orth.nao
        np.random.seed(3)
        xc = 'r2scan'
        nkpts = len(kpts)
        dm = np.empty((nkpts,nao,nao), dtype = np.complex128)
        dm.real = np.random.random((nkpts,nao,nao)) - .5
        dm.imag = np.random.random((nkpts,nao,nao)) - .5
        dm = np.einsum('kpr,kqr->kpq', dm, dm.conj()) # Make sure dm is Hermitian positive definite, so rho > 0 and tau > 0
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        mf = pcell.KRKS(xc=xc)

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm, kpts=kpts)
        # vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        # ref += vj
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=False, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc.get()).max() < 1e-7

        dm = np.array([dm, dm])
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
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

    def test_get_vxc_mgga_kpts_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(4)
        xc = 'r2scan'
        nkpts = len(kpts)
        dm = np.empty((nkpts,nao,nao), dtype = np.complex128)
        dm.real = np.random.random((nkpts,nao,nao)) - .5
        dm.imag = np.random.random((nkpts,nao,nao)) - .5
        dm = np.einsum('kpr,kqr->kpq', dm, dm.conj()) # Make sure dm is Hermitian positive definite, so rho > 0 and tau > 0
        pcell = cell_nonorth.copy()
        mf = pcell.KRKS(xc=xc)

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(cell_nonorth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc.get()).max() < 1e-7

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    def test_band_rks_gamma(self):
        cell = gto.M(
            verbose = 0,
            a = np.diag([3.6, 3.2, 4.5]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        np.random.seed(1)
        kpts_band = np.random.random((4,3))

        test_mf = cell.RKS(xc='r2scan').to_gpu()
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.RKS(xc='r2scan')
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-7
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_band_krks_kpts(self):
        cell = gto.M(
            verbose = 0,
            a = np.array([[3.6, 0, 0], [0, 3.2, 0.2], [0, 0, 4.5]]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        kpts = cell.make_kpts([1,3,1])

        np.random.seed(1)
        kpts_band = np.random.random((1,3)) # Yes, one non-zero k point, as an edge case

        test_mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.KRKS(xc='pbe', kpts=kpts)
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-8
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_band_kuks_kpts(self):
        cell = gto.M(
            verbose = 0,
            a = np.diag([3.6, 3.2, 4.5]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        kpts = cell.make_kpts([1,1,3])

        np.random.seed(1)
        kpts_band = np.random.random((2,3))


        test_mf = cell.KUKS(xc='lda', kpts=kpts).to_gpu()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.KUKS(xc='lda', kpts=kpts)
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-7
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_unique_image_pairs(self):
        Lx = np.append(np.arange(0, 4), np.arange(-5, 0))
        Ly = np.append(np.arange(0, 3), np.arange(-4, 0))
        Lz = np.append(np.arange(0, 4), np.arange(-2, 0))
        Ls = lib.cartesian_prod([Lx, Ly, Lz])
        Ls = cp.array(Ls)
        ret = multigrid._unique_image_pair(Ls)
        assert ret[0].shape == (2431, 3)
        assert abs(lib.fp(ret[0].get()) - 1.5047201402319172) < 1e-10
        assert abs(lib.fp(ret[1].get()) - -483.0210637951298) < 1e-10

        np.random.seed(2)
        Ls = Ls[np.random.rand(len(Ls)) > .5]
        ret = multigrid._unique_image_pair(Ls)
        assert ret[0].shape == (2377, 3)
        assert abs(lib.fp(ret[0].get()) - -34.99306750756055) < 1e-10
        assert abs(lib.fp(ret[1].get()) - 1347.464804553046) < 1e-10

        Lx = np.append(np.arange(0, 3), np.arange(-3, 0))
        Ly = np.append(np.arange(0, 3), np.arange(-3, 0))
        Lz = np.append(np.arange(0, 3), np.arange(-3, 0))
        Ls = lib.cartesian_prod([Lx, Ly, Lz])
        Ls = cp.array(Ls)
        ret = multigrid._unique_image_pair(Ls)
        assert ret[0].shape == (1331, 3)
        assert abs(lib.fp(ret[0].get()) - -1.355090495130784) < 1e-10
        assert abs(lib.fp(ret[1].get()) - 2145.771285837819) < 1e-10

    def test_image_pair_to_difference(self):
        cell = gto.M(a=np.eye(3)*3, atom='He 0. 0. 0.', basis=[[0, [1, 1]]])
        Ls = cell.get_lattice_Ls()
        difference_images, inverse = multigrid.image_pair_to_difference(Ls, cell.lattice_vectors())
        assert difference_images.shape == (25, 3)
        assert len(inverse) == len(Ls)**2

    def test_shell_splitting_for_large_fock_in_imagediff_space_gamma(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8817  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 0,
        )

        kpts = cell.make_kpts([1,1,1])
        mf = KRKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**28)):
            mf = mf.multigrid_numint()
            assert type(mf._numint) is multigrid.MultiGridNumInt

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -44.93180128532909
        ref_gradient = np.array([
            [ 2.87614262e-03,  1.33298682e-03,  1.33298682e-03],
            [-8.42690061e-03,  1.01735391e-05,  1.01735391e-05],
            [ 2.82851274e-03,  1.28135252e-03, -1.28134877e-03],
            [-1.00471252e-04, -8.51551903e-06,  8.51502092e-06],
            [ 2.82851274e-03, -1.28134877e-03,  1.28135252e-03],
            [-1.00471252e-04,  8.51502093e-06, -8.51551903e-06],
            [ 2.87618947e-03, -1.33322853e-03, -1.33322853e-03],
            [-2.78640919e-03, -8.51738633e-06, -8.51738633e-06],
        ])

        assert abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-8

    def test_shell_splitting_for_large_fock_in_imagediff_space_k(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8817  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 5,
        )

        kpts = cell.make_kpts([1,1,3])
        mf = KRKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**28)):
            mf = mf.multigrid_numint()
            assert type(mf._numint) is multigrid.MultiGridNumInt

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -45.30199423477792
        ref_gradient = np.array([
            [ 2.36897360e-03,  1.19228739e-03,  6.67961844e-04],
            [-9.11593616e-03,  1.03369618e-05,  8.42224689e-06],
            [ 2.32829121e-03,  1.14450844e-03, -6.20435413e-04],
            [ 2.38310045e-04, -8.52269883e-06,  6.81929906e-06],
            [ 2.32828725e-03, -1.14450700e-03,  6.20436720e-04],
            [ 1.14957254e-03,  8.52837460e-06, -6.81790444e-06],
            [ 2.36729361e-03, -1.19142895e-03, -6.64432085e-04],
            [-1.66792993e-03, -8.52961493e-06, -6.81679641e-06],
        ])

        assert abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-8

    def test_shell_splitting_for_large_fock_in_imagediff_space_unrestricted(self):
        cell = gto.M(
            a = '''
                0.000000000, 3.370137329, 3.370137329
                3.370137329, 0.000000000, 3.370137329
                3.370137329, 3.370137329, 0.000000000
            ''',
            atom = '''
                C 0 0 0
                C 1.675068664391 1.685068664391 1.685068664391
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 5,
        )

        kpts = cell.make_kpts([1,1,3])
        mf = KUKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**25)):
            mf = mf.multigrid_numint()
            assert type(mf._numint) is multigrid.MultiGridNumInt

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -10.82283467913058
        ref_gradient = np.array([
            [-0.00776978, -0.00816341,  0.00816341],
            [ 0.00777063,  0.00816369, -0.00816369],
        ])

        assert abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-7

if __name__ == '__main__':
    print("Full Tests for multigrid")
    unittest.main()
