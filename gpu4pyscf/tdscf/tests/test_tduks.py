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
from pyscf import lib, gto
from gpu4pyscf import tdscf

def diagonalize(a, b, nroots=4):
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    b_aa = b_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    b_ab = b_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    b_bb = b_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = np.block([[ a_aa  , a_ab],
                     [ a_ab.T, a_bb]])
    b = np.block([[ b_aa  , b_ab],
                     [ b_ab.T, b_bb]])
    abba = np.asarray(np.block([[a        , b       ],
                                      [-b.conj(),-a.conj()]]))
    e = np.linalg.eig(abba)[0]
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e


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

        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol1.basis = '631g'
        cls.mol1 = mol1.build()

        cls.mf_uhf = mf_uhf = mol.UHF().to_gpu().run()
        cls.td_hf = mf_uhf.TDHF().run(conv_tol=1e-6)

        mf_lda = mol.UKS().set(xc='lda', conv_tol=1e-12).to_gpu()
        mf_lda.grids.prune = None
        mf_lda.cphf_grids = mf_lda.grids
        cls.mf_lda = mf_lda.density_fit().run()

        mf_lda_nodf = mol.UKS().set(xc='lda', conv_tol=1e-12).to_gpu()
        mf_lda_nodf.grids.prune = None
        mf_lda_nodf.cphf_grids = mf_lda_nodf.grids
        cls.mf_lda_nodf = mf_lda_nodf.run()

        mf_bp86 = mol.UKS().set(xc='b88,p86', conv_tol=1e-12).to_gpu()
        mf_bp86.grids.prune = None
        mf_bp86.cphf_grids = mf_bp86.grids
        cls.mf_bp86 = mf_bp86.density_fit().run()

        mf_bp86_nodf = mol.UKS().set(xc='b88,p86', conv_tol=1e-12).to_gpu()
        mf_bp86_nodf.grids.prune = None
        mf_bp86_nodf.cphf_grids = mf_bp86_nodf.grids
        cls.mf_bp86_nodf = mf_bp86_nodf.run()

        mf_b3lyp = mol.UKS().set(xc='b3lyp5', conv_tol=1e-12).to_gpu()
        mf_b3lyp.grids.prune = None
        mf_b3lyp.cphf_grids = mf_b3lyp.grids
        cls.mf_b3lyp = mf_b3lyp.density_fit().run()

        mf_b3lyp_nodf = mol.UKS().set(xc='b3lyp5', conv_tol=1e-12).to_gpu()
        mf_b3lyp_nodf.grids.prune = None
        mf_b3lyp_nodf.cphf_grids = mf_b3lyp_nodf.grids
        cls.mf_b3lyp_nodf = mf_b3lyp_nodf.run()

        mf_m06l = mol.UKS().to_gpu().density_fit().run(xc='m06l')
        mf_m06l.cphf_grids = mf_m06l.grids
        cls.mf_m06l = mf_m06l

        mf_m06l_nodf = mol.UKS().to_gpu().run(xc='m06l')
        mf_m06l_nodf.cphf_grids = mf_m06l_nodf.grids
        cls.mf_m06l_nodf = mf_m06l_nodf

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_nohybrid_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.0476763425122965, 6)
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 4)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)

        td = self.mf_lda_nodf.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 4)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='lda, vwn_rpa').run()
        mf.cphf_grids = mf.grids
        td = mf.CasidaTDDFT().to_gpu()
        assert td.device == 'gpu'
        td.nstates = 5
        es = td.kernel()[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_nohybrid_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05383891686210346, 6)
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 4)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)

        td = self.mf_bp86_nodf.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 4)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)

    def test_tddft_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.0476763425122965, 6)

    def test_tddft_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05383891686259823, 6)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='b88,p86').run()
        mf.cphf_grids = mf.grids
        td = mf.TDDFT().to_gpu()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tddft_b3lyp(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDDFT()
        td.lindep=1.0E-6
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.047793873508724743, 6)

    def test_tddft_camb3lyp(self):
        mol1 = self.mol1
        mf = mol1.UKS(xc='camb3lyp').run()
        mf.cphf_grids = mf.grids
        td = mf.TDDFT().to_gpu()
        td.lindep=1.0E-6
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 7)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.2827429269753051, 6)
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)

        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 5)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 6)

    def test_tda_b3lyp(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.052638024165134974, 6)

    def test_tda_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05368082550881462, 6)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='lda,vwn').run()
        mf.cphf_grids = mf.grids
        td = mf.TDA().to_gpu()
        assert td.device == 'gpu'
        td.nstates = 5
        es = td.kernel()[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tda_m06l(self):
        mf_m06l = self.mf_m06l
        td = mf_m06l.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref[:5]).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(es), -0.7530329968766932, 5)

    def test_tda_vind(self):
        mf = self.mf_bp86
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDA().gen_vind()[0](zs)
        dat = mf.TDA().gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tddft_vind(self):
        mf = self.mf_b3lyp
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,2,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDDFT().gen_vind()[0](zs)
        dat = mf.TDDFT().gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_casida_tddft_vind(self):
        mf = self.mf_lda
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().CasidaTDDFT().gen_vind()[0](zs)
        dat = mf.CasidaTDDFT().gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_ab_hf(self):
        mf = self.mf_uhf
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_lda(self):
        mf = self.mf_lda_nodf
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_lda_df(self):
        mf = self.mf_lda
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = self.mf_b3lyp_nodf
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_b3lyp_df(self):
        mf = self.mf_b3lyp
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_mgga(self):
        mf = self.mf_m06l_nodf
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

    def test_ab_mgga_df(self):
        mf = self.mf_m06l
        td = tdscf.uks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = tdscf.uhf.gen_tda_operation(td, mf)[0]
        ftdhf = tdscf.uhf.gen_tdhf_operation(td, mf)[0]
        nocc_a = int(np.count_nonzero(mf.mo_occ[0] == 1))
        nvir_a = int(np.count_nonzero(mf.mo_occ[0] == 0))
        nocc_b = int(np.count_nonzero(mf.mo_occ[1] == 1))
        nvir_b = int(np.count_nonzero(mf.mo_occ[1] == 0))
        np.random.seed(2)
        xa, ya = np.random.random((2,nocc_a,nvir_a))
        xb, yb = np.random.random((2,nocc_b,nvir_b))
        x = np.hstack((xa.ravel(), xb.ravel()))
        y = np.hstack((ya.ravel(), yb.ravel()))
        xy = np.hstack((x, y))
        ax_a = np.einsum('iajb,jb->ia', a[0], xa)
        ax_a+= np.einsum('iajb,jb->ia', a[1], xb)
        ax_b = np.einsum('jbia,jb->ia', a[1], xa)
        ax_b+= np.einsum('iajb,jb->ia', a[2], xb)
        ax = np.hstack((ax_a.ravel(), ax_b.ravel()))
        self.assertAlmostEqual(abs(ax - ftda([x]).get()).max(), 0, 9)

        ay_a = np.einsum('iajb,jb->ia', a[0], ya)
        ay_a+= np.einsum('iajb,jb->ia', a[1], yb)
        ay_b = np.einsum('jbia,jb->ia', a[1], ya)
        ay_b+= np.einsum('iajb,jb->ia', a[2], yb)
        ay = np.hstack((ay_a.ravel(), ay_b.ravel()))

        bx_a = np.einsum('iajb,jb->ia', b[0], xa)
        bx_a+= np.einsum('iajb,jb->ia', b[1], xb)
        bx_b = np.einsum('jbia,jb->ia', b[1], xa)
        bx_b+= np.einsum('iajb,jb->ia', b[2], xb)
        bx = np.hstack((bx_a.ravel(), bx_b.ravel()))

        by_a = np.einsum('iajb,jb->ia', b[0], ya)
        by_a+= np.einsum('iajb,jb->ia', b[1], yb)
        by_b = np.einsum('jbia,jb->ia', b[1], ya)
        by_b+= np.einsum('iajb,jb->ia', b[2], yb)
        by = np.hstack((by_a.ravel(), by_b.ravel()))

        ab1 = ax + by
        ab2 =-bx - ay
        ab12 = np.hstack((ab1.ravel(),ab2.ravel()))
        abxy_ref = ftdhf(cp.asarray([xy])).get()
        self.assertAlmostEqual(abs(ab12 - abxy_ref).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for TD-UKS")
    unittest.main()
