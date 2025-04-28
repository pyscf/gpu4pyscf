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
from gpu4pyscf.tdscf import rhf, rks
from gpu4pyscf import tdscf


def diagonalize(a, b, nroots=4):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e = np.linalg.eig(np.asarray(h))[0]
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e = np.linalg.eig(np.asarray(a))[0]
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        cls.mol = mol.build()

        cls.mf = mf = mol.RHF().to_gpu().run()
        cls.td_hf = mf.TDHF().run(conv_tol=1e-6, lindep=1.0E-6)

        mf_lda = mol.RKS().to_gpu().density_fit()
        mf_lda.xc = 'lda, vwn'
        mf_lda.grids.prune = None
        mf_lda.cphf_grids = mf_lda.grids
        cls.mf_lda = mf_lda.run(conv_tol=1e-10)

        mf_lda_nodf = mol.RKS().to_gpu()
        mf_lda_nodf.xc = 'lda, vwn'
        mf_lda_nodf.cphf_grids = mf_lda.grids
        cls.mf_lda_nodf = mf_lda_nodf.run(conv_tol=1e-10)

        mf_bp86 = mol.RKS().to_gpu().density_fit()
        mf_bp86.xc = 'b88,p86'
        mf_bp86.grids.prune = None
        mf_bp86.cphf_grids = mf_bp86.grids
        cls.mf_bp86 = mf_bp86.run(conv_tol=1e-10)

        mf_bp86_nodf = mol.RKS().to_gpu()
        mf_bp86_nodf.xc = 'b88,p86'
        mf_bp86_nodf.cphf_grids = mf_bp86_nodf.grids
        cls.mf_bp86_nodf = mf_bp86_nodf.run(conv_tol=1e-10)

        mf_b3lyp = mol.RKS().to_gpu().density_fit()
        mf_b3lyp.xc = 'b3lyp5'
        mf_b3lyp.grids.prune = None
        mf_b3lyp.cphf_grids = mf_b3lyp.grids
        cls.mf_b3lyp = mf_b3lyp.run(conv_tol=1e-10)

        mf_b3lyp_nodf = mol.RKS().to_gpu()
        mf_b3lyp_nodf.xc = 'b3lyp5'
        mf_b3lyp_nodf.cphf_grids = mf_b3lyp_nodf.grids
        cls.mf_b3lyp_nodf = mf_b3lyp_nodf.run(conv_tol=1e-10)

        mf_m06l = mol.RKS().to_gpu().density_fit()
        mf_m06l.xc = 'm06l'
        mf_m06l.cphf_grids = mf_m06l.grids
        cls.mf_m06l = mf_m06l.run(conv_tol=1e-10)

        mf_m06l_nodf = mol.RKS().to_gpu()
        mf_m06l_nodf.xc = 'm06l'
        mf_m06l_nodf.cphf_grids = mf_m06l_nodf.grids
        cls.mf_m06l_nodf = mf_m06l_nodf.run(conv_tol=1e-10)

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_nohbrid_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)
        self.assertAlmostEqual(lib.fp(es), -1.5103950945691957, 5)

    def test_nohbrid_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.4869180666784665, 6)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

        td = self.mf_bp86_nodf.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tddft_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.5103950945691957, 6)

    def test_tddft_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.TDDFT()
        assert td.device == 'gpu'
        td.conv_tol = 1e-5
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.4869180666784665, 6)

    def test_tddft_b3lyp(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.5175884245769546, 6)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

        td = self.mf_b3lyp_nodf.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tddft_camb3lyp(self):
        mol = self.mol
        mf = mol.RKS(xc='camb3lyp').run()
        mf.cphf_grids = mf.grids
        td = mf.TDDFT().to_gpu()
        assert td.device == 'gpu'
        td.conv_tol = 1e-5
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 9.00540521503348, 6)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=4)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tda_b3lypg(self):
        mol = self.mol
        mf = mol.RKS()
        mf.xc = 'b3lypg'
        mf.grids.prune = None
        mf.cphf_grids = mf.grids
        mf.scf()
        td = mf.TDA().to_gpu()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.520888995669812, 6)

    def test_tda_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.5141057378565799, 6)

    def test_tda_b3lyp_triplet(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.4707787881198082, 6)
        td.analyze()
        a, b = td.get_ab()
        ref = diagonalize_tda(a, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

        mf_b3lyp_nodf = self.mf_b3lyp_nodf
        td = mf_b3lyp_nodf.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        td.lindep=1.0E-6
        es = td.kernel(nstates=5)[0]
        a, b = td.get_ab()
        ref = diagonalize_tda(a, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)


    def test_tda_lda_triplet(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        es = td.kernel(nstates=6)[0]
        ref = td.to_cpu().kernel(nstates=6)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[[0,1,2,4,5]]), -1.4695846533898422, 6)
        a, b = td.get_ab()
        ref = diagonalize_tda(a, nroots=6)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

        mf_lda_nodf = self.mf_lda_nodf
        td = mf_lda_nodf.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        td.lindep=1.0E-6
        es = td.kernel(nstates=5)[0]
        a, b = td.get_ab()
        ref = diagonalize_tda(a, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tddft_b88p86_triplet(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.TDDFT()
        assert td.device == 'gpu'
        td.singlet = False
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.4412243124430528, 6)
        a, b = td.get_ab()
        ref = diagonalize(a, b, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

        mf_bp86_nodf = self.mf_bp86_nodf
        td = mf_bp86_nodf.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        td.lindep=1.0E-6
        es = td.kernel(nstates=5)[0]
        a, b = td.get_ab()
        ref = diagonalize_tda(a, nroots=5)
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tda_rsh(self):
        mol = gto.M(atom='H 0 0 0.6; H 0 0 0', basis = "6-31g")
        mf = mol.RKS()
        mf.xc = 'wb97'
        mf.kernel()
        mf.cphf_grids = mf.grids
        td = mf.TDA().to_gpu()
        assert td.device == 'gpu'
        e_td = td.set(nstates=5).kernel()[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(e_td - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(e_td), 0.3953917940299652, 6)

    def test_tda_m06l_singlet(self):
        mf_m06l = self.mf_m06l
        td = mf_m06l.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.5620823865741496, 6)

    def test_analyze(self):
        td_hf = self.td_hf
        assert td_hf.device == 'gpu'
        f = td_hf.oscillator_strength(gauge='length')
        self.assertAlmostEqual(lib.fp(f), -0.13908774016795605, 5)
        f = td_hf.oscillator_strength(gauge='velocity', order=2)
        self.assertAlmostEqual(lib.fp(f), -0.096991134490587522, 5)

        note_args = []
        def temp_logger_note(rec, msg, *args):
            note_args.append(args)
        with lib.temporary_env(lib.logger.Logger, note=temp_logger_note):
            td_hf.analyze()
        ref = [(),
               (1, 11.834865910142547, 104.76181013351982, 0.01075359074556743),
               (2, 11.834865910142618, 104.76181013351919, 0.010753590745567499),
               (3, 16.66308427853695, 74.40651170629978, 0.3740302871966713)]
        self.assertAlmostEqual(abs(np.hstack(ref) -
                                   np.hstack(note_args)).max(), 0, 3)

        self.assertEqual(td_hf.nroots, td_hf.nstates)
        mf = self.mf
        self.assertAlmostEqual(lib.fp(td_hf.e_tot-mf.e_tot), 0.41508325757603637, 5)

    def test_scanner(self):
        mol = self.mol
        td_hf = self.td_hf
        td_scan = td_hf.as_scanner().as_scanner()
        td_scan.nroots = 3
        td_scan(mol)
        self.assertAlmostEqual(lib.fp(td_scan.e), 0.41508325757603637, 5)

    def test_transition_multipoles(self):
        td_hf = self.td_hf
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_dipole()             [2])), 0.39833021312014988, 4)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_quadrupole()         [2])), 0.14862776196563565, 4)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_octupole()           [2])), 2.79058994496489410, 4)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_velocity_dipole()    [2])), 0.24021409469918567, 4)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_magnetic_dipole()    [2])), 0                  , 4)
        self.assertAlmostEqual(abs(lib.fp(td_hf.transition_magnetic_quadrupole()[2])), 0.16558596265719450, 4)

    def test_reset(self):
        mol1 = gto.M(atom='C')
        mol = self.mol
        td = mol.RHF().newton().TDHF().to_gpu()
        assert td.device == 'gpu'
        td.reset(mol1)
        self.assertTrue(td.mol is mol1)
        self.assertTrue(td._scf.mol is mol1)

    def test_tda_vind(self):
        mf = self.mf_bp86
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,nocc,nvir)
        ref = mf.to_cpu().TDA().set(singlet=False).gen_vind()[0](zs)
        dat = mf.TDA().set(singlet=False).gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tddft_vind(self):
        mf = self.mf_b3lyp
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,2,nocc,nvir)
        ref = mf.to_cpu().TDDFT().set(singlet=True).gen_vind()[0](zs)
        dat = mf.TDDFT().set(singlet=True).gen_vind()[0](zs).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_casida_tddft_vind(self):
        mf = self.mf_lda
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,nocc,nvir)
        ref = mf.to_cpu().CasidaTDDFT().gen_vind()[0](zs)
        dat = mf.CasidaTDDFT().gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_ab_hf(self):
        mf = self.mf
        td = rhf.TDHF(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 5)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)
        
    def test_ab_lda(self):
        mf = self.mf_lda_nodf
        td = rks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_lda_df(self):
        mf = self.mf_lda
        td = rks.TDDFT(mf)
        a, b = td.get_ab(mf)
        ftda = rhf.gen_tda_operation(td, mf, singlet=True)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_b3lyp(self):
        mf = self.mf_b3lyp_nodf
        td = rks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_b3lyp_df(self):
        mf = self.mf_b3lyp
        td = rks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_mgga(self):
        mf = self.mf_m06l_nodf
        td = rks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_mgga_df(self):
        mf = self.mf_m06l
        td = rks.TDDFT(mf)
        a, b = td.get_ab()
        ftda = rhf.gen_tda_operation(td, mf, singlet=None)[0]
        ftdhf = rhf.gen_tdhf_operation(td, mf, singlet=True)[0]
        nocc = int(np.count_nonzero(mf.mo_occ == 2))
        nvir = int(np.count_nonzero(mf.mo_occ == 0))
        np.random.seed(2)
        x, y = xy = np.random.random((2,nocc,nvir))
        ax = np.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).get().reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + np.einsum('iajb,jb->ia', b, y)
        ab2 =-np.einsum('iajb,jb->ia', b, x)
        ab2-= np.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf(cp.asarray([xy])).get().reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
