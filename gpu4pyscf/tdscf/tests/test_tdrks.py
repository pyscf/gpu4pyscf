#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto
from gpu4pyscf import tdscf

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
        cls.td_hf = mf.TDHF().run(conv_tol=1e-6)

        mf_lda = mol.RKS().to_gpu().density_fit()
        mf_lda.xc = 'lda, vwn'
        mf_lda.grids.prune = None
        mf_lda.cphf_grids = mf_lda.grids
        cls.mf_lda = mf_lda.run(conv_tol=1e-10)

        mf_bp86 = mol.RKS().to_gpu().density_fit()
        mf_bp86.xc = 'b88,p86'
        mf_bp86.grids.prune = None
        mf_bp86.cphf_grids = mf_bp86.grids
        cls.mf_bp86 = mf_bp86.run(conv_tol=1e-10)

        mf_b3lyp = mol.RKS().to_gpu().density_fit()
        mf_b3lyp.xc = 'b3lyp5'
        mf_b3lyp.grids.prune = None
        mf_b3lyp.cphf_grids = mf_b3lyp.grids
        cls.mf_b3lyp = mf_b3lyp.run(conv_tol=1e-10)

        mf_m06l = mol.RKS().to_gpu().density_fit()
        mf_m06l.xc = 'm06l'
        mf_m06l.cphf_grids = mf_m06l.grids
        cls.mf_m06l = mf_m06l.run(conv_tol=1e-10)

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

    def test_tda_lda_triplet(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDA()
        assert td.device == 'gpu'
        td.singlet = False
        es = td.kernel(nstates=6)[0]
        ref = td.to_cpu().kernel(nstates=6)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[[0,1,2,4,5]]), -1.4695846533898422, 6)

    def test_tddft_b88p86_triplet(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.TDDFT()
        assert td.device == 'gpu'
        td.singlet = False
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -1.4412243124430528, 6)

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
        dat = mf.TDA().set(singlet=False).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tddft_vind(self):
        mf = self.mf_b3lyp
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,2,nocc,nvir)
        ref = mf.to_cpu().TDDFT().set(singlet=True).gen_vind()[0](zs)
        dat = mf.TDDFT().set(singlet=True).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_casida_tddft_vind(self):
        mf = self.mf_lda
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,nocc,nvir)
        ref = mf.to_cpu().CasidaTDDFT().set().gen_vind()[0](zs)
        dat = mf.CasidaTDDFT().set().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for TD-RKS")
    unittest.main()
