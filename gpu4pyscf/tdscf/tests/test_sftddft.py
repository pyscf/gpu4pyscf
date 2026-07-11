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
from pyscf import gto
from gpu4pyscf.tdscf import uhf

def diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=5):
    a, b = uhf.get_ab_sf(mf, collinear=collinear, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, np.zeros_like(B_abba_2d)],
                              [np.zeros_like(-B_baab_2d), -A_baba_2d]])
    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a*n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a*n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else: 
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e

def diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=5):
    a, b = uhf.get_ab_sf(mf, collinear=collinear, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                              [-B_baab_2d, -A_baba_2d]])
    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a*n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a*n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else: 
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_mcol_lda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().run()
        ref = np.array([0.4502240188, 0.5791758572])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([-0.3265810447,  0.0000000052])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_col_b3lyp(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().run()
        ref = np.array([0.4737123152, 0.6066070401])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([0.4733582978, 0.6059906153])
        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_mcol_tpss(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().run()
        ref = np.array([-0.2869994089,  0.0006366278])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([0.4478236446, 0.5654751841])
        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=0, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_mcol_cam(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().run()
        ref = np.array([-0.2975653443,  0.0006832701])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([-0.2979385439, -0.0000000297])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_df_mcol_lda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().density_fit().run()
        ref = np.array([0.4503402430, 0.5792893957])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([-0.3265288973, 0.0000000053])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_df_col_b3lyp(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().density_fit().run()
        ref = np.array([0.4738260866, 0.6067229861])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([0.4734730524, 0.6061069324])
        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_df_mcol_tpss(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().density_fit().run()
        ref = np.array([0.4499769856, 0.5708273458])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([-0.2873426295, -0.0000000019])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    def test_df_mcol_cam(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().density_fit().run()
        ref = np.array([-0.2975214622, 0.0006832504])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

        ref = np.array([-0.2978946526, -0.0000000329])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)

    # TODO: add a test for scanner
    def test_tda_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            spin = 2,
            basis = '631g')
        mf = mol.UHF().to_gpu().density_fit().run()
        td = mf.SFTDA()
        td.extype = 1
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, delta=1e-6)
    
    def test_tddft_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            spin = 2,
            basis = '631g')
        mf = mol.UHF().to_gpu().density_fit().run()
        td = mf.SFTDHF()
        td.extype = 0
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, delta=1e-6)


if __name__ == "__main__":
    print("Full Tests for spin-flip TDA and TDDFT with multicollinear functionals and collinear functionals")
    unittest.main()