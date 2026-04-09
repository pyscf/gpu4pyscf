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

def diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=5):
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

    def test_hf_tda(self):
        mf = self.mol.UKS(xc='HF').to_gpu().run()
        ref = np.array([0.4664408971, 0.5575560788])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2157451986,  0.0027042225])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_lda_tda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().run()
        ref = np.array([0.4502240188, 0.5791758572])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.3264298143,  0.0003751741])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().run()
        ref = np.array([0.4594126525, 0.5779958668])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2962917674,  0.0006700177])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_col_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().run()
        ref = np.array([0.4737123152, 0.6066070401])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2851971087,  0.0428751344])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().run()
        ref = np.array([0.449865372, 0.5707190159])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2869994089,  0.0006366278])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)
    
    def test_mcol_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().run()
        ref = np.array([0.4617250119, 0.5771124927])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2975653443,  0.0006832701])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)
    
    def test_col_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().run()
        ref = np.array([0.4754934156, 0.6049670138])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2873952196,  0.0303549298])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_hf_tda(self):
        mf = self.mol.UKS(xc='HF').to_gpu().density_fit().run()
        ref = np.array([0.4665429336, 0.5576517319])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2157436091, 0.0027040676])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_mcol_lda_tda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().density_fit().run()
        ref = np.array([0.4503402430, 0.5792893957])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.3263777300, 0.0003752049])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().density_fit().run()
        ref = np.array([0.4595219635, 0.5781024505])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2962480098, 0.0006700395])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_col_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().density_fit().run()
        ref = np.array([0.4738260866, 0.6067229861])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2851546772, 0.0428729313])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().density_fit().run()
        ref = np.array([0.4499769856, 0.5708273458])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2869471028, 0.0006366175])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_mcol_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().density_fit().run()
        ref = np.array([0.4618334410, 0.5772180434])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2975214622, 0.0006832504])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=50, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='mcol', collinear_samples=50, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

    def test_df_col_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().density_fit().run()
        ref = np.array([0.4756061400, 0.6050817906])
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)

        ref = np.array([-0.2873527047, 0.0303510215])
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 4)
        e = diagonalize_tda(mf, extype=1, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 4)


if __name__ == "__main__":
    print("Full Tests for spin-flip TDA with multicollinear functionals and collinear functionals")
    unittest.main()