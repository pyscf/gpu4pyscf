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

def cal_exact_sf_tda_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1):
    a, b = uhf.get_ab_sf(mf, collinear=collinear, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a), order='C')

    Casida_matrix = np.block([[A_abab_2d, np.zeros_like(B_abba_2d)], [-np.zeros_like(B_baab_2d), -A_baba_2d]])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[: n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b :], axis=0) ** 2

    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
        eigenvecs = eigenvecs[:, mask].transpose(1, 0)

        def norm_xy(z):
            x = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            y = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, 0)
    else:
        mask = norms < -1e-3
        valid_e = -eigenvals[mask].real
        idx = np.argsort(valid_e)
        valid_e = valid_e[::-1]
        eigenvecs = eigenvecs[:, mask][:, ::-1].transpose(1, 0)

        def norm_xy(z):
            x = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            y = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, 0)

    fake_td = uhf.SpinFlipTDA(mf)
    if collinear == 'mcol':
        fake_td.collinear = 'mcol'
        fake_td.collinear_samples = collinear_samples
    else:
        fake_td.collinear = collinear
    fake_td.extype = extype
    fake_td.e = valid_e
    fake_td.xy = [norm_xy(z) for z in eigenvecs]

    tdg = fake_td.Gradients()
    return tdg.kernel(state=state)


def cal_exact_sf_tddft_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1):
    a, b = uhf.get_ab_sf(mf, collinear=collinear, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a), order='C')

    Casida_matrix = np.block([[A_abab_2d, B_abba_2d], [-B_baab_2d, -A_baba_2d]])

    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[: n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b :], axis=0) ** 2

    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
        eigenvecs = eigenvecs[:, mask].transpose(1, 0)

        def norm_xy(z):
            x = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            y = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, y * norm)
    else:
        mask = norms < -1e-3
        valid_e = -eigenvals[mask].real
        idx = np.argsort(valid_e)
        valid_e = valid_e[::-1]
        eigenvecs = eigenvecs[:, mask][:, ::-1].transpose(1, 0)

        def norm_xy(z):
            x = z[n_occ_a * n_virt_b :].reshape(n_occ_b, n_virt_a)
            y = z[: n_occ_a * n_virt_b].reshape(n_occ_a, n_virt_b)
            norm = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
            norm = np.sqrt(1.0 / norm)
            return (x * norm, y * norm)

    fake_td = uhf.SpinFlipTDHF(mf)
    if collinear == 'mcol':
        fake_td.collinear = 'mcol'
        fake_td.collinear_samples = collinear_samples
    else:
        fake_td.collinear = collinear
    fake_td.extype = extype
    fake_td.e = valid_e
    fake_td.xy = [norm_xy(z) for z in eigenvecs]

    tdg = fake_td.Gradients()
    return tdg.kernel(state=state)


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587"""
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_mcol_lda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [-1.1347069729e-16, -1.1084528691e-14, 2.8558017754e-01],
                [-1.1092137160e-16, 2.7449714063e-01, -1.4279559143e-01],
                [5.9671903444e-17, -2.7449714063e-01, -1.4279559143e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [-6.7904482786e-16, 1.2476156342e-14, 3.6571945011e-03],
                [-9.3757276431e-17, 1.4533596842e-02, -1.8349716472e-03],
                [4.8848910182e-17, -1.4533596842e-02, -1.8349716472e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

    def test_col_b3lyp(self):
        mf = self.mol.UKS(xc='B3LYP').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=0, collinear='col').run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=0, collinear='col', state=1)
        ref = np.array(
            [
                [-2.5468184163e-16, -1.1180009418e-14, 2.9019466200e-01],
                [-9.0775873947e-17, 2.8359926220e-01, -1.4509830953e-01],
                [1.1512910412e-16, -2.8359926220e-01, -1.4509830953e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='col').run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear='col', state=1)
        ref = np.array(
            [
                [-1.5105624594e-16, 1.2602467888e-15, 2.9075066796e-01],
                [6.5823041051e-18, 2.8383333522e-01, -1.4537631436e-01],
                [-7.5347051462e-17, -2.8383333522e-01, -1.4537631436e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

    def test_mcol_tpss(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [1.4878309854e-16, 1.8283875050e-14, 8.7167801429e-03],
                [4.6186501645e-17, 1.4280801923e-02, -4.3644887121e-03],
                [-1.1302550886e-16, -1.4280801923e-02, -4.3644887121e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=0, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [-2.8308341496e-16, 1.0704054160e-15, 3.0645165527e-01],
                [-1.1971039167e-16, 2.8704791604e-01, -1.5324080858e-01],
                [3.3992205735e-17, -2.8704791604e-01, -1.5324080858e-01],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

    def test_mcol_cam(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tda_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [1.5152147863e-16, 1.9028098735e-15, -1.0403780070e-02],
                [-1.0684663896e-16, 1.1793709730e-02, 5.1952611230e-03],
                [-1.1109928193e-16, -1.1793709730e-02, 5.1952611230e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        grad_iter = td.Gradients().kernel(state=1)
        grad_exact = cal_exact_sf_tddft_gradient(mf, extype=1, collinear='mcol', collinear_samples=20, state=1)
        ref = np.array(
            [
                [-4.5098470283e-16, 2.7736173739e-15, -1.0510769880e-02],
                [-1.9483417972e-16, 1.1800860322e-02, 5.2487299209e-03],
                [1.1710777201e-16, -1.1800860322e-02, 5.2487299209e-03],
            ]
        )
        self.assertAlmostEqual(abs(grad_exact - ref).max(), 0, delta=1e-5)
        self.assertAlmostEqual(abs(grad_iter - ref).max(), 0, delta=1e-5)

    def test_grad_b3lyp_tda_spinflip_up_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu")
        assert abs(etot - -75.9674347270528) < 1e-8
        assert abs(e - np.array([0.46618494, 0.53438998, 0.60047275, 0.65786033, 0.92091718])).max() < 1e-5
        ref = np.array([[ 8.79547051e-16,  8.63728537e-14,  1.87755267e-01],
                        [-4.31890391e-16,  2.15026042e-01, -9.38746716e-02],
                        [-4.50003252e-16, -2.15026042e-01, -9.38746716e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5
        
    def test_grad_b3lyp_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu", extype=1)
        assert abs(etot - -75.96743472705282) < 1e-8
        assert abs(e - np.array([0.0034149,  0.08157731, 0.23027453, 0.50644857, 0.51065628])).max() < 1e-5
        ref = np.array([[-3.01640558e-16,  1.52982216e-13,  5.10689029e-02],
                        [ 1.36165869e-16,  4.52872857e-02, -2.55387304e-02],
                        [-3.08111636e-17, -4.52872857e-02, -2.55387304e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_svwn_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="svwn", tol=5e-10, method="cpu", extype=1)
        assert abs(etot - -75.39033965461661) < 1e-8
        assert abs(e - np.array([0.00210504, 0.07530215, 0.22255285, 0.50300732, 0.50382963])).max() < 1e-5
        ref = np.array([[-8.15030724e-16, -6.13885762e-14,  6.41681368e-02],
                        [ 1.12931062e-16,  5.34632826e-02, -3.20887796e-02],
                        [ 7.97399496e-17, -5.34632826e-02, -3.20887796e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_camb3lyp_tda_spinflip_down_cpu(self):
        etot, e, grad_gpu = _check_grad(mol, xc="camb3lyp", tol=5e-10, method="cpu", extype=1)
        assert abs(etot - -75.93920847775132) < 1e-8
        assert abs(e - np.array([0.00335301, 0.07772481, 0.2267033, 0.50960632, 0.5133939])).max() < 1e-5
        ref = np.array([[-7.43754261e-18, -1.56347842e-13,  4.99263503e-02],
                        [-1.84572351e-17,  4.52908126e-02, -2.49673842e-02],
                        [ 2.40683934e-17, -4.52908126e-02, -2.49673842e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_b3lyp_tda_spinflip_up_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="b3lyp", tol=5e-10, method="cpu")
        assert abs(etot - -76.42037833354925) < 1e-8
        assert abs(e - np.array([0.25433265, 0.33124974, 0.3313682, 0.40247177, 0.47307456])).max() < 1e-5
        ref = np.array([[ 1.29088518e-16,  6.98423827e-14,  1.25014262e-01],
                        [-1.36624149e-16,  8.37484153e-02, -6.25098673e-02],
                        [ 1.80012190e-16, -8.37484153e-02, -6.25098673e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5
        
    def test_grad_b3lyp_tda_spinflip_down_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="b3lyp", tol=5e-10, method="cpu", extype=1)
        assert abs(etot - -76.42037833354925) < 1e-8
        assert abs(e - np.array([0.2543327, 0.33124974, 0.3313685, 0.40247202, 0.4730746])).max() < 1e-5
        ref = np.array([[-5.16805682e-16,  7.28823057e-14,  1.25014068e-01],
                        [ 1.94935391e-16,  8.37484121e-02, -6.25097703e-02],
                        [ 1.20139074e-17, -8.37484121e-02, -6.25097703e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_svwn_tda_spinflip_down_cpu_closed(self):
        etot, e, grad_gpu = _check_grad(mol1, xc="svwn", tol=5e-10, method="cpu", extype=1)
        assert abs(etot - -75.85470242125601) < 1e-8
        assert abs(e - np.array([0.25020513, 0.32400566, 0.32879602, 0.39954396, 0.47440403])).max() < 1e-5
        ref = np.array([[-1.04007210e-16,  2.76349222e-15,  1.40334993e-01],
                        [ 4.57442221e-17,  9.05506406e-02, -7.01720839e-02],
                        [ 1.95402062e-16, -9.05506406e-02, -7.01720839e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5


if __name__ == '__main__':
    print('Full Tests for spin-flip TDA and TDDFT analytic gradient with multicollinear functionals and collinear functionals')
    unittest.main()