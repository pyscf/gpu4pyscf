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
import io
import re
import cupy as cp
import numpy as np
from pyscf import gto, fci
from pyscf.data import nist
from gpu4pyscf.lib import logger
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

    def check_sf_observables(self, td, expect_y):
        if expect_y:
            self.assertGreater(sum(float(cp.linalg.norm(y)) for _, y in td.xy), 1e-6)
        dipoles = td.transition_dipole(ref=1)
        self.assertEqual(dipoles.shape, (td.nstates-1, 3))
        np.testing.assert_allclose(td.transition_dipole(ref=1, state=[2]), dipoles[:1], atol=1e-12)
        reverse = td.transition_dipole(ref=2, state=1)
        np.testing.assert_allclose(reverse, dipoles[:1].conj(), atol=1e-10)
        expected = (2./3.)*(td.e[1]-td.e[0])*np.vdot(dipoles[0], dipoles[0]).real
        self.assertAlmostEqual(float(td.oscillator_strength(ref=1, state=2)), expected, delta=1e-11)
        np.testing.assert_allclose(td.oscillator_strength(ref=1, state=[2]), [expected], atol=1e-11)
        np.testing.assert_allclose(td.oscillator_strength(ref=1), [expected], atol=1e-11)
        self.assertAlmostEqual(float(td.oscillator_strength(ref=2, state=1)), -expected, delta=1e-11)
        s2 = td.spin_square()
        self.assertTrue(np.isfinite(s2).all())
        self.assertAlmostEqual(float(td.spin_square(state=0)), s2[0], delta=1e-12)
        np.testing.assert_allclose(td.spin_square(state=[1, 0]), s2[[1, 0]], atol=1e-12)
        output = io.StringIO()
        td.analyze(verbose=logger.Logger(output, logger.INFO))
        rows = re.findall(r'Excited State\s+(\d+):\s+([-+\d.eE]+) eV\s+<S\^2>:\s+([-+\d.eE]+)',
                          output.getvalue())
        self.assertEqual([int(row[0]) for row in rows], list(range(1, td.nstates+1)))
        # Analyze prints energies to five decimals and spin squares to four.
        np.testing.assert_allclose([float(row[1]) for row in rows], td.e*nist.HARTREE2EV,
                                   atol=6e-6, rtol=0)
        np.testing.assert_allclose([float(row[2]) for row in rows], s2, atol=6e-5, rtol=0)
        original_xy = td.xy
        try:
            # A sign change of an actual eigenvector changes a transition
            # amplitude's sign, but cannot change its intensity or spin.
            x, y = original_xy[1]
            td.xy = [original_xy[0], (-x, -y)]
            np.testing.assert_allclose(td.transition_dipole(ref=1), -dipoles, atol=1e-10)
            self.assertAlmostEqual(float(td.oscillator_strength(ref=1, state=2)), expected, delta=1e-11)
            np.testing.assert_allclose(td.spin_square(), s2, atol=1e-10)
        finally:
            td.xy = original_xy

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
        self.check_sf_observables(td, expect_y=False)

        ref = np.array([0.4733582978, 0.6059906153])
        td = uhf.SpinFlipTDHF(mf).set(extype=0, collinear='col', nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=0, collinear='col', nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)
        self.check_sf_observables(td, expect_y=True)

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
        self.check_sf_observables(td, expect_y=False)

        ref = np.array([-0.2979385439, -0.0000000297])
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20, nstates=2).run()
        self.assertTrue(np.all(td.converged))
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, delta=1e-5)
        e = diagonalize_tddft(mf, extype=1, collinear='mcol', collinear_samples=20, nstates=2)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, delta=1e-5)
        self.check_sf_observables(td, expect_y=True)

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

    def _check_closed_shell_h2_spin_flip_triplets(self, extype):
        mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='631g', verbose=0)
        mf = mol.UHF().to_gpu().run(conv_tol=1e-12)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(float(mf.spin_square()[0]), 0., delta=1e-9)
        td = uhf.SpinFlipTDA(mf).set(extype=extype, nstates=2, conv_tol=1e-10).run()
        self.assertTrue(np.all(td.converged))
        a, _ = td.get_ab()
        active = a[0 if extype == 0 else 1].reshape(3, 3)
        np.testing.assert_allclose(td.e, np.linalg.eigvalsh(active)[:2], atol=1e-9)
        # Two electrons with equal spin are pure triplets, independent
        # of their spatial wavefunction or excitation energies.
        np.testing.assert_allclose(td.spin_square(), [2., 2.], atol=1e-9)
        self.check_sf_observables(td, expect_y=False)
        coeff = cp.asnumpy(mf.mo_coeff[0 if extype == 0 else 1])
        nmo = coeff.shape[1]
        nelec = (2, 0) if extype == 0 else (0, 2)
        ndet = fci.cistring.num_strings(nmo, 2)
        ci = []
        for x, _ in td.xy:
            vector = np.zeros((ndet, 1) if extype == 0 else (1, ndet))
            for a, amplitude in enumerate(x.get()[0], 1):
                address = fci.cistring.str2addr(nmo, 2, 1 | (1 << a))
                vector.flat[address] = amplitude
            ci.append(vector)
        # Independent Slater-determinant transition density for the
        # actual converged TDA amplitudes, not a second tensor formula.
        dm = fci.direct_spin1.trans_rdm1(ci[0], ci[1], nmo, nelec)
        dip_mo = coeff.T @ mol.intor_symmetric('int1e_r', comp=3) @ coeff
        expected = np.einsum('xij,ji->x', dip_mo, dm)
        self.assertGreater(np.linalg.norm(expected), 1e-3)
        np.testing.assert_allclose(td.transition_dipole(ref=1, state=2)[0], expected, atol=1e-9)

    def test_closed_shell_h2_spin_flip_triplets_spin_up(self):
        self._check_closed_shell_h2_spin_flip_triplets(0)

    def test_closed_shell_h2_spin_flip_triplets_spin_down(self):
        self._check_closed_shell_h2_spin_flip_triplets(1)

    def _check_single_root_empty_transition_selection(self, name, shape):
        mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='631g', verbose=0)
        mf = mol.UHF().to_gpu().run(conv_tol=1e-12)
        td = uhf.SpinFlipTDA(mf).set(extype=0, nstates=1, conv_tol=1e-10).run()
        self.assertTrue(np.all(td.converged))
        self.assertEqual(len(td.xy), 1)
        method = getattr(td, name)
        for targets in (None, [], [1], np.array([], dtype=int)):
            with self.subTest(observable=name, targets=targets):
                result = method(ref=1, state=targets)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, shape)
        for target in (1, np.int64(1)):
            with self.assertRaisesRegex(ValueError, 'must differ'):
                method(ref=1, state=target)
        for ref in (0, 2):
            with self.assertRaisesRegex(ValueError, '1-based'):
                method(ref=ref, state=[])

    def test_single_root_empty_transition_selection_dipole(self):
        self._check_single_root_empty_transition_selection('transition_dipole', (0, 3))

    def test_single_root_empty_transition_selection_oscillator_strength(self):
        self._check_single_root_empty_transition_selection('oscillator_strength', (0,))


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


class SpinFlipRestart(unittest.TestCase):
    def test_spin_flip_stored_restart_and_host_orbitals(self):
        for method in [uhf.SpinFlipTDA, uhf.SpinFlipTDHF]:
            with self.subTest(method=method):
                mol = gto.M(atom='H 0 0 0; H 0 0 .75; H 1.2 .3 .4', spin=1,
                            basis='sto-3g', verbose=0)
                mf = mol.UHF().to_gpu()
                mf.conv_tol = 1e-12
                assert mf.mo_energy is None
                td = method(mf).set(extype=1, collinear='col', conv_tol=1e-10, lindep=1e-24)
                # The complete small spin-lowering spectrum includes the spin-rotation mode.
                nstates = mf.nelec[0]*(mol.nao-mf.nelec[1]) if method is uhf.SpinFlipTDHF else 2
                td.kernel(nstates=nstates)
                assert mf.converged and np.all(td.converged)
                a, b = td.get_ab()
                nb = a[0].shape[0]*a[0].shape[1]
                na = a[1].shape[0]*a[1].shape[1]
                dense = a[1].reshape(na, na)
                if method is uhf.SpinFlipTDHF:
                    dense = np.block([[dense, b[1].reshape(na, nb)],
                                      [-b[0].reshape(nb, na), -a[0].reshape(nb, nb)]])
                    roots, vectors = np.linalg.eig(dense)
                    metric = np.sum(abs(vectors[:na])**2, axis=0) - np.sum(abs(vectors[na:])**2, axis=0)
                    assert np.max(abs(roots.imag)) < 1e-10
                    expected = np.sort(roots.real[metric > 1e-8])[:nstates]
                else:
                    expected = np.linalg.eigvalsh(dense)[:2]
                np.testing.assert_allclose(cp.asnumpy(td.e), expected, atol=1e-9, rtol=0)

                host_a, host_b = uhf.get_ab_sf(mf, mo_coeff=cp.asnumpy(mf.mo_coeff),
                                              mo_energy=cp.asnumpy(mf.mo_energy),
                                              mo_occ=cp.asnumpy(mf.mo_occ), collinear='col')
                for reference, actual in zip(a+b, host_a+host_b):
                    np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
                host_mf = mf.copy()
                host_mf.mo_coeff = tuple(cp.asnumpy(c) for c in mf.mo_coeff)
                host_td = method(host_mf).set(extype=1, collinear='col')
                vind, _ = host_td.gen_vind()
                np.testing.assert_allclose(vind(cp.eye(len(dense))).get(), dense.T, atol=1e-10)

                vectors = cp.stack([cp.concatenate((x.ravel(), y.ravel()))
                                    if isinstance(y, cp.ndarray) else x.ravel() for x, y in td.xy])
                transferred = td._transfer_initial_guess(td.xy, mf.mo_coeff, mf.mo_occ)
                np.testing.assert_allclose(transferred.get(), vectors.get(), atol=1e-10, rtol=0)
                # The nonsymmetric Davidson solver also compares successive energies.
                td.max_cycle = 2
                td.kernel(nstates=nstates)
                assert np.all(td.converged)
                np.testing.assert_allclose(cp.asnumpy(td.e), expected, atol=1e-9, rtol=0)
                for energy, (x, y) in zip(cp.asnumpy(td.e), td.xy):
                    vector = np.r_[x.get().ravel(), y.get().ravel()] if isinstance(y, cp.ndarray) else x.get().ravel()
                    assert np.linalg.norm(dense @ vector - energy*vector) < 1e-9


if __name__ == "__main__":
    print("Full Tests for spin-flip TDA and TDDFT with multicollinear functionals and collinear functionals")
    unittest.main()
