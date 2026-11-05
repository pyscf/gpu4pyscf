# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import numpy as np
import pyscf

from gpu4pyscf import fci as gpu_fci
from gpu4pyscf.fci.direct_spin1 import FCISolver
from gpu4pyscf.fci.direct_spin1 import contract_2e
from gpu4pyscf.fci.direct_spin1 import make_rdm12
from pyscf import ao2mo, mcscf, scf
from pyscf.fci import direct_spin1
from pyscf.fci.addons import fix_spin_


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = pyscf.M(atom='H 0 0 0; H .9 .1 0; H 1.9 -.2 .15; H 2.7 .35 -.1',
                      basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        assert mf.converged
        coefficients = mf.mo_coeff
        h1 = coefficients.T @ mf.get_hcore() @ coefficients
        eri = ao2mo.restore(4, ao2mo.kernel(mol, coefficients), 4)
        cls.h4_integrals = mol, h1, eri

    @classmethod
    def tearDownClass(cls):
        del cls.h4_integrals

    def test_spin_penalty_unsupported(self):
        solver = fix_spin_(FCISolver(), shift=2., ss=2.)
        with self.assertRaisesRegex(NotImplementedError, 'spin penalties'):
            solver.kernel(np.diag([0., 1.]), np.zeros((2, 2, 2, 2)),
                          2, (1, 1))

    def test_multiple_roots_from_single_guess(self):
        h1e = np.diag([0., 1.])
        eri = np.zeros((2, 2, 2, 2))
        cpu = direct_spin1.FCISolver()
        _, ci0 = cpu.kernel(h1e, eri, 2, (1, 1))
        ref, _ = cpu.kernel(h1e, eri, 2, (1, 1), ci0=ci0, nroots=2)
        hdiag = cpu.make_hdiag(h1e, eri, 2, (1, 1))
        excited_guess = cpu.get_init_guess(2, (1, 1), 2, hdiag)[1]
        for guess in (cp.asarray(ci0), cp.asarray(excited_guess)):
            with self.subTest(guess=guess):
                solver = FCISolver()
                energies, roots = solver.kernel(
                    h1e, eri, 2, (1, 1), ci0=guess, nroots=2)
                self.assertTrue(solver.converged.all())
                self.assertEqual(len(roots), 2)
                np.testing.assert_allclose(energies, ref, atol=1e-10)

    def test_cpu_gpu(self):
        norb = 6
        nelec = (3, 3)
        mol = pyscf.M(
            atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        mc = mcscf.CASCI(mf, norb, sum(nelec))
        mc.canonicalization = False
        e_ref, _, ci = mc.kernel()[:3]
        h1e, ecore = mc.get_h1eff()
        eri = ao2mo.restore(4, mc.get_h2eff(), norb)

        link = direct_spin1._unpack(norb, nelec, None)
        ref = direct_spin1.contract_2e(eri, ci, norb, nelec, link)
        out = contract_2e(eri, cp.asarray(ci), norb, nelec, link)
        self.assertLess(abs(out.get() - ref).max(), 1e-10)

        solver = FCISolver()
        e_gpu, ci_gpu = solver.kernel(
            h1e, eri, norb, nelec, ecore=ecore, max_memory=0)

        self.assertTrue(solver.converged)
        self.assertIsInstance(ci_gpu, cp.ndarray)
        self.assertLess(abs(e_gpu - e_ref), 1e-8)

        dm1, dm2 = make_rdm12(ci_gpu, norb, nelec)
        ref1, ref2 = direct_spin1.make_rdm12(
            cp.asnumpy(ci_gpu), norb, nelec)
        self.assertLess(abs(dm1.get() - ref1).max(), 1e-10)
        self.assertLess(abs(dm2.get() - ref2).max(), 1e-10)

    def test_callable_ci0_with_small_davidson_space(self):
        mol, h1, eri = self.h4_integrals
        norb, nelec, nroots = 4, (2, 2), 2
        cpu = direct_spin1.FCISolver()
        diagonal = cpu.make_hdiag(h1, eri, norb, nelec)
        initial = np.asarray(cpu.get_init_guess(norb, nelec, nroots, diagonal))
        supplied = cp.asarray(initial)
        reference, _ = cpu.kernel(h1, eri, norb, nelec, nroots=nroots,
                                  ecore=mol.energy_nuc(), tol=1e-12)
        solver = gpu_fci.solver(mol)
        solver.conv_tol_residual = 1e-9
        # Restrict the Davidson space below the 36-determinant CI space.
        energies, roots = solver.kernel(
            h1, eri, norb, nelec, ci0=lambda: supplied, nroots=nroots,
            ecore=mol.energy_nuc(), tol=1e-12, lindep=1e-24,
            max_cycle=80, max_space=4, verbose=0)
        assert bool(np.asarray(solver.converged).all())
        np.testing.assert_array_equal(cp.asnumpy(supplied), initial)
        np.testing.assert_allclose(energies, reference, atol=1e-9, rtol=0)
        vectors = np.stack([c.get().ravel() for c in roots])
        np.testing.assert_allclose(vectors @ vectors.T, np.eye(nroots), atol=1e-10)
        h2 = cpu.absorb_h1e(h1, eri, norb, nelec, .5)
        for energy, ci in zip(energies, roots):
            ci = ci.get()
            residual = cpu.contract_2e(h2, ci, norb, nelec)
            residual -= (energy-mol.energy_nuc()) * ci
            assert np.linalg.norm(residual) < 1e-9

    def test_open_shell_density_and_spin(self):
        mol, h1, eri = self.h4_integrals
        norb, nelec = 4, (3, 1)
        cpu = direct_spin1.FCISolver()
        reference, reference_ci = cpu.kernel(h1, eri, norb, nelec, tol=1e-12)
        solver = gpu_fci.solver(mol)
        solver.spin = 2
        energy, ci = solver.kernel(h1, eri, norb, nelec, tol=1e-12, verbose=0)
        assert solver.converged
        self.assertAlmostEqual(energy, reference, delta=1e-9)
        np.testing.assert_allclose(solver.spin_square(ci, norb, nelec),
                                   cpu.spin_square(reference_ci, norb, nelec), atol=1e-9)
        dm_alpha, dm_beta = solver.make_rdm1s(ci, norb, nelec)
        np.testing.assert_allclose([np.trace(dm_alpha), np.trace(dm_beta)], nelec, atol=1e-10)
        dm1, dm2 = make_rdm12(ci, norb, nelec)
        ref1, ref2 = cpu.make_rdm12(ci.get(), norb, nelec)
        np.testing.assert_allclose(dm1.get(), ref1, atol=1e-11)
        np.testing.assert_allclose(dm2.get(), ref2, atol=1e-11)
        np.testing.assert_allclose(dm1.get(), dm_alpha+dm_beta, atol=1e-12)
        rdm_energy = np.einsum('pq,qp', h1, dm1.get())
        rdm_energy += .5*np.einsum('pqrs,pqrs', ao2mo.restore(1, eri, norb), dm2.get())
        self.assertAlmostEqual(rdm_energy, energy, delta=1e-10)


if __name__ == '__main__':
    unittest.main()
