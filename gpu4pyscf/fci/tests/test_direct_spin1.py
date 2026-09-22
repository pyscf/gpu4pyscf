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

from gpu4pyscf.fci.direct_spin1 import FCISolver
from gpu4pyscf.fci.direct_spin1 import contract_2e
from gpu4pyscf.fci.direct_spin1 import make_rdm12
from pyscf import ao2mo, mcscf, scf
from pyscf.fci import direct_spin1
from pyscf.fci.addons import fix_spin_


class KnownValues(unittest.TestCase):
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
        for guess in (ci0, cp.asarray(ci0), excited_guess):
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


if __name__ == '__main__':
    unittest.main()
