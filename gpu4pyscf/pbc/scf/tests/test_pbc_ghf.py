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
import numpy as np
from pyscf.pbc import gto, scf
from gpu4pyscf.pbc import scf as gpu_scf


def setUpModule():
    global cell
    # Adapted from pyscf/pbc/scf/test/test_hf.py: test_ghf_exx_ewald.
    cell = gto.M(
        unit='B', a=np.eye(3)*4, mesh=[21]*3, precision=1e-10,
        atom='He 1.5 2 1.5; He 2 2 2.5',
        basis={'He': [[0, (.8, 1.)], [0, (1., 1.)], [0, (1.2, 1.)]]},
        verbose=0)


class KnownValues(unittest.TestCase):
    def test_ghf_exx_ewald(self):
        for kpt in (np.zeros(3), np.array([.12, .08, .05])):
            with self.subTest(kpt=kpt):
                ref = scf.GHF(cell, kpt, exxdiv='ewald').run(
                    init_guess='hcore', conv_tol=1e-10)
                mf = gpu_scf.GHF(cell, kpt, exxdiv='ewald').run(
                    init_guess='hcore', conv_tol=1e-10)
                self.assertTrue(ref.converged and mf.converged)
                self.assertAlmostEqual(mf.e_tot, ref.e_tot, delta=1e-7)
                # GPU k-point meshes must include Gamma.
                if not kpt.any():
                    kmf = gpu_scf.KGHF(cell, kpt[None], exxdiv='ewald').run(
                        init_guess='hcore', conv_tol=1e-10)
                    self.assertTrue(kmf.converged)
                    self.assertAlmostEqual(kmf.e_tot, ref.e_tot, delta=1e-7)
                self.assertEqual(mf.mo_coeff.dtype,
                                 np.float64 if not kpt.any() else np.complex128)

    def test_kghf_kmesh(self):
        kpts = cell.make_kpts([2, 1, 1])
        ref = scf.KGHF(cell, kpts, exxdiv='ewald').run(
            init_guess='hcore', conv_tol=1e-10)
        mf = gpu_scf.KGHF(cell, kpts, exxdiv='ewald').run(
            init_guess='hcore', conv_tol=1e-10)
        self.assertTrue(ref.converged and mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, delta=1e-7)
        dm = mf.make_rdm1()
        vj, vk = mf.get_jk(dm=dm, hermi=1)
        jref, kref = ref.get_jk(dm=dm.get(), hermi=1)
        np.testing.assert_allclose(vj.get(), jref, atol=1e-7, rtol=0)
        np.testing.assert_allclose(vk.get(), kref, atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
