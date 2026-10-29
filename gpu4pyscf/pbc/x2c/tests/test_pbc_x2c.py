# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
import tempfile
import numpy as np
import cupy as cp
from pyscf.pbc import gto as pbcgto
from pyscf.x2c import x2c as x2c_cpu
import pyscf
from pyscf import lib
from pyscf.pbc.df import df as df_cpu
from pyscf.pbc.x2c import sfx2c1e as sfx2c1e_cpu
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.pbc.x2c import x2c1e

class KnownValues(unittest.TestCase):
    def test_get_pvdotp(self):
        L = 5.
        n = 11
        np.random.seed(9)
        cell = pyscf.M(
            a = np.eye(3) * L + np.random.rand(3,3) - .5,
            mesh = [n] * 3,
            atom = '''C    3.    2.       3.
                      C    1.    1.       1.''',
            basis = 'ccpvdz',
            precision = 1e-8,
            verbose = 0,
        )

        cell2 = cell.copy()
        cell2.precision = 1e-10
        cell2.build(0, 0)

        kpts4 = cell.make_kpts([4,1,1])
        sorted_cell = SortedGTO.from_cell(cell)

        ref = GDF(cell, kpts4).get_nuc(kpts=kpts4)
        v1 = x2c1e._get_pnucp(sorted_cell, kpts=kpts4, intor='nuc')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1 - ref).max() < 1e-8

        ref = sfx2c1e_cpu.get_pnucp(df_cpu.GDF(cell2))
        v1 = x2c1e._get_pnucp(sorted_cell, intor='pnucp')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1.get() - ref).max() < 1e-7

        ref = sfx2c1e_cpu.get_pnucp(df_cpu.GDF(cell2), kpts4)
        v1 = x2c1e._get_pnucp(sorted_cell, kpts=kpts4, intor='pnucp')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1.get() - ref).max() < 1e-7

    def test_get_hcore_high_cost(self):
        cell = pyscf.M(
            atom='C 0 0 0; C 1.685 1.685 1.685',
            a='''
            0.00, 3.37, 3.37
            3.37, 0.00, 3.37
            3.37, 3.37, 0.00''',
            basis=('ccpvdz', [[3, [.7, 1]]])
        )
        with lib.temporary_env(lib.param, LIGHT_SPEED=10.):
            cell1 = cell.copy()
            cell1.precision = 1e-11
            cell1.build(0, 0)
            kpts = cell.make_kpts([3,2,1])
            mf = cell1.KRHF(kpts=kpts).x2c1e()
            ref = mf.get_hcore()

            mf = cell.KRHF(kpts=kpts).to_gpu().x2c1e()
            dat = mf.get_hcore()
            assert abs(ref - dat.get()).max() < 1e-8

    @unittest.skipUnless(hasattr(sfx2c1e_cpu.SFX2C1E_SCF, 'to_gpu'),
                         'CPU periodic scalar-X2C conversion is unavailable')
    def test_to_cpu(self):
        cell = pyscf.M(
            unit= 'B',
            a = np.eye(3)*4,
            # The original 11^3 FFT grid leaves a 1e-6 exchange-energy error.
            # 19^3 and 31^3 agree below 1e-12 for this small all-electron cell.
            mesh = [19]*3,
            atom = 'H 0 0 0; H 0 0 1.8',
            basis='sto3g', precision=1e-10)
        with lib.temporary_env(lib.param, LIGHT_SPEED = 2):
            mf = cell.RHF().sfx2c1e()
            mf.conv_tol = 1e-11
            ref = mf.kernel()
            self.assertTrue(mf.converged)

            mf = mf.to_gpu()
            assert isinstance(mf, x2c1e.SFX2C1E_SCF)
            assert isinstance(mf.with_x2c, x2c1e.SpinFreeX2CHelper)
            mf.run()
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(mf.e_tot, ref, 8)

            gpu_mf = mf
            gpu_integration = mf._numint
            self.assertIsNotNone(gpu_integration)
            expected_hcore = mf.get_hcore().get()
            mf = mf.to_cpu()
            self.assertIs(gpu_mf._numint, gpu_integration)
            self.assertIsNone(getattr(mf, '_numint', None))
            np.testing.assert_allclose(mf.get_hcore(), expected_hcore, atol=1e-8, rtol=0)
            assert isinstance(mf, sfx2c1e_cpu.SFX2C1E_SCF)
            assert isinstance(mf.with_x2c, sfx2c1e_cpu.SpinFreeX2CHelper)

            mf = cell.KRHF(kpts=cell.make_kpts([3,1,1])).sfx2c1e()
            mf.conv_tol = 1e-11
            ref = mf.kernel()
            self.assertTrue(mf.converged)

            mf = mf.to_gpu().run()
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(mf.e_tot, ref, 8)


class SmallScalarX2C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = pyscf.M(atom='H .2 .3 .4; H 1.6 .3 .4',
                           a=np.eye(3)*7, unit='Bohr', basis='sto-3g',
                           mesh=[13]*3, precision=1e-10, verbose=0)
        # Two commensurate points; the second gives nonzero imaginary AO terms.
        cls.kpts = cls.cell.make_kpts([3, 1, 1])[:2]

    def setUp(self):
        self.light_speed = lib.temporary_env(lib.param, LIGHT_SPEED=10.)
        self.light_speed.__enter__()
        self.addCleanup(self.light_speed.__exit__, None, None, None)

    def test_get_hcore(self):
        for kpts in (np.zeros(3), self.kpts):
            with self.subTest(kpts=kpts.shape):
                if kpts.ndim == 1:
                    cpu = self.cell.RHF().sfx2c1e()
                    gpu = self.cell.RHF().to_gpu().sfx2c1e()
                else:
                    cpu = self.cell.KRHF(kpts=kpts).sfx2c1e()
                    gpu = self.cell.KRHF(kpts=kpts).to_gpu().sfx2c1e()
                expected = cpu.get_hcore()
                actual = gpu.get_hcore().get()
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                np.testing.assert_allclose(actual, actual.swapaxes(-1, -2).conj(),
                                           atol=1e-12, rtol=0)
                if kpts.ndim == 2:
                    self.assertGreater(abs(actual.imag).max(), 1e-3)

    def test_explicit_single_kpoint_and_scalar_spin_blocks(self):
        cpu = self.cell.RHF().sfx2c1e()
        gpu = self.cell.RHF().to_gpu().sfx2c1e()
        kpt = self.kpts[1]
        np.testing.assert_allclose(gpu.get_hcore(kpt=kpt).get(),
                                   cpu.get_hcore(kpt=kpt), atol=1e-8, rtol=0)
        # Use the public scalar factory, not the unsupported SOC factory.
        for kpts in (None, self.kpts):
            with self.subTest(kpts=kpts):
                base = self.cell.GHF() if kpts is None else self.cell.KGHF(kpts=kpts)
                cpu = sfx2c1e_cpu.sfx2c1e(base)
                gpu = x2c1e.sfx2c1e(base.to_gpu())
                actual = cp.asarray(gpu.get_hcore()).get()
                reference = np.asarray(cpu.get_hcore())
                np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=0)
                n = self.cell.nao
                np.testing.assert_array_equal(actual[..., :n, n:], 0.)
                np.testing.assert_array_equal(actual[..., n:, :n], 0.)
                np.testing.assert_array_equal(actual[..., :n, :n], actual[..., n:, n:])

    def test_helper_default_gamma_and_single_batch_shapes(self):
        helper = x2c1e.SpinFreeX2CHelper(self.cell)
        for method in (helper.get_hcore, helper.get_xmat):
            with self.subTest(method=method.__name__):
                default = method()
                single = method(kpts=np.zeros(3))
                batch = method(kpts=np.zeros((1, 3)))
                self.assertEqual(default.ndim, 2)
                self.assertEqual(default.shape, single.shape)
                self.assertEqual(batch.shape, (1, *single.shape))
                np.testing.assert_allclose(default.get(), single.get(), atol=1e-12, rtol=0)
                np.testing.assert_allclose(batch[0].get(), single.get(), atol=1e-12, rtol=0)

    def test_xmat_decoupling_and_hcore_reconstruction(self):
        helper = x2c1e.SpinFreeX2CHelper(self.cell)
        xcell = helper.get_xmol()
        # CPU integrals in the same sorted primitive basis isolate the X2C
        # decoupling/recontraction from differences in decontraction ordering.
        primitive = xcell.view(pbcgto.Cell)
        c = cp.asnumpy(xcell.ctr_coeff)
        kpts = self.kpts
        t = primitive.pbc_intor('int1e_kin', hermi=1, kpts=kpts)
        overlap = primitive.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        fitting = df_cpu.GDF(primitive)
        v = fitting.get_nuc(kpts)
        w = sfx2c1e_cpu.get_pnucp(fitting, kpts)
        x = helper.get_xmat(kpts=kpts).get()
        hcore = helper.get_hcore(kpts=kpts).get()
        speed = lib.param.LIGHT_SPEED
        for k in range(len(kpts)):
            # Lower block of H[I;X] = M[I;X] E, where E=S^-1(V+TX).
            effective = np.linalg.solve(overlap[k], v[k] + t[k] @ x[k])
            residual = (t[k] + (w[k]/(4*speed**2) - t[k]) @ x[k] -
                        t[k] @ x[k] @ effective/(2*speed**2))
            np.testing.assert_allclose(residual, 0., atol=1e-8, rtol=0)
            primitive_h = x2c_cpu._get_hcore_fw(t[k], v[k], w[k], overlap[k], x[k], speed)
            np.testing.assert_allclose(c.T @ primitive_h @ c, hcore[k], atol=1e-8, rtol=0)
        np.testing.assert_allclose(helper.get_xmat(kpts=kpts[1]).get(), x[1], atol=1e-12, rtol=0)

    def test_standalone_helper_cpu_roundtrip(self):
        cpu = sfx2c1e_cpu.SpinFreeX2CHelper(self.cell)
        cpu.approx = '1E'
        gpu = cpu.to_gpu()
        self.assertIsInstance(gpu, x2c1e.SpinFreeX2CHelper)
        self.assertIs(gpu.cell, gpu.mol)
        np.testing.assert_array_equal(gpu.cell.atom_coords(), self.cell.atom_coords())
        returned = gpu.to_cpu()
        self.assertIsInstance(returned, sfx2c1e_cpu.SpinFreeX2CHelper)
        self.assertIs(returned.cell, returned.mol)
        np.testing.assert_array_equal(returned.cell.atom_coords(), self.cell.atom_coords())
        self.assertEqual(returned.approx, '1E')
        np.testing.assert_allclose(gpu.get_hcore().get(), cpu.get_hcore(), atol=1e-8, rtol=0)
        np.testing.assert_allclose(returned.get_hcore(), cpu.get_hcore(), atol=1e-12, rtol=0)

    def test_orbital_pari_cart2sph(self):
        cell = pyscf.M(atom='He .2 .3 .4', a=np.eye(3)*7, unit='Bohr',
                       basis=[[0, [.8, 1.]], [1, [.7, 1.]], [2, [.6, 1.]]],
                       precision=1e-10, verbose=0)
        sorted_cell = SortedGTO.from_cell(cell)
        primitive = sorted_cell.view(pbcgto.Cell)
        kpts = cell.make_kpts([3, 1, 1])[:2]
        cart = primitive.pbc_intor('int1e_kin_cart', hermi=1, kpts=kpts)
        c = primitive.cart2sph_coeff()
        expected = np.array([c.T @ h @ c for h in cart])
        actual = x2c1e._orbital_pair_cart2sph(sorted_cell, cp.asarray(cart))
        listed = x2c1e._orbital_pair_cart2sph(sorted_cell, [cp.asarray(cart)])[0]
        np.testing.assert_allclose(actual.get(), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(listed.get(), expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    print("Full Tests for PBC X2C")
    unittest.main()
