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
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.scf import addons
from gpu4pyscf.dft.numint2c import NumInt2C
from gpu4pyscf.df.df_jk import _DFHF
from gpu4pyscf.lib.cupy_helper import tag_array


def setUpModule():
    global rhf_ref, rohf_ref
    h2 = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
    li = gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
    rhf_ref = h2.RHF().to_gpu().run()
    rohf_ref = li.ROHF().to_gpu().run()


class KnownValues(unittest.TestCase):
    def test_rhf_conversions(self):
        rhf = addons.convert_to_rhf(rhf_ref)
        uhf = addons.convert_to_uhf(rhf_ref)
        ghf = addons.convert_to_ghf(rhf_ref)
        self.assertIsInstance(rhf, scf.hf.RHF)
        self.assertIsInstance(uhf, scf.UHF)
        self.assertIsInstance(ghf, scf.GHF)
        self.assertIsNot(rhf, rhf_ref)
        cp.testing.assert_allclose(uhf.make_rdm1().sum(axis=0), rhf.make_rdm1())
        self.assertEqual(ghf.mo_coeff.shape, (4, 4))
        cp.testing.assert_array_equal(ghf.mo_coeff.orbspin, cp.array([0, 1, 0, 1]))

    def test_rohf_conversions(self):
        rohf = addons.convert_to_rhf(rohf_ref)
        uhf = addons.convert_to_uhf(rohf_ref)
        ghf = addons.convert_to_ghf(rohf_ref)
        self.assertIsInstance(rohf, scf.ROHF)
        self.assertIsInstance(uhf, scf.UHF)
        self.assertIsInstance(ghf, scf.GHF)
        cp.testing.assert_allclose(uhf.make_rdm1(), rohf.make_rdm1())
        self.assertEqual(float(ghf.mo_occ.sum()), 3.)

    def test_uhf_to_restricted(self):
        closed = addons.convert_to_rhf(rhf_ref.to_uhf())
        opened = addons.convert_to_rhf(rohf_ref.to_uhf())
        self.assertIs(type(closed), scf.hf.RHF)
        self.assertIs(type(opened), scf.ROHF)
        self.assertFalse(closed.converged)
        self.assertFalse(opened.converged)
        cp.testing.assert_allclose(closed.make_rdm1(), rhf_ref.make_rdm1())
        cp.testing.assert_allclose(opened.make_rdm1(), rohf_ref.make_rdm1())

    def test_rks_conversions(self):
        mf = dft.RKS(rhf_ref.mol)
        self.assertIsInstance(addons.convert_to_rhf(mf), dft.rks.RKS)
        self.assertIsInstance(addons.convert_to_uhf(mf), dft.UKS)
        result = addons.convert_to_ghf(mf)
        self.assertIsInstance(result, dft.GKS)
        self.assertIsInstance(result._numint, NumInt2C)
        self.assertIsNone(result.mo_coeff)

    def test_uks_conversions(self):
        mf = dft.UKS(rhf_ref.mol)
        self.assertIsInstance(addons.convert_to_rhf(mf), dft.rks.RKS)
        self.assertIsInstance(addons.convert_to_uhf(mf), dft.UKS)
        result = addons.convert_to_ghf(mf)
        self.assertIsInstance(result, dft.GKS)
        self.assertIsInstance(result._numint, NumInt2C)

    def test_density_fit_rhf_conversions(self):
        mf = rhf_ref.density_fit('weigend')
        rhf = addons.convert_to_rhf(mf)
        uhf = addons.convert_to_uhf(mf)
        ghf = addons.convert_to_ghf(mf)
        self.assertIsInstance(rhf, _DFHF)
        self.assertIsInstance(uhf, _DFHF)
        self.assertIsInstance(ghf, _DFHF)
        self.assertIs(rhf.with_df, mf.with_df)
        self.assertIs(uhf.with_df, mf.with_df)
        self.assertIs(ghf.with_df, mf.with_df)

    def test_density_fit_uhf_conversions(self):
        mf = rhf_ref.to_uhf().density_fit('weigend')
        rhf = addons.convert_to_rhf(mf)
        uhf = addons.convert_to_uhf(mf)
        ghf = addons.convert_to_ghf(mf)
        self.assertIsInstance(rhf, _DFHF)
        self.assertIsInstance(uhf, _DFHF)
        self.assertIsInstance(ghf, _DFHF)
        self.assertIs(rhf.with_df, mf.with_df)
        self.assertIs(uhf.with_df, mf.with_df)
        self.assertIs(ghf.with_df, mf.with_df)

    def test_density_fit_ghf_copy(self):
        mf = rhf_ref.to_ghf().density_fit('weigend')
        result = addons.convert_to_ghf(mf)
        self.assertIsNot(result, mf)
        self.assertIsInstance(result, _DFHF)
        self.assertIs(result.with_df, mf.with_df)

    def test_density_fit_plain_out(self):
        mf = rhf_ref.density_fit('weigend')
        out = scf.UHF(mf.mol)
        self.assertIs(addons.convert_to_uhf(mf, out=out), out)
        self.assertNotIsInstance(out, _DFHF)
        cp.testing.assert_allclose(out.make_rdm1().sum(axis=0), mf.make_rdm1())

    def test_remove_density_fit_ghf(self):
        mf = rhf_ref.to_uhf().density_fit('weigend')
        result = addons.convert_to_ghf(mf, out=scf.GHF(mf.mol), remove_df=True)
        self.assertNotIsInstance(result, _DFHF)
        self.assertFalse(hasattr(result, 'with_df'))
        self.assertEqual(float(result.mo_occ.sum()), 2.)

    def test_x2c_density_fit_conversion(self):
        mf = scf.RHF(rhf_ref.mol).x2c().density_fit('weigend')
        result = addons.convert_to_uhf(mf)
        self.assertIsInstance(result, scf.UHF)
        self.assertIsInstance(result, _DFHF)
        self.assertIs(result.with_x2c, mf.with_x2c)
        self.assertIs(result.with_df, mf.with_df)

    def test_x2c_uhf_to_ghf(self):
        mf = scf.UHF(rhf_ref.mol).x2c().density_fit('weigend')
        result = addons.convert_to_ghf(mf)
        self.assertIsInstance(result, scf.GHF)
        self.assertIs(result.with_x2c, mf.with_x2c)
        self.assertIs(result.with_df, mf.with_df)

    def test_x2c_newton_hessian_removed(self):
        mf = scf.RHF(rhf_ref.mol).x2c().newton().density_fit('weigend')
        result = addons.convert_to_uhf(mf)
        self.assertIsInstance(result, scf.UHF)
        self.assertIs(result.with_x2c, mf.with_x2c)
        self.assertFalse(hasattr(result, '_scf'))
        self.assertNotIsInstance(result, _DFHF)

    def test_uhf_newton_hessian_removed(self):
        mf = scf.UHF(rhf_ref.mol).newton().density_fit('weigend')
        result = addons.convert_to_ghf(mf)
        self.assertIsInstance(result, scf.GHF)
        self.assertFalse(hasattr(result, '_scf'))
        self.assertNotIsInstance(result, _DFHF)

    def test_uhf_df_newton_preserves_df(self):
        mf = scf.UHF(rhf_ref.mol).density_fit('weigend').newton()
        result = addons.convert_to_rhf(mf)
        self.assertIsInstance(result, scf.hf.RHF)
        self.assertIsInstance(result, _DFHF)
        self.assertIs(result.with_df, mf.with_df)
        self.assertFalse(hasattr(result, '_scf'))

    def test_density_fit_preserved(self):
        mf = scf.RHF(rhf_ref.mol).density_fit('weigend', only_dfj=True)
        result = mf.to_uks('pbe')
        self.assertIsInstance(result, _DFHF)
        self.assertIs(result.with_df, mf.with_df)
        self.assertTrue(result.only_dfj)

    def test_newton_removed(self):
        mf = scf.RHF(rhf_ref.mol).density_fit('weigend').newton()
        result = mf.to_uhf()
        self.assertFalse(hasattr(result, '_scf'))
        self.assertIs(result.with_df, mf.with_df)

    def test_remove_density_fit(self):
        mf = scf.RHF(rhf_ref.mol).density_fit('weigend')
        result = addons.convert_to_uhf(mf, remove_df=True)
        self.assertNotIsInstance(result, _DFHF)
        self.assertFalse(hasattr(result, 'with_df'))
        self.assertIsInstance(mf, _DFHF)

    def test_uhf_out(self):
        mf = rohf_ref
        out = scf.UHF(mf.mol)
        self.assertIs(addons.convert_to_uhf(mf, out=out), out)
        self.assertIsInstance(out.mo_coeff, cp.ndarray)
        cp.testing.assert_allclose(out.make_rdm1(), mf.make_rdm1())
        cp.testing.assert_array_equal(out.mo_occ.sum(axis=1), cp.array(mf.mol.nelec))

    def test_rhf_out(self):
        mf = rhf_ref.to_uhf()
        out = scf.RHF(mf.mol)
        self.assertIs(addons.convert_to_rhf(mf, out=out), out)
        cp.testing.assert_allclose(out.mo_coeff, mf.mo_coeff[0])
        cp.testing.assert_allclose(out.mo_occ, mf.mo_occ.sum(axis=0))
        self.assertFalse(out.converged)

    def test_ghf_out(self):
        mf = rohf_ref
        out = scf.GHF(mf.mol)
        self.assertIs(addons.convert_to_ghf(mf, out=out), out)
        nao = mf.mol.nao
        dm = out.make_rdm1()
        cp.testing.assert_allclose(dm[:nao, :nao], mf.make_rdm1()[0])
        cp.testing.assert_allclose(dm[nao:, nao:], mf.make_rdm1()[1])

    def test_fractional_rhf_to_uhf(self):
        mf = rhf_ref.copy()
        mf.mo_occ = cp.array([1.6, .4])
        result = mf.to_uhf()
        cp.testing.assert_allclose(result.mo_occ, cp.array([[.8, .2], [.8, .2]]))
        cp.testing.assert_allclose(result.make_rdm1().sum(axis=0), mf.make_rdm1(),
                                   rtol=0, atol=1e-12)

    def test_fractional_rks_to_gks(self):
        mf = rhf_ref.to_rks('pbe')
        mf.mo_occ = cp.array([1., 1.])
        result = mf.to_gks()
        nao = mf.mol.nao
        dm = result.make_rdm1()
        cp.testing.assert_allclose(dm[:nao, :nao], mf.make_rdm1() * .5,
                                   rtol=0, atol=1e-12)
        cp.testing.assert_allclose(dm[nao:, nao:], mf.make_rdm1() * .5,
                                   rtol=0, atol=1e-12)
        cp.testing.assert_array_equal(dm[:nao, nao:], 0)

    def test_closed_shell_roks_to_rks(self):
        mf = dft.ROKS(rhf_ref.mol)
        result = mf.to_rks()
        self.assertIs(type(result), dft.rks.RKS)
        self.assertIsNot(result, mf)

    def test_uks_to_gks_omega(self):
        mf = dft.UKS(rhf_ref.mol, xc='wb97x')
        mf.omega = .23
        out = dft.GKS(mf.mol)
        result = addons.convert_to_ghf(mf, out=out)
        self.assertIs(result, out)
        self.assertIsInstance(result._numint, NumInt2C)
        self.assertEqual(result.omega, .23)

    def test_complex_orbitals_and_spin_order(self):
        ref = rohf_ref.to_uhf()
        ref.mo_coeff = ref.mo_coeff.astype(complex)
        ref.mo_coeff[1] *= 1j
        # Exercise interleaved spins, near degeneracies, and virtual ordering.
        ref.mo_energy = cp.array([[-2., -.5, .2, .4, .6],
                                  [-2.00000001, -.4, .1, .3, .5]])
        mf = ref
        result = mf.to_ghf()
        cpu_result = ref.to_cpu().to_ghf()
        np.testing.assert_allclose(result.mo_coeff.get(), cpu_result.mo_coeff)
        np.testing.assert_allclose(result.mo_energy.get(), cpu_result.mo_energy)
        np.testing.assert_array_equal(result.mo_coeff.orbspin.get(),
                                      cpu_result.mo_coeff.orbspin)
        nao = ref.mol.nao
        dm = result.make_rdm1().get()
        dm_ref = ref.make_rdm1().get()
        np.testing.assert_allclose(dm[:nao, :nao], dm_ref[0], atol=1e-12)
        np.testing.assert_allclose(dm[nao:, nao:], dm_ref[1], atol=1e-12)
        np.testing.assert_allclose(dm[:nao, nao:], 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
