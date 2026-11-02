# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf

atom = """
O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696
"""

bas0 = "def2tzvp"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class NACGradients_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Three roots suffice for ground/excited and excited/excited batches.
        cls.small_mol = pyscf.M(atom=atom, basis='6-31g', verbose=0)
        cls._td_cache = {}

    @classmethod
    def tearDownClass(cls):
        cls._td_cache.clear()
        del cls.small_mol

    def _get_td(self, xc='b3lyp', full_td=False):
        key = (xc, full_td)
        if key not in self._td_cache:
            mf = dft.RKS(self.small_mol, xc=xc).to_gpu()
            mf.conv_tol = 1e-12
            mf.kernel()
            self.assertTrue(mf.converged)
            td = mf.TDDFT() if full_td else mf.TDA()
            td.nstates = 3
            td.conv_tol = 1e-10
            if full_td:
                # The cutoff acts on squared trial-vector norms; retain the
                # small corrections needed for the requested residual accuracy.
                td.lindep = td.conv_tol**2
            td.kernel()
            self.assertTrue(np.all(cp.asnumpy(td.converged)))
            energies = cp.asnumpy(td.e)
            self.assertEqual(energies.shape, (3,))
            self.assertTrue(np.isfinite(energies).all())
            # Scaled couplings divide by energy gaps; avoid near-degenerate roots.
            self.assertGreater(np.diff(np.r_[0., energies]).min(), 1e-5)
            if full_td:
                self.assertGreater(sum(float(cp.linalg.norm(y)) for _, y in td.xy), 1e-8)
            self._td_cache[key] = td
        return self._td_cache[key]

    def _new_batch(self, td, target_state=None):
        batch = td.nac_gradient_method()
        batch.states = (0, 1, 2)
        batch.grad_state = 1
        batch.target_state = target_state
        batch.cphf_conv_tol = 1e-10
        return batch

    def _assert_pair(self, td, result, pair):
        direct = td.nac_method()
        direct.states = pair
        direct.cphf_conv_tol = 1e-10
        direct.kernel()
        self.assertGreater(np.linalg.norm(cp.asnumpy(direct.de)), 1e-6)
        for name in ('de', 'de_scaled', 'de_etf', 'de_etf_scaled'):
            actual = cp.asnumpy(result[name])
            expected = cp.asnumpy(getattr(direct, name))
            self.assertEqual(actual.shape, (self.small_mol.natm, 3))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0,
                                       err_msg=f'{pair}: {name}')

    def _check_batch(self, td, target_state=None):
        batch = self._new_batch(td, target_state)
        batch.kernel()
        pairs = {(0, 1), (1, 2)}
        if target_state is None:
            pairs.add((0, 2))
        self.assertEqual(set(batch.results), pairs)
        for pair in ((0, 1), (1, 2)):
            self._assert_pair(td, batch.results[pair], pair)
        gradient = td.nuc_grad_method()
        gradient.state = 1
        gradient.cphf_conv_tol = 1e-10
        gradient.kernel()
        np.testing.assert_allclose(cp.asnumpy(batch.grad_result),
                                   cp.asnumpy(gradient.de), atol=1e-6, rtol=0)

    def test_batch_svwn(self):
        self._check_batch(self._get_td('svwn'))

    def test_batch_camb3lyp(self):
        self._check_batch(self._get_td('camb3lyp'))

    def test_batch_tpss(self):
        self._check_batch(self._get_td('tpss'))

    def test_batch_tddft_b3lyp(self):
        self._check_batch(self._get_td('b3lyp', full_td=True))

    def test_target_state(self):
        self._check_batch(self._get_td(), target_state=1)


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet_svwn(self):
        mf = dft.RKS(mol, xc='svwn').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

    def test_nac_grad_tda_singlet_svwn(self):
        mf = dft.RKS(mol, xc='svwn').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_tda_singlet_b3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

    def test_nac_grad_tda_singlet_b3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_tda_singlet_camb3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='camb3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

    def test_nac_grad_tda_singlet_camb3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='camb3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_tda_singlet_tpss(self):
        mf = dft.RKS(mol, xc='tpss').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

    def test_nac_grad_tda_singlet_tpss(self):
        mf = dft.RKS(mol, xc='tpss').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_tddft_singlet_b3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

    def test_nac_grad_tddft_singlet_b3lyp_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_grad_tda_singlet_b3lyp_target_state_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.target_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(0,1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0,1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_grad_tda_singlet_b3lyp_target_state_ris_zvector_solver_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.target_state = 1
        nac_test.ris_zvector_solver = True
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.ris_zvector_solver = True
        nac1.kernel()
        # low accuracy due to z-vector solver.
        assert abs(np.abs(nac_test.results[(0,1)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0,1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.ris_zvector_solver = True
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_z_vector_reuse_high_cost(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states = (0, 1, 2)
        nac_test.grad_state = 1
        
        # Run 1: Compute from scratch, should cache the z-vector
        nac_test.kernel()
        de_run1_01 = cp.asnumpy(nac_test.results[(0,1)]['de'])
        de_run1_12 = cp.asnumpy(nac_test.results[(1,2)]['de'])
        
        # Verify cache attributes are populated
        assert getattr(nac_test, '_z_prev', None) is not None
        assert getattr(nac_test, '_z_tasks', None) is not None

        # Run 2: Same states, should reuse z-vector
        nac_test.kernel()
        de_run2_01 = cp.asnumpy(nac_test.results[(0,1)]['de'])
        de_run2_12 = cp.asnumpy(nac_test.results[(1,2)]['de'])

        # Results should be exactly identical since the system hasn't changed
        assert abs(de_run1_01 - de_run2_01).max() < 1e-5
        assert abs(de_run1_12 - de_run2_12).max() < 1e-5

        # Verify against standard unbatched method
        nac_std = td.nac_method()
        nac_std.states = (0, 1)
        nac_std.kernel()
        assert abs(np.abs(de_run2_01) - np.abs(nac_std.de)).max() < 1e-5

        # Run 3: Change states to invalidate cache
        nac_test.states = (1, 2, 3)
        nac_test.kernel()
        de_run3_12 = cp.asnumpy(nac_test.results[(1,2)]['de'])

        # Verify that changing states still gives the correct result
        nac_std = td.nac_method()
        nac_std.states = (1, 2)
        nac_std.kernel()
        assert abs(np.abs(de_run3_12) - np.abs(nac_std.de)).max() < 1e-6


if __name__ == "__main__":
    print("Full Tests for batched TD-RKS nonadiabatic coupling vectors between excited states.")
    unittest.main()
