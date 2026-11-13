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
from unittest import mock
from packaging.version import Version
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
from gpu4pyscf.nac import tdrhf_grad_nacv
import gpu4pyscf
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696
"""

bas0 = "def2tzvp"
CPU_GRADIENT_REFERENCE = Version(pyscf.__version__) >= Version('2.14')

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1,3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1,3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(2,3)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(2,3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

    def test_nac_grad_tda_singlet_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.results[(0,1)]['de']) - np.abs(nac1.de)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(0,1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-4

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-5

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()
        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-6

    def test_nac_tdhf_singlet_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDHF().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(1,3)
        nac1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.results[(1,3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-4
        assert abs(np.abs(nac_test.results[(1,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6

        nac1 = td.nac_method()
        nac1.states=(2,3)
        nac1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.results[(2,3)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2,3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(2,3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-4

    def test_nac_grad_tdhf_singlet_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDHF().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()
        # low accuracy due to z-vector calculation
        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-5

    def test_nac_grad_tda_target_state_singlet_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.target_state = 1
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()

        # low accuracy due to z-vector solver
        assert abs(np.abs(nac_test.results[(0,1)]['de']) - np.abs(nac1.de)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(0,1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-5
        assert abs(np.abs(nac_test.results[(0,1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-4

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

    def test_nac_z_vector_reuse_high_cost(self):
        mf = scf.RHF(mol).to_gpu()
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


class NACGradients_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_mol = pyscf.M(atom=atom, basis='6-31g', verbose=0)
        cls.mf = scf.RHF(cls.small_mol).to_gpu().set(conv_tol=1e-12).run()
        cls.td = cls.mf.TDA().set(nstates=2, conv_tol=1e-10, lindep=1e-20).run()
        assert cls.mf.converged and all(cls.td.converged)
        cls.cpu = None
        if CPU_GRADIENT_REFERENCE:
            cls.cpu = scf.RHF(cls.small_mol).set(conv_tol=1e-12).run()
            assert cls.cpu.converged

    @classmethod
    def tearDownClass(cls):
        del cls.td, cls.mf, cls.cpu, cls.small_mol

    def test_ground_target_and_cpu_ground_gradient(self):
        td = self.td
        batch = td.nac_gradient_method().set(cphf_conv_tol=1e-10)
        results = batch.kernel(states=(2, 1), target_state=0,
                               grad_state=0)
        self.assertEqual(set(results), {(0, 1), (0, 2)})
        if CPU_GRADIENT_REFERENCE:
            reference = self.cpu.nuc_grad_method().kernel()
            np.testing.assert_allclose(batch.grad_result, reference, atol=1e-7, rtol=0)
        for (_, state), values in results.items():
            for value in values.values():
                self.assertEqual(value.shape, (td.mol.natm, 3))
                self.assertTrue(np.isfinite(value).all())
            gap = float(td.e[state - 1])
            np.testing.assert_allclose(values['de_scaled'] * gap, values['de'],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(values['de_etf_scaled'] * gap, values['de_etf'],
                                       atol=1e-12, rtol=0)

    def test_full_td_target_gradient(self):
        td = self.mf.TDHF().set(nstates=2, conv_tol=1e-10, lindep=1e-20).run()
        self.assertTrue(all(td.converged))
        self.assertGreater(sum(float(cp.linalg.norm(y)) for _, y in td.xy), 1e-8)
        batch = td.nac_gradient_method().set(cphf_conv_tol=1e-10)
        result = batch.kernel(states=(0, 1, 2), target_state=1, grad_state=1)
        self.assertEqual(set(result), {(0, 1), (1, 2)})
        for pair, values in result.items():
            direct = td.nac_method().set(cphf_conv_tol=1e-10).kernel(states=pair)
            for name, expected in zip(('de', 'de_scaled', 'de_etf', 'de_etf_scaled'), direct):
                np.testing.assert_allclose(values[name], expected, atol=1e-6, rtol=0)
        if CPU_GRADIENT_REFERENCE:
            cpu_td = self.cpu.TDHF().set(nstates=2, conv_tol=1e-10).run()
            self.assertTrue(all(cpu_td.converged))
            np.testing.assert_allclose(cpu_td.e, cp.asnumpy(td.e), atol=1e-8, rtol=0)
            reference = cpu_td.nuc_grad_method().kernel(state=1)
            np.testing.assert_allclose(batch.grad_result, reference, atol=1e-6, rtol=0)

    def test_z_vector_reuse(self):
        batch = self.td.nac_gradient_method().set(cphf_conv_tol=1e-10)
        with mock.patch.object(tdrhf_grad_nacv, '_solve_zvector',
                               wraps=tdrhf_grad_nacv._solve_zvector) as solve:
            first = batch.kernel(states=(0, 1, 2), grad_state=1)
            repeated = batch.kernel()
            # Retain the excited-only, no-gradient batch path from the costly tests.
            batch.grad_state = None
            changed = batch.kernel(states=(1, 2))
        self.assertEqual(solve.call_count, 3)
        self.assertIsNone(solve.call_args_list[0].kwargs['z0'])
        self.assertIsNotNone(solve.call_args_list[1].kwargs['z0'])
        self.assertIsNone(solve.call_args_list[2].kwargs['z0'])
        self.assertEqual(set(changed), {(1, 2)})
        for pair, values in first.items():
            for name, value in values.items():
                np.testing.assert_allclose(repeated[pair][name], value, atol=1e-5, rtol=0)
        direct = self.td.nac_method().set(cphf_conv_tol=1e-10).kernel(states=(1, 2))
        for name, reference in zip(('de', 'de_scaled', 'de_etf', 'de_etf_scaled'), direct):
            np.testing.assert_allclose(changed[(1, 2)][name], reference, atol=1e-6, rtol=0)

    @unittest.skipUnless(CPU_GRADIENT_REFERENCE, 'CPU gradient comparison requires PySCF >= 2.14')
    def test_cpu_excited_gradient(self):
        batch = self.td.nac_gradient_method().set(cphf_conv_tol=1e-10)
        batch.kernel(states=(0, 1, 2), grad_state=1)
        cpu_td = self.cpu.TDA().set(nstates=2, conv_tol=1e-10).run()
        self.assertTrue(all(cpu_td.converged))
        np.testing.assert_allclose(cpu_td.e, cp.asnumpy(self.td.e), atol=1e-8, rtol=0)
        reference = cpu_td.nuc_grad_method().kernel(state=1)
        np.testing.assert_allclose(batch.grad_result, reference, atol=1e-6, rtol=0)


if __name__ == "__main__":
    print("Full Tests for batched TD-RHF nonadiabatic coupling vectors between excited states.")
    unittest.main()
