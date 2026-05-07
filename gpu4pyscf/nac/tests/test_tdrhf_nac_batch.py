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
import pytest
from gpu4pyscf.lib.multi_gpu import num_devices

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


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet(self):
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

    def test_nac_grad_tda_singlet(self):
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

    def test_nac_tdhf_singlet(self):
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

    def test_nac_grad_tdhf_singlet(self):
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

    def test_nac_grad_tda_target_state_singlet(self):
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

    def test_nac_z_vector_reuse(self):
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


if __name__ == "__main__":
    print("Full Tests for batched TD-RHF nonadiabatic coupling vectors between excited states.")
    unittest.main()
