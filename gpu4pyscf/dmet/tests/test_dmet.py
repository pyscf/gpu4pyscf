# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.dmet import DMET 


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.mol = gto.Mole()
        cls.mol.atom = '''
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            H 0.0 0.0 2.0
            H 0.0 0.0 3.0
        '''
        cls.mol.basis = 'sto-3g'
        cls.mol.spin = 0
        cls.mol.charge = 0
        cls.mol.verbose = 0
        cls.mol.build()

        cls.fragments = [[0, 1], [2, 3]]

        cls.mf_outer = gpu_hf.RHF(cls.mol)
        cls.mf_inner_template = gpu_hf.RHF(cls.mol)

    @classmethod
    def tearDownClass(cls):
        del cls.mol
        del cls.mf_outer
        del cls.mf_inner_template
        cp.get_default_memory_pool().free_all_blocks()

    def test_dmet_initialization(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-5
        )

        nao = self.mol.nao_nr()
        
        self.assertEqual(dmet_solver.nfrags, 2, "Number of fragments should be 2.")
        self.assertEqual(len(dmet_solver.frag_idx), 2, "Fragment indices list should have length 2.")
        
        self.assertEqual(dmet_solver.u_oao.shape, (nao, nao), "Correlation potential u_oao should be of shape (nao, nao).")
        self.assertTrue(isinstance(dmet_solver.u_oao, cp.ndarray), "Correlation potential should be a CuPy array.")

    def test_dmet_execution_and_convergence(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-5,
            max_macro_iter=20,
            macro_tol=1e-3
        )

        e_tot = dmet_solver.kernel()

        self.assertIsNotNone(e_tot, "DMET kernel should return a valid energy value, not None.")
        self.assertIsInstance(e_tot, float, "The returned total energy must be a float.")

        self.assertLess(e_tot, 0.0, "Total energy of H4 molecule should be negative.")

        self.assertIsNotNone(dmet_solver.bath_orb[0], "Bath orbitals for fragment 0 should be generated.")
        self.assertIsNotNone(dmet_solver.h_emb[0], "Embedded Hamiltonian for fragment 0 should be generated.")
        
        self.assertTrue(isinstance(dmet_solver.dm_core[0], cp.ndarray), "Core density matrix should be a CuPy array.")


if __name__ == '__main__':
    print("Full Tests for DMET")
    unittest.main()