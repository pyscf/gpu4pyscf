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
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.dmet import DMET 
from gpu4pyscf import dmet


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

        cls.mol2 = gto.Mole()
        cls.mol2.atom = '''
            C      -0.76091    -0.00000     0.00000
            C       0.76091    -0.00000     0.00000
            H      -1.16001     1.02029     0.00000
            H      -1.16001    -0.51014    -0.88357
            H      -1.16001    -0.51014     0.88357
            H       1.16001    -1.02029     0.00000
            H       1.16001     0.51014     0.88357
            H       1.16001     0.51014    -0.88357    
        '''
        cls.mol2.basis = '6-31g'
        cls.mol2.spin = 0
        cls.mol2.charge = 0
        cls.mol2.verbose = 0
        cls.mol2.build()

        cls.fragments2 = [[0, 2, 3, 4], [1, 5, 6, 7]]

        cls.mf_outer2 = gpu_hf.RHF(cls.mol2)
        cls.mf_outer2.conv_tol = 1e-12
        cls.mf_inner_template2 = gpu_hf.RHF(cls.mol2)
        cls.mf_inner_template2.conv_tol = 1e-12

    @classmethod
    def tearDownClass(cls):
        del cls.mol
        del cls.mf_outer
        del cls.mf_inner_template
        del cls.mol2
        del cls.mf_outer2
        del cls.mf_inner_template2

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

    def test_lowdin(self):
        ovlp = self.mf_outer.get_ovlp()
        X, _ = dmet.dmet.lowdin_orth(ovlp)
        X_ref = cp.array([[ 1.1214051976, -0.3278815514,  0.0611473762, -0.0095874461],
                          [-0.3278815514,  1.2643824327, -0.3597401082,  0.0611473762],
                          [ 0.0611473762, -0.3597401082,  1.2643824327, -0.3278815514],
                          [-0.0095874461,  0.0611473762, -0.3278815514,  1.1214051976]])
        assert np.abs(X - X_ref).max() < 1e-8, "Lowdin orthogonalization should yield a close-to-identity matrix."

    def test_schmidt(self):
        pass

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

        e_tot_ref = self.mf_outer.kernel()
        
        assert np.abs(e_tot - e_tot_ref) < 1e-8, "DMET energy should be close to the reference energy."

        dmet_solver2 = DMET(
            mf_outer=self.mf_outer2,
            mf_inner=self.mf_inner_template2,
            fragments=self.fragments2,
            threshold=1e-5,
            max_macro_iter=20,
            macro_tol=1e-3
        )

        e_tot = dmet_solver2.kernel()
        self.mf_outer2.mo_coeff = None
        e_tot_ref = self.mf_outer2.kernel()
        
        assert np.abs(e_tot - e_tot_ref) < 1e-8, "DMET energy should be close to the reference energy."


if __name__ == '__main__':
    print("Full Tests for DMET")
    unittest.main()