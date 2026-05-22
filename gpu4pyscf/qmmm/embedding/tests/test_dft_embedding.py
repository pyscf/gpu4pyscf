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
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding import embedding
from gpu4pyscf.qmmm.embedding.embeding_dft import SingleFragmentEmbedding


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.mol = gto.Mole()
        cls.mol.atom = '''
            C      -0.76091    -0.00000     0.00000
            C       0.76091    -0.00000     0.00000
            H      -1.16001     1.02029     0.00000
            H      -1.16001    -0.51014    -0.88357
            H      -1.16001    -0.51014     0.88357
            H       1.16001    -1.02029     0.00000
            H       1.16001     0.51014     0.88357
            H       1.16001     0.51014    -0.88357    
        '''
        cls.mol.basis = '6-31g'
        cls.mol.spin = 0
        cls.mol.charge = 0
        cls.mol.verbose = 0
        cls.mol.build()

        cls.fragments = [[0, 1], [2, 3]]

    @classmethod
    def tearDownClass(cls):
        del cls.mol
    
    def test_b3lyp_in_b3lyp(self):

        mf_outer = rks.RKS(self.mol, xc='B3LYP')
        mf_inner_template = rks.RKS(self.mol, xc='B3LYP')

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template, [0, 2, 3, 4])
        emb_obj.kernel()

        e_ref = mf_outer.kernel()

        assert np.abs(e_ref - emb_obj.e_tot) < 1e-8, f"Reference energy {e_ref} != Embedding energy {emb_obj.energy}"

    def test_b3lyp_in_pbe(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner_template = rks.RKS(self.mol, xc='B3LYP')

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template, [i for i in range(8)])
        emb_obj.kernel()

        e_ref = mf_inner_template.kernel()

        assert np.abs(e_ref - emb_obj.e_tot) < 1e-8, f"Reference energy {e_ref} != Embedding energy {emb_obj.energy}"

    def test_algebraic_properties(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, [0, 1, 2])
        emb_obj.kernel()

        S_ao = cp.asarray(mf_outer.get_ovlp())
        B = emb_obj.B[0]
        D_core = emb_obj.dm_core[0]

        # Check B^T * S * B == I (Orthonormality of embedding basis)
        ortho_check = B.T @ S_ao @ B
        identity = cp.eye(B.shape[1])
        max_ortho_err = float(cp.abs(ortho_check - identity).max())
        self.assertTrue(max_ortho_err < 1e-10, 
                        f"Basis B is not orthogonal, max error: {max_ortho_err}")

        # Check Spatial Isolation (Core DM projected onto the active space must be zero)
        core_overlap = B.T @ S_ao @ D_core @ S_ao @ B
        max_overlap_err = float(cp.abs(core_overlap).max())
        self.assertTrue(max_overlap_err < 1e-10, 
                        f"Core DM leaks into Active Space, max error: {max_overlap_err}")

    def test_electron_conservation(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='B3LYP')
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, [0, 1])
        emb_obj.kernel()
        
        S_ao = cp.asarray(mf_outer.get_ovlp())
        D_emb_high = cp.asarray(emb_obj.mf_inner[0].make_rdm1())
        D_core = emb_obj.dm_core[0]
        B = emb_obj.B[0]
        
        # Project local active density back to full AO basis
        D_emb_ao = B @ D_emb_high @ B.T # Identity S ignored
        D_total_ao = D_core + D_emb_ao
        
        n_elec_calc = float(cp.trace(D_total_ao @ S_ao))
        n_elec_exact = float(self.mol.nelectron)
        
        self.assertAlmostEqual(n_elec_calc, n_elec_exact, places=8, 
                               msg=f"Electron loss: {n_elec_calc} != {n_elec_exact}")

    def test_template_isolation_and_convergence(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner_template = rks.RKS(self.mol, xc='PBE')
        
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template, [0, 2, 3, 4], threshold=-1.0)
        emb_obj.kernel()
        
        mf_inner_template.kernel()
        
        # Assert the template is still clean and converges properly
        self.assertTrue(mf_inner_template.converged, 
                        "Template object was poisoned and failed to converge!")


if __name__ == '__main__':
    print("Full Tests for ONIOM-like DFT embedding.")
    unittest.main()
