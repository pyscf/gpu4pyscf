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
from gpu4pyscf.qmmm.embedding.embedding_dft import SingleFragmentEmbedding


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

    def test_hexane_core_isolation_and_exactness(self):
        mol = gto.Mole()
        mol.atom = '''
            C   1.4522500000  -2.8230000000   0.0000000000
            C   1.4522500000  -1.2830000000   0.0000000000
            C   0.0002500000  -0.7700000000   0.0000000000
            C   0.0002500000   0.7700000000   0.0000000000
            C  -1.4517500000   1.2830000000   0.0000000000
            C  -1.4517500000   2.8230000000   0.0000000000
            H   2.4792500000  -3.1870000000   0.0000000000
            H   0.9382500000  -3.1870000000   0.8900000000
            H   0.9382500000  -3.1870000000  -0.8900000000
            H   1.9652500000  -0.9200000000   0.8900000000
            H   1.9652500000  -0.9200000000  -0.8900000000
            H  -0.5137500000  -1.1330000000  -0.8900000000
            H  -0.5137500000  -1.1330000000   0.8900000000
            H   0.5132500000   1.1330000000   0.8900000000
            H   0.5132500000   1.1330000000  -0.8900000000
            H  -1.9657500000   0.9200000000  -0.8900000000
            H  -1.9657500000   0.9200000000   0.8900000000
            H  -2.4797500000   3.1870000000   0.0000000000
            H  -0.9377500000   3.1870000000   0.8900000000
            H  -0.9377500000   3.1870000000  -0.8900000000
        '''
        mol.basis = 'sto3g'
        mol.spin = 0
        mol.verbose = 0
        mol.build()

        mf_outer = rks.RKS(mol, xc='PBE')
        mf_inner = rks.RKS(mol, xc='PBE')
        
        methyl_fragment = [0, 6, 7, 8]
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, methyl_fragment, threshold=1e-5)
        emb_obj.kernel()
        
        mf_outer.kernel()
        e_global = mf_outer.e_tot
        e_embedded = emb_obj.e_tot
        self.assertTrue(np.abs(e_global - e_embedded) < 1e-6, 
                        f"PBE-in-PBE Exactness failed! Error: {np.abs(e_global - e_embedded)}")
        
        dm_core_sum = float(cp.sum(emb_obj.dm_core[0]))
        self.assertTrue(dm_core_sum > 1.0, 
                        "Hexane test did not generate a non-trivial Core DM. SVD truncation might be failing.")

    def test_pure_dft_vk_bypass(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, self.fragments[0])
        try:
            emb_obj.kernel()
        except AttributeError as e:
            self.fail(f"Embedding failed for Pure DFT due to missing vk attribute: {e}")
            
        self.assertTrue(emb_obj.e_tot is not None, "Pure DFT embedding failed to return an energy.")


if __name__ == '__main__':
    print("Full Tests for ONIOM-like DFT embedding.")
    unittest.main()
