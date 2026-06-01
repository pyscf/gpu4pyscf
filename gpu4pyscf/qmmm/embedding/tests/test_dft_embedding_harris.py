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
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding.embedding_dft import SingleFragmentEmbedding
from gpu4pyscf.qmmm.embedding.embedding_dft_harris import HarrisRKS, SingleFragmentEmbedding_ML


def dummy_eval_density_func(mol, xc, grids, atomic_weights=None, grid_weights=None):
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.grids = grids
    mf.verbose = 0
    mf.kernel()
    
    dm = cp.asarray(mf.make_rdm1())
    
    # Calculate exact J and K matrices
    vj, vk = mf.get_jk(mol, dm)
    e_j = 0.5 * float(cp.sum(dm * vj))
    
    is_hybrid = mf._numint.libxc.is_hybrid_xc(xc)
    if is_hybrid:
        hyb = mf._numint.libxc.hybrid_coeff(xc, spin=mol.spin)
        vk = vk * hyb
        e_k = 0.5 * float(cp.sum(dm * vk))
    else:
        vk = None
        e_k = 0.0
        
    # Calculate exact Vxc and Exc
    _, e_xc, vxc = mf._numint.nr_rks(mol, grids, xc, dm)
    int_rho_vxc = float(cp.sum(dm * vxc))
    
    return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc


class TestMLEmbedding(unittest.TestCase):
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

        cls.methyl_fragment = [0, 2, 3, 4]
        cls.full_fragment = [i for i in range(cls.mol.natm)]

    @classmethod
    def tearDownClass(cls):
        del cls.mol

    def test_harris_rks_exactness(self):
        mf_ref = rks.RKS(self.mol, xc='PBE')
        mf_ref.verbose = 0
        e_ref = mf_ref.kernel()

        mf_harris = HarrisRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_harris.verbose = 0
        e_harris = mf_harris.kernel()

        self.assertAlmostEqual(e_ref, e_harris, places=8, 
                               msg=f"HarrisRKS energy {e_harris} differs from exact RKS {e_ref}")

    def test_full_system_pbe_in_pbe(self):
        mf_outer = HarrisRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        
        emb_obj = SingleFragmentEmbedding_ML(mf_outer, mf_inner, self.full_fragment, verbose=0)
        emb_obj.kernel()
        
        mf_outer.kernel()
        e_global = mf_outer.e_tot
        e_emb = emb_obj.e_tot
        
        self.assertAlmostEqual(e_global, e_emb, places=8, 
                               msg="Full-system PBE-in-PBE failed exact cancellation.")

    def test_equivalence_to_standard_embedding(self):

        mf_outer_std = rks.RKS(self.mol, xc='PBE')
        mf_inner_std = rks.RKS(self.mol, xc='B3LYP')
        emb_std = SingleFragmentEmbedding(mf_outer_std, mf_inner_std, self.methyl_fragment, verbose=0)
        e_std = emb_std.kernel()

        mf_outer_ml = HarrisRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_inner_ml = rks.RKS(self.mol, xc='B3LYP')
        emb_ml = SingleFragmentEmbedding_ML(mf_outer_ml, mf_inner_ml, self.methyl_fragment, verbose=0)
        e_ml = emb_ml.kernel()

        self.assertAlmostEqual(e_std, e_ml, places=8, 
                               msg=f"ML Embedding {e_ml} diverged from Standard Embedding {e_std}!")

    def test_harris_max_cycle_override(self):

        mf_harris = HarrisRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_harris.max_cycle = 100 
        mf_harris.verbose = 0
        
        mf_harris.kernel()
        
        self.assertEqual(mf_harris.max_cycle, 1, 
                         "HarrisRKS failed to override malicious max_cycle setting!")

if __name__ == '__main__':
    print("Full Tests for ML-Driven ONIOM-like Embedding...")
    unittest.main()

