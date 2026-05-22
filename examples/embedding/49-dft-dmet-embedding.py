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

"""
Example 49: Single-Fragment Delta-Method DFT-in-DFT Embedding.

This script demonstrates how to embed a high-level hybrid DFT functional (B3LYP)
into a localized region of a low-level GGA DFT environment (PBE) using a 
highly-optimized projection basis without macroscopic iterations.
"""

from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding.embeding_dft import SingleFragmentEmbedding

def run_dft_embedding_example():
    # 1. Define the system (Ethane molecule with 6-31G basis)
    mol = gto.Mole()
    mol.atom = '''
        C      -0.76091    -0.00000     0.00000
        C       0.76091    -0.00000     0.00000
        H      -1.16001     1.02029     0.00000
        H      -1.16001    -0.51014    -0.88357
        H      -1.16001    -0.51014     0.88357
        H       1.16001    -1.02029     0.00000
        H       1.16001     0.51014     0.88357
        H       1.16001     0.51014    -0.88357    
    '''
    mol.basis = '6-31g'
    mol.verbose = 4  # Enable to monitor localized cluster basis dimensions and logs
    mol.build()

    print("--- Step 1: Prepare Environment (PBE) and Active Region (B3LYP) Solvers ---")
    # Low-level full system solver (Environment description)
    mf_outer = rks.RKS(mol, xc='PBE')
    mf_outer.conv_tol = 1e-10
    
    # High-level solver template (Active cluster description)
    mf_inner_template = rks.RKS(mol, xc='B3LYP')
    mf_inner_template.conv_tol = 1e-10

    print("\n--- Step 2: Define Single Target Active Fragment ---")
    # Select only one methyl group as the active QM region. 
    # The other half will automatically serve as the embedding environment.
    active_fragment = [0, 2, 3, 4]
    print(f"Target QM Active Region atom indices: {active_fragment}")

    print("\n--- Step 3: Initialize and Run Single Fragment Embedding ---")
    # Construct the single-shot embedding object. Notice that mf_inner_template 
    # will be cloned internally via .copy() to completely avoid cache poisoning.
    emb_obj = SingleFragmentEmbedding(
        mf_outer=mf_outer,
        mf_inner=mf_inner_template,
        fragment=active_fragment,
        threshold=1e-5  # Filters out pure fragment states and numerical noise
    )

    # Compute the final multi-scale total energy via the delta method:
    # E_tot = E_PBE(Full) + [E_B3LYP(Active) - E_PBE(Active)]
    e_embedded_tot = emb_obj.kernel()

    print("\n--- Step 4: Verification of Template Isolation ---")
    # Verify that our protection armor works seamlessly: 
    # Executing the template after embedding must converge successfully without any side effects.
    print("Verifying inner template isolation status...")
    mf_inner_template.kernel()
    if mf_inner_template.converged:
        print("Template isolation check passed successfully! No cache poisoning detected.")
    else:
        print("Warning: Template convergence failed, check cache isolation leaks.")

if __name__ == '__main__':
    run_dft_embedding_example()