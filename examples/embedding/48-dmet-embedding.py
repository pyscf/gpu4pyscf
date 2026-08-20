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
Example 48: Standard Multi-Fragment Self-Consistent DMET Calculation.

This script demonstrates how to partition a molecule into multiple fragments
and optimize the global correlation potential (u_oao) to match high-level and 
low-level local 1-RDM density matrices self-consistently.
"""

from pyscf import gto
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.qmmm.embedding.embedding import DMET

def run_dmet_example():
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
    mol.verbose = 4  # Set verbose to see detailed DMET iteration logs
    mol.build()

    print("--- Step 1: Initialize Low-Level and High-Level Solver Templates ---")
    # In this classic exact-back-to-exact test case, we nest RHF within RHF.
    # DMET should converge the correlation potential to exactly zero.
    mf_outer = gpu_hf.RHF(mol)
    mf_outer.conv_tol = 1e-12
    
    mf_inner_template = gpu_hf.RHF(mol)
    mf_inner_template.conv_tol = 1e-12

    print("\n--- Step 2: Define Molecular Fragments ---")
    # Partition the Ethane molecule into two methyl fragments based on atom indices:
    # Fragment 0: First Methyl group [C1, H1, H2, H3]
    # Fragment 1: Second Methyl group [C2, H4, H5, H6]
    fragments = [
        [0, 2, 3, 4],
        [1, 5, 6, 7]
    ]
    print(f"Fragment 0 atom indices: {fragments[0]}")
    print(f"Fragment 1 atom indices: {fragments[1]}")

    print("\n--- Step 3: Setup and Execute the Self-Consistent DMET Solver ---")
    dmet_solver = DMET(
        mf_outer=mf_outer,
        mf_inner=mf_inner_template,
        fragments=fragments,
        threshold=1e-5,       # SVD eigenvalue threshold for bath selection
        max_macro_iter=20,    # Max macro loops for correlation potential fitting
        macro_tol=1e-4        # Convergence tolerance for the density matching cost
    )

    # Trigger the DMET macroscopic self-consistent optimization
    e_dmet = dmet_solver.kernel()

    print("\n--- Final Results Summary ---")
    # Run the raw full system RHF as an exact reference
    e_hf_ref = mf_outer.kernel()
    
    print(f"Global Reference RHF Energy  : {e_hf_ref:.8f} Hartree") # -79.19706462
    print(f"Macroscopic DMET Total Energy: {e_dmet:.8f} Hartree") # -79.19706462
    print(f"Absolute Energy Deviation    : {abs(e_dmet - e_hf_ref):.2e} Hartree") # 9.15e-11 Hartree

if __name__ == '__main__':
    run_dmet_example()