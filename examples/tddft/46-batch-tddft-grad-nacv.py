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

"""
Example: Simultaneous computation of Non-Adiabatic Coupling Vectors (NACVs) 
and Analytical Nuclear Gradients using batched GPU operations in gpu4pyscf.
"""

import numpy as np
import pyscf
from pyscf import dft
import gpu4pyscf

# =====================================================================
# 1. Define the Molecular System
# =====================================================================
mol = pyscf.M(
    atom="""
    O   0.000000000   0.000000000   0.117400000
    H  -0.757000000   0.000000000  -0.469600000
    H   0.757000000   0.000000000  -0.469600000
    """,
    basis="def2-tzvp",
    verbose=3
)

# =====================================================================
# 2. Ground State Calculation
# =====================================================================
print("\n--- Starting Ground State (RKS) Calculation ---")
mf = dft.RKS(mol, xc='b3lyp').to_gpu()
mf.kernel()

# =====================================================================
# 3. Excited State Calculation
# =====================================================================
print("\n--- Starting Excited State (TD-RKS) Calculation ---")
td = mf.TDA().set(nstates=4)
td.kernel()

# =====================================================================
# 4. Configure Batched NACV and Gradient Solver
# =====================================================================
print("\n--- Starting Batched NACV and Gradient Calculation ---")
# Instantiate the multistate NAC solver. 
# (Based on your tests, it is bound to the TD object as `nac_gradient_method`)
nac_solver = td.nac_gradient_method()

# Define the states between which the NACVs will be computed.
# '0' represents the ground state. '1', '2', '3' are excited states.
# The solver will automatically compute all unique pairs: 
# (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
nac_solver.states = (0, 1, 2, 3)

# Specify the target state for the analytical nuclear gradient computation.
# E.g., setting it to 1 will compute the gradient (forces) for S1.
# This gradient computation will be seamlessly batched with the NACV Z-vector equations.
nac_solver.grad_state = 1

# Execute the batched calculation. 
results = nac_solver.kernel()

# =====================================================================
# 5. Extract and Display the Results
# =====================================================================
print("\n" + "="*60)
print(" FINAL RESULTS ")
print("="*60)

# 5.1 Print Nuclear Gradient of the target state
print(f"\n[Nuclear Gradient for State S{nac_solver.grad_state}] (Hartree/Bohr):")
grad_s1 = nac_solver.grad_result

for i, atom_name in enumerate(mol.elements):
    print(f"{atom_name:>2}: "
          f"{grad_s1[i][0]:>12.6f} {grad_s1[i][1]:>12.6f} {grad_s1[i][2]:>12.6f}")

# 5.2 Print Non-Adiabatic Coupling Vectors (NACVs)
print("\n[Non-Adiabatic Coupling Vectors (NACVs)] (a.u.):")
# Iterate over all computed state pairs
for pair in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
    if pair in results:
        print(f"\n--- Coupling between S{pair[0]} and S{pair[1]} ---")
        
        # 'de_etf_scaled' is usually the standard NACV (Derivative Coupling with ETF, divided by dE)
        nacv_matrix = results[pair]['de_etf_scaled']
        
        for i, atom_name in enumerate(mol.elements):
            print(f"{atom_name:>2}: "
                  f"{nacv_matrix[i][0]:>12.6f} {nacv_matrix[i][1]:>12.6f} {nacv_matrix[i][2]:>12.6f}")

print("\n" + "="*60)
print("Calculation completed successfully!")