# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
from pyscf import gto
from gpu4pyscf.dft.ucdft import CDFT_UKS

# --- Molecule Setup ---
mol = gto.Mole()
mol.atom = '''O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''
mol.basis = '6-31g'
mol.charge = 0
mol.spin = 0
mol.build()

# Define constraints (list of lists)
# Constrain O (atom 0) to 8.1 electrons
# Constrain first H (atom 1) to 0.95 electrons
charge_constraints = [[0, 1], [8.1, 0.95]]

# =============================================================================
# 1. Standard Lagrange Multiplier Method
# =============================================================================
print("\n" + "="*40)
print("Method 1: Lagrange Multiplier (Default)")
print("="*40)

mf = CDFT_UKS(mol, charge_constraints=charge_constraints)
mf.grids.atom_grid = (99, 590)
mf.xc = 'b3lyp'
mf.kernel()  # E = -76.2906028922253

print("\n>>> Analysis of Results")
print(f"Converged Lagrange Multipliers (V): {mf.v_lagrange}")
# Expect: [ 0.32179487 -0.04154594]

# Verification of constraints
dm = mf.make_rdm1()
projs = mf.build_projectors()

# Calculate population on atoms using the projectors
# Note: projs list corresponds to the order of constraints defined (charge first)
O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])

print(f"Number of projectors: {len(projs)}")
print(f"Target O : {charge_constraints[1][0]:.6f} -> Result: {float(O_charge):.6f}")
print(f"Target H1: {charge_constraints[1][1]:.6f} -> Result: {float(H1_charge):.6f}")

# Get canonical MO energies
# Note: mf.mo_energy contains contributions from the constraint potential (V_cons).
# mf.get_canonical_mo() returns the eigenvalues of the standard Fock matrix
# computed with the constrained density.
mo_energy = mf.get_canonical_mo()[0]
print("\nCanonical MO energies (Alpha):")
print(mo_energy)
"""
[[-19.36209987  -1.12444181  -0.60693476  -0.46851442  -0.41940472
    0.03791074   0.12346241   0.79344581   0.79368322   0.80307202
    0.91837328   0.9845149    1.35833387]
 ...]
"""

# =============================================================================
# 2. Newton-Raphson Method (SOSCF)
# =============================================================================
print("\n" + "="*40)
print("Method 2: Newton-Raphson (SOSCF)")
print("="*40)
print("Using Newton-Raphson method for better convergence.")

# If the default nested iteration method fails to converge,
# you can try using the Newton-Raphson method.
# This typically converges.
mf_soscf = mf.newton()
mf_soscf.kernel()  # E = -76.2906028922247


# =============================================================================
# 3. Penalty Method
# =============================================================================
print("\n" + "="*40)
print("Method 3: Penalty Function")
print("="*40)

# We provide an alternative method using a penalty function.
# This can be useful if the Lagrange multiplier method struggles to converge.
# The penalty function is defined as: E_pen = lambda * (N - Nt)^2 
# lambda is the penalty weight.
mf_penalty = CDFT_UKS(mol, charge_constraints=charge_constraints,
                      method='penalty',
                      penalty_weight=10.0)
mf_penalty.grids.atom_grid = (99, 590)
mf_penalty.xc = 'b3lyp'

# The penalty method converges much more easily using SOSCF.
mf_penalty = mf_penalty.newton_penalty()
mf_penalty.kernel()  # E = -76.2955818883445

dm = mf_penalty.make_rdm1()
projs = mf_penalty._scf.build_projectors()
O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])

print("\n>>> Analysis of Penalty Results")
print("Note: The penalty method is an approximation and will inherently have residual errors.")
print(f"Target O : {charge_constraints[1][0]:.6f} -> Result: {float(O_charge):.6f}")
print(f"Target H1: {charge_constraints[1][1]:.6f} -> Result: {float(H1_charge):.6f}")


# =============================================================================
# 4. Gradients
# =============================================================================
print("\n" + "="*40)
print("Gradients")
print("="*40)

# Note: Analytical gradients are currently supported only for the MINAO 
# partition method. Gradients for the penalty method are not yet supported.

print("Calculating gradients for Lagrange method...")
g = mf.Gradients()
g.kernel()
"""
--------------- CDFT_UKS gradients ---------------
         x                y                z
0 O     0.0000000000    -0.0150321699     0.1229336008
1 H    -0.0000000000     0.0811590504    -0.0672953529
2 H     0.0000000000    -0.0661268790    -0.0556389656
--------------------------------------------------
"""

print("\nCalculating gradients for Second-Order Lagrange method...")
g = mf_soscf.Gradients()
g.kernel()
"""
--------------- SecondOrderCDFT_UKS gradients ---------------
         x                y                z
0 O     0.0000000000    -0.0150321704     0.1229335989
1 H    -0.0000000000     0.0811590496    -0.0672953523
2 H     0.0000000000    -0.0661268777    -0.0556389644
-------------------------------------------------------------
"""