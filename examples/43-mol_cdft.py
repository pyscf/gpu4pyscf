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

from pyscf import gto
from gpu4pyscf.dft.ucdft import CDFT_UKS
from gpu4pyscf import dft
import cupy as cp
mol = gto.Mole()
mol.atom = '''O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

# Define Constraints (List of Lists)
# Constrain Left N (atom 0) to 6.5 electrons
# Constrain Right N (atom 1) to 7.5 electrons
charge_constraints = [ [0], [8.1] ]
    
# Initialize CDFT
mf = CDFT_UKS(mol, charge_constraints=charge_constraints)
# mf.grids.atom_grid = (99,590)
# mf.grids.coords = cp.array(
#     [
#         [0.1, 0.1, 0.1],
#         [0.0, -0.757*2, 0.587*2],
#         [0.0, 0.757*2, 0.587*2],
#     ]
# )
# mf.grids.weights = cp.array([1.0, 1.0, 1.0])
mf.xc = 'b3lyp'
print(">>> Starting Voronoi-CDFT Calculation...")
mf.kernel()

print("\n>>> Analysis of Results")
print(f"Converged Lagrange Multipliers V: {mf.v_lagrange}")

# Verification
dm = mf.make_rdm1()
projs = mf.build_atom_projectors()

# Calculate populations
n_left = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])

print(f"Number of projs: {len(projs)}")
print(f"Target N (Left):  {charge_constraints[1][0]}")
print(f"Result N (Left):  {float(n_left):.6f}")
