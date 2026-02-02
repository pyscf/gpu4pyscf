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

# --- Example: Benzene S1/S2 Conical Intersection ---
# This example reproduces the calculation for the S1/S2 conical intersection
# of benzene.

from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.nac.mecp import ConicalIntersectionOptimizer

mol = gto.Mole()
mol.atom = [
    ['C', ( 0.0000,  1.3970, 0.1000)],
    ['C', ( 1.2100,  0.6985, 0.0000)],
    ['C', ( 1.2100, -0.6985, 0.0000)],
    ['C', ( 0.0000, -1.3970, 0.0000)],
    ['C', (-1.2100, -0.6985, 0.0000)],
    ['C', (-1.2100,  0.6985, 0.0000)],
    ['H', ( 0.0000,  2.4770, 0.0000)],
    ['H', ( 2.1450,  1.2385, 0.0000)],
    ['H', ( 2.1450, -1.2385, 0.0000)],
    ['H', ( 0.0000, -2.4770, 0.0000)],
    ['H', (-2.1450, -1.2385, 0.0000)],
    ['H', (-2.1450,  1.2385, 0.0000)],
]
mol.basis = 'ccpvdz'
mol.build()
    
mf = scf.RHF(mol).run()
td = mf.TDA()
td.nstates = 5
td.kernel()
ci_optimizer = ConicalIntersectionOptimizer(td, states=(1, 2), crossing_type='n-2')
print("Starting conical intersection optimization for Benzene S1/S2 using 'pyscf.geomopt.geometric_solver'...")
    
optimized_mol = ci_optimizer.optimize()
print("\n--- Optimization Finished ---")
print("Final optimized geometry (in Angstrom):")
print(optimized_mol.atom_coords(unit='A'))
    
# Final energy check at the optimized geometry
print("\nFinal state energies at the optimized geometry:")
ci_optimizer.get_eff_energy_and_gradient() # Run one last time to print final values
final_e = ci_optimizer.td.e
print(f"  E(S1) = {final_e[0]:.6f} Ha")
print(f"  E(S2) = {final_e[1]:.6f} Ha")
print(f"  Energy Gap = {abs(final_e[0] - final_e[1]):.6f} Ha")