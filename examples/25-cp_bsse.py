#!/usr/bin/env python
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

####################################################
# Example of interaction energy with counterpoise correction
####################################################

import pyscf
from gpu4pyscf.dft import rks

atom_A = [
    ('O', (0.000000, 0.000000, 0.000000)),
    ('H', (0.000000, 0.757160, 0.586260)),
    ('H', (0.000000, -0.757160, 0.586260))
]

atom_B = [
    ('O', (0.000000, 0.000000, 2.913530)),
    ('H', (0.000000, 0.757160, 3.499790)),
    ('H', (0.000000, -0.757160, 3.499790))
]

atom_AB = atom_A + atom_B

mol_A = pyscf.M(atom=atom_A, basis='cc-pVDZ').build()
mol_B = pyscf.M(atom=atom_B, basis='cc-pVDZ').build()
mol_AB = pyscf.M(atom=atom_AB, basis='cc-pVDZ').build()

# Monomer A in the dimer basis
mol_A_ghost = mol_A.copy()
ghost_atoms_B = mol_B.atom
mol_A_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_B])
mol_A_ghost.build()

# Monomer B in the dimer basis
mol_B_ghost = mol_B.copy()
ghost_atoms_A = mol_A.atom
mol_B_ghost.atom.extend([('X-' + atom[0], atom[1]) for atom in ghost_atoms_A])
mol_B_ghost.build()

def solve_dft(mol, xc='b3lyp'):
    mf = rks.RKS(mol, xc=xc).density_fit()
    mf.grids.atom_grid = (99,590)
    return mf.kernel()

E_AB = solve_dft(mol_AB)
E_A = solve_dft(mol_A)
E_B = solve_dft(mol_B)
interaction_energy_no_bsse = E_AB - (E_A + E_B)
print(f"Interaction Energy without BSSE Correction: {interaction_energy_no_bsse:.6f} Hartree")

E_A_ghost = solve_dft(mol_A_ghost)
E_B_ghost = solve_dft(mol_B_ghost)
interaction_energy_bsse = E_AB - (E_A_ghost + E_B_ghost)
print(f"Interaction Energy with BSSE Correction: {interaction_energy_bsse:.6f} Hartree")
