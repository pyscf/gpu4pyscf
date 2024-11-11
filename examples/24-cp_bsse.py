# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
    mf = rks.RKS(mol, xc='b3lyp').density_fit()
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
