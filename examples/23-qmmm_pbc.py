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

import pyscf
from gpu4pyscf.scf import hf
from gpu4pyscf.qmmm.pbc import itrf, mm_mole

import numpy as np

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='3-21g', max_memory=40000)

mol.verbose = 4
mf_GPU = hf.RHF(mol)

# Add mm charges:
# -0.8 charge at 1.0, 2.0,-1.0
#  0.8 charge at 3.0, 4.0, 5.0
# Place the QM and MM atoms in a box of 12 A
# Real-space cutoff for Ewald set to 8 A
# Exact MM potential computed for MM charges within 6 A of QM geometric center
mf_GPU = itrf.add_mm_charges(mf_GPU, [[1,2,-1],[3,4,5]], np.eye(3)*12, [-0.8,0.8], [0.8,1.2], rcut_ewald=8, rcut_hcore=6)

# Compute Energy of QM-QM and QM-MM but NOT MM-MM
e_dft = mf_GPU.kernel()
print(f"total energy = {e_dft}")

# Compute Gradient
g = mf_GPU.nuc_grad_method()
g.max_memory = 40000
g.auxbasis_response = True
# energy gradient w.r.t. QM atom positions
g_dft = g.kernel()
# energy gradient w.r.t. MM atom positions
g_mm = g.grad_nuc_mm() + g.grad_hcore_mm(mf_GPU.make_rdm1()) + g.de_ewald_mm
