# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

###############################################################
#  Example of evaluating and saving electron density on grids
###############################################################

import numpy as np
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.grids.level = 3
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50

# Compute Energy
e_dft = mf_GPU.kernel()
dm = mf_GPU.make_rdm1()
grids = mf_GPU.grids
charges = mol.atom_charges()

# Prepare results for denspart
coords = grids.coords.get()
weights = grids.weights.get()
density = mf_GPU._numint.get_rho(mol, dm, grids)
atnums = np.array(charges)
atcoords = mol.atom_coords(unit='B')

# this file can be used to calculate the partial charge with denspart
np.savez(
    'density.npz',
    points=coords,
    weights=weights,
    density=density,
    atnums=atnums,
    atcoords=atcoords)