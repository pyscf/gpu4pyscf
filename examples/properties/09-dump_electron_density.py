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
