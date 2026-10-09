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

############################################
#  Example of DFT with SMD solvent model
############################################

import pyscf
from gpu4pyscf import dft

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp', verbose=1)
mf = dft.rks.RKS(mol, xc='HYB_GGA_XC_B3LYP').density_fit()
mf.grids.atom_grid = (99,590)
e_gas = mf.kernel()
print('total energy in gas phase:', e_gas)

# PCM can be initialized with a solvent name
# - A solvent name sets both eps and the optical dielectric constant used by TDDFT.
# - Names are case-insensitive: Names such as 'DMSO', 'THF', and 'MeCN' all work.
# - Also support kwarg "solvent", such as mf.SMD(solvent='water')
mf = mf.SMD('water')   # Add SMD model to the mean-field object

# Assign a solvent name to with_solvent attribute will update both eps and
# optical dielectric constant
mf.with_solvent.solvent = 'water'

mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids,
# Assigning mf.with_solvent.solvent also changes solvents and clears caches.
e_smd = mf.kernel()
print('total energy in water:', e_smd)

print('Solvation free energy:', e_smd - e_gas)
