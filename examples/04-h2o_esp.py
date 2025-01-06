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

#########################################################
#  Example of calculating Electrostatic potential (ESP)
#########################################################

import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.gto.int3c1e import int1e_grids

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='B3LYP').density_fit()
mf.kernel()
dm = mf.make_rdm1()  # compute one-electron density matrix

# Use default Lebedev grids
coords = mf.grids.coords

# Calculate electrostatic potential
v = int1e_grids(mol, coords, dm=dm) # performing 'ijp,ij->p' efficiently
