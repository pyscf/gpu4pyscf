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

#################################################
#  Example of DFT with nonlocal corrections
#################################################

import pyscf
import time
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

start_time = time.time()
mol = pyscf.M(atom=atom, verbose=4)

print(f'{mol.nao} atomic orbitals')
mf = rks.RKS(mol, xc='HYB_MGGA_XC_WB97M_V').density_fit()
mf.grids.atom_grid = (99,590)
mf.nlcgrids.atom_grid = (50,194)
mf.conv_tol = 1e-8
mf.direct_scf_tol = 1e-14
e_tot = mf.kernel()
end_time = time.time()
print(f'Wallclock time: {end_time-start_time}')

# Compute gradient
gobj = mf.nuc_grad_method()
gobj.kernel()

# Compute Hessian
h = mf.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
