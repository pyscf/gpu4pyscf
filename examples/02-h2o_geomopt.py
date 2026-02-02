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
# Example of geometry optimization with DFT
####################################################

import time
import pyscf
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.disp = 'd3bj'
mf_GPU.grids.level = 3
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50

gradients = []
def callback(envs):
    gradients.append(envs['gradients'])

start_time = time.time()
mol_eq = optimize(mf_GPU, maxsteps=20, callback=callback)
print("Optimized coordinate:")
print(mol_eq.atom_coords())
print('Geometry optimization took', time.time() - start_time, 's')
