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
mf_GPU = rks.RKS(mol, xc='b3lyp', disp='d3bj').density_fit()
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
print('geometry optimization took', time.time() - start_time, 's')
