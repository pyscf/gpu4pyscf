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

import pyscf
from pyscf import lib
from pyscf.dft import rks
lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc = 'B3LYP'
bas = 'sto3g'
scf_tol = 1e-10
max_scf_cycles = 50
screen_tol = 1e-14
grids_level = 3

mol = pyscf.M(atom=atom, basis=bas, max_memory=32000, output='./pyscf.log')

mol.verbose = 4
mf = rks.RKS(mol, xc=xc).density_fit()
mf_GPU = mf.to_gpu()
mf_GPU.grids.level = grids_level
mf_GPU.conv_tol = scf_tol
mf_GPU.max_cycle = max_scf_cycles
mf_GPU.screen_tol = screen_tol

# Compute Energy
e_dft = mf_GPU.kernel()
print(f"total energy = {e_dft}")

# Compute Gradient
g = mf_GPU.nuc_grad_method()
g.max_memory = 20000
g.auxbasis_response = True
g_dft = g.kernel()

# Compute Hessian
h = mf_GPU.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
