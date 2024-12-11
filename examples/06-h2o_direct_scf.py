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

#######################################################
#  Example of DFT with direct SCF scheme
#######################################################

from pyscf import gto
from gpu4pyscf import scf
import numpy as np

h2o = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = gto.M(atom=h2o, basis='def2-tzvpp', cart=1, verbose=4)

# Calculation on GPU
mf = scf.hf.RHF(mol)
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
e_gpu = mf.kernel()

gpu_gradient = mf.nuc_grad_method()
g_gpu = gpu_gradient.kernel()

gpu_hess = mf.Hessian()
h_gpu = gpu_hess.kernel()

# Move the object to CPU
mf = mf.to_cpu()
e_cpu = mf.kernel()

cpu_gradient = mf.nuc_grad_method()
g_cpu = cpu_gradient.kernel()

cpu_hess = mf.Hessian()
h_cpu = cpu_hess.kernel()

print('e diff = ', e_cpu - e_gpu)
print('g diff = \n', g_cpu - g_gpu)
print('h diff = \n', h_cpu - h_gpu)
