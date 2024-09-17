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

mol = gto.M(atom=
'''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
basis='def2-tzvpp',
cart=1,
verbose=4)

# Calculation on GPU
mf = scf.hf.RHF(mol)
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
e_gpu = mf.kernel()

gpu_gradient = mf.nuc_grad_method()
gpu_gradient.kernel()

# Move the object to CPU
mf = mf.to_cpu()
e_cpu = mf.kernel()
cpu_gradient = mf.nuc_grad_method()
cpu_gradient.kernel()
cpu_de = cpu_gradient.de
cpu_de -= np.sum(cpu_de, axis=0)/mol.natm

print('e diff = ', e_cpu - e_gpu)
print('g diff = \n', cpu_de - gpu_gradient.de)

assert np.max(np.abs(cpu_de - gpu_gradient.de)) < 1e-6
