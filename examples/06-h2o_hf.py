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

from pyscf import gto, scf, grad
import numpy as np

mol = gto.M(atom=
'''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
basis='cc-pvtz',
verbose=4)

mf = scf.hf.RHF(mol)
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
e_cpu = mf.kernel()

cpu_gradient = grad.rhf.Gradients(mf)
cpu_gradient.kernel()

from gpu4pyscf import scf
import gpu4pyscf

mf = scf.hf.RHF(mol)
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
e_gpu = mf.kernel()
gpu_gradient = gpu4pyscf.grad.rhf.Gradients(mf)
gpu_gradient.kernel()

cpu_de = cpu_gradient.de
print(cpu_de)
print(gpu_gradient.de)

cpu_de -= np.sum(cpu_de, axis=0)/mol.natm
print('e diff = ', e_cpu - e_gpu)
print('g diff = \n', cpu_de - gpu_gradient.de)

assert np.max(np.abs(cpu_de - gpu_gradient.de)) < 1e-6
