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

########################################################################
#  Example of the conversions between PySCF object and GPU4PySCF object
########################################################################

import numpy as np
import pyscf
from pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

# ----------------------- to_gpu -------------------------------
print('running to_gpu ...')
mol = pyscf.M(atom=atom, basis='sto3g', output='./pyscf.log')
mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.conv_tol = 1e-10
mf.conv_tol_cpscf = 1e-6

# Compute Energy
e_cpu = mf.kernel()
e_gpu = mf.to_gpu().kernel()
print(f"Energy diff {e_cpu - e_gpu}")

# Compute Gradient
g = mf.nuc_grad_method()
g_cpu = g.kernel()
g_gpu = g.to_gpu().kernel()
print(f"Gradient diff {np.linalg.norm(g_cpu - g_gpu)}")

# Compute Hessian
h = mf.Hessian()
h_cpu = h.kernel()
h_gpu = h.to_gpu().kernel()
print(f"Hessian diff {np.linalg.norm(h_cpu - h_gpu)}")

# ----------------------- to_cpu -------------------------------
from gpu4pyscf.dft import rks
print('running to_cpu ...')
mol = pyscf.M(atom=atom, basis='sto3g', output='./pyscf.log')
mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.conv_tol = 1e-10
mf.conv_tol_cpscf = 1e-6

e_gpu = mf.kernel()
e_cpu = mf.to_cpu().kernel()
print(f"Energy diff {e_cpu - e_gpu}")

# Compute Gradient
g = mf.nuc_grad_method()
g_gpu = g.kernel()
g_cpu = g.to_cpu().kernel()
print(f"Gradient diff {np.linalg.norm(g_cpu - g_gpu)}")

# Compute Hessian
h = mf.Hessian()
h_gpu = h.kernel()
h_cpu = h.to_cpu().kernel()
print(f"Hessian diff {np.linalg.norm(h_cpu - h_gpu)}")

