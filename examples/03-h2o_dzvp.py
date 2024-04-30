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
#  Example of DFT with different customized auxbasis
####################################################

import pyscf
import numpy as np
from pyscf import gto
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='dzvp')
auxbasis = {}
from pyscf import df
etb_basis = df.aug_etb(mol, beta=2.0)
for i in range(mol.natm):
    sym = mol.atom_pure_symbol(i)
    if sym not in auxbasis:
        # use even-tempered basis for I and other heavy atoms (nuc_charge > 36)
        nuc_charge = gto.charge(sym)
        if nuc_charge > 36:
            auxbasis[sym] = etb_basis[sym]
        else:
            auxbasis[sym] = 'def2-tzvpp-jkfit'

mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.disp = 'd3bj'
mf_GPU.grids.level = 3
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50

# Compute Energy
print('------------------- Energy -----------------------------')
e_dft = mf_GPU.kernel()
print('DFT energy by GPU4PySCF')
print(e_dft)

# Compute Gradient
print('------------------ Gradient ----------------------------')
g = mf_GPU.nuc_grad_method()
g.auxbasis_response = True
g_dft = g.kernel()
print('Gradient by GPU4PySCF')
print(g_dft)

# Compute Hessian
print('------------------- Hessian -----------------------------')
h = mf_GPU.Hessian()
h.auxbasis_response = 2
h_dft = h.kernel()
print('Diagonal entries of Mass-weighted Hessian by GPU4PySCF')
mass = [15.99491, 1.00783, 1.00783]
for i in range(3):
    for j in range(3):
        h_dft[i,j] = h_dft[i,j]/np.sqrt(mass[i]*mass[j])
n = h_dft.shape[0]
h_dft = h_dft.transpose([0,2,1,3]).reshape(3*n,3*n)
print(np.diag(h_dft))
