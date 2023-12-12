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


import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf.dft import rks

lib.num_threads(8)

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc = 'B3LYP'
bas = 'def2-tzvpp'
auxbasis = 'def2-tzvpp-jkfit'
scf_tol = 1e-10
max_scf_cycles = 50
screen_tol = 1e-14
grids_level = 8

mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
# set verbose >= 6 for debugging timer
mol.verbose = 1
mf_GPU = rks.RKS(mol, xc=xc, disp='d3bj').density_fit(auxbasis=auxbasis)
mf_GPU.grids.level = grids_level
mf_GPU.conv_tol = scf_tol
mf_GPU.max_cycle = max_scf_cycles
mf_GPU.screen_tol = screen_tol

# Compute Energy
print('------------------- Energy -----------------------------')
e_dft = mf_GPU.kernel()
print('DFT energy by GPU4PySCF')
print(e_dft)
print('DFT energy by Q-Chem')
print(-76.4672557846) # reference from q-chem: -76.4672557846

# Compute Gradient
print('------------------ Gradient ----------------------------')
g = mf_GPU.nuc_grad_method()
g.auxbasis_response = True
g_dft = g.kernel()
print('Gradient by GPU4PySCF')
print(g_dft)
print('Gradient by Q-Chem')
# reference from q-chem
print(np.array([[0.0000000,   0.0030278,  -0.0030278],
        [-0.0000000,  -0.0000000,   0.0000000],
        [-0.0023449,   0.0011724,   0.0011724]]).T)

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
print('Diagonals entries of Mass-weighted Hessian by Q-Chem')
hess_qchem = np.loadtxt('hess_qchem.txt')
print(np.diag(hess_qchem))
