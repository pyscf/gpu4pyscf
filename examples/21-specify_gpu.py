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

###########################################################
#  Example of specifying device in multi-GPU environment
###########################################################

import numpy as np
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

def run_dft():
    mol = pyscf.M(atom=atom, basis='def2-tzvpp', verbose=1)
    mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()

    # Compute Energy
    e_dft = mf_GPU.kernel()
    
    # Compute Gradient
    g = mf_GPU.nuc_grad_method()
    g_dft = g.kernel()
    
    # Compute Hessian
    h = mf_GPU.Hessian()
    h_dft = h.kernel()

import cupy
# Select Device #1 to run
with cupy.cuda.Device(1):
    run_dft()

with cupy.cuda.Device(0):
    run_dft()