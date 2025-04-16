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
