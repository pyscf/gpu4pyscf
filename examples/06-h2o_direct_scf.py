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
