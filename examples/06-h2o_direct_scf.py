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
