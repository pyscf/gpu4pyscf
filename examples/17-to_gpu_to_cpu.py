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

