#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

########################################
#  Example of GPU DF-CASSCF
########################################

import pyscf
from pyscf import mcscf as cpu_mcscf

from gpu4pyscf import mcscf as gpu_mcscf


mol = pyscf.M(
    atom='N 0 0 -0.7; N 0 0 0.7',
    basis='6-31g*',
)
mf = mol.RHF().to_gpu().density_fit().run()

mc_cpu = cpu_mcscf.DFCASSCF(mf.to_cpu(), 6, 6)
e_cpu = mc_cpu.kernel()[0]

mc_gpu = gpu_mcscf.DFCASSCF(mf.to_gpu(), 6, 6)
e_gpu = mc_gpu.kernel()[0]

print(f'CPU DF-CASSCF energy: {e_cpu:.12f}')
print(f'GPU DF-CASSCF energy: {e_gpu:.12f}')
print(f'CPU-GPU difference:   {e_gpu - e_cpu:.3e}')
