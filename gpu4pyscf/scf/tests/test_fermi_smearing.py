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

import numpy as np
import pyscf
from pyscf.hessian import thermo
from pyscf.scf import addons as cpu_addons

from gpu4pyscf.dft import rks
from gpu4pyscf.scf import addons

atom = """
Fe       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
"""

mol = pyscf.M(
    atom=atom,  # water molecule
    basis="def2-tzvpp",  # basis set
)

mf_GPU = rks.RKS(  # restricted Kohn-Sham DFT
    mol, xc="b3lyp"  # pyscf.gto.object  # xc funtionals, such as pbe0, wb97m-v, tpss,
).density_fit()  # density fitting

mf_GPU.grids.atom_grid = (99, 590)  # (99,590) lebedev grids, (75,302) is often enough
mf_GPU.conv_tol = 1e-10  # controls SCF convergence tolerance
mf_GPU.max_cycle = 50  # controls max iterations of SCF
mf_GPU.conv_tol_cpscf = 1e-3  # controls max iterations of CPSCF (for hessian)

mf_CPU = mf_GPU.to_cpu()
mf_CPU = cpu_addons.smearing(mf_CPU, sigma=0.01)
cpu_energy = mf_CPU.kernel()
cpu_gradient = mf_CPU.nuc_grad_method().kernel()

mf_GPU = addons.smearing(mf_GPU, sigma=0.01)
gpu_energy = mf_GPU.kernel()
gpu_gradient = mf_GPU.nuc_grad_method().kernel()

assert np.allclose(cpu_energy, gpu_energy, atol=1e-12)
assert np.allclose(cpu_gradient, gpu_gradient, atol=1e-12)
