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

import pyscf
from gpu4pyscf.scf import hf
from gpu4pyscf.qmmm.pbc import itrf, mm_mole

import numpy as np

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='3-21g', max_memory=40000)

mol.verbose = 4
mf_GPU = hf.RHF(mol)

# Add mm charges:
# -0.8 charge at 1.0, 2.0,-1.0
#  0.8 charge at 3.0, 4.0, 5.0
# Place the QM and MM atoms in a box of 12 A
# Real-space cutoff for Ewald set to 8 A
# Exact MM potential computed for MM charges within 6 A of QM geometric center
mf_GPU = itrf.add_mm_charges(mf_GPU, [[1,2,-1],[3,4,5]], np.eye(3)*12, [-0.8,0.8], [0.8,1.2], rcut_ewald=8, rcut_hcore=6)

# Compute Energy of QM-QM and QM-MM but NOT MM-MM
e_dft = mf_GPU.kernel()
print(f"total energy = {e_dft}")

# Compute Gradient
g = mf_GPU.nuc_grad_method()
g.max_memory = 40000
g.auxbasis_response = True
# energy gradient w.r.t. QM atom positions
g_dft = g.kernel()
# energy gradient w.r.t. MM atom positions
g_mm = g.grad_nuc_mm() + g.grad_hcore_mm(mf_GPU.make_rdm1()) + g.de_ewald_mm
