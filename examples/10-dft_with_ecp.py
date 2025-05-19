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

####################################################
#   Example of DFT with ECP
####################################################
import pyscf
from gpu4pyscf.dft import rks

atom = '''
I 0 0 0
I 1 0 0
'''

# def2-qzvpp contains ecp for heavy atoms
# One needs to specify ecp separately
mol = pyscf.M(atom=atom, basis='def2-qzvpp', ecp='def2-qzvpp')
mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.grids.level = 6   # more grids are needed for heavy atoms
e_dft = mf.kernel()

# ECP contributions are accelerated with GPU
g = mf.nuc_grad_method()
grad = g.kernel()

h = mf.Hessian()
hess = h.kernel()
