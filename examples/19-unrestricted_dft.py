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

###########################################
#  Example of unrestricted DFT
###########################################

import pyscf
from gpu4pyscf.dft import uks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

# SCF, gradient, and Hessian for DF-UKS
mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mol.charge = 1
mol.spin = 1
mf = uks.UKS(mol, xc='b3lyp').density_fit()
mf.kernel()

gobj = mf.nuc_grad_method()
g = gobj.kernel()

hobj = mf.Hessian()
h = hobj.kernel()

# SCF, gradient, and Hessian for DF-UKS with IEF-PCM
mf_with_pcm = mf.PCM()
mf_with_pcm.with_solvent.method = 'IEF-PCM'
mf_with_pcm.kernel()

gobj_with_pcm = mf_with_pcm.nuc_grad_method()
g = gobj_with_pcm.kernel()

hobj_with_pcm = mf_with_pcm.Hessian()
h = hobj_with_pcm.kernel()
