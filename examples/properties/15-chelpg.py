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

###################################
#  Example of CHELG charge
###################################

from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm import chelpg

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    [1 , (1. ,  0.     , 0.000)],
    [1 , (0. ,  1.     , 0.000)],
    [1 , (0. , -1.517  , 1.177)],
    [1 , (0. ,  1.517  , 1.177)] ]
mol.basis = '631g'
mol.unit = 'B'
mol.build()
mol.verbose = 4

xc = 'b3lyp'
mf = rks.RKS(mol, xc=xc)
mf.grids.level = 5
mf.kernel()
q = chelpg.eval_chelpg_layer_gpu(mf)
print('Partial charge with CHELPG, using modified Bondi radii')
print(q) # [ 0.04402311  0.11333945 -0.25767919  0.10031663]

# Customize the radii used for calculating CHELPG charges
from pyscf.data import radii
q = chelpg.eval_chelpg_layer_gpu(mf, Rvdw=radii.UFF)
print('Partial charge with CHELPG, using UFF radii')
print(q)
