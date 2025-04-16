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

################################################################
#  Example of removing the linear dependence of atomic orbitals
################################################################

import time
from pyscf import gto
from gpu4pyscf.dft import rks

mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz', symmetry=1)

mf = rks.RKS(mol, xc='B3LYP').density_fit()
mf.grids.level = 5
mf.conv_tol = 1e-10
mf.direct_scf_tol = 1e-14
mf.nlcgrids.level = 2

import cupy
from functools import reduce
tol = 1e-6
def eig(h, s):
    d, t = cupy.linalg.eigh(s)
    x = t[:,d>tol] / cupy.sqrt(d[d>tol])
    xhx = reduce(cupy.dot, (x.T, h, x))
    e, c = cupy.linalg.eigh(xhx)
    c = cupy.dot(x, c)
    return e, c
from gpu4pyscf import scf
scf.hf.eigh = eig

e_tot = mf.kernel()
end_time = time.time()
