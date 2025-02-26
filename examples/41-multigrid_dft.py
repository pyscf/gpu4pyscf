#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Using multi-grid algorithm for DFT calculation
'''

import numpy as np
import pyscf
from gpu4pyscf.pbc.dft.multigrid import MultiGridNumInt

cell = pyscf.M(
    a = np.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-dzvp',
    pseudo = 'gth-pbe',
    verbose = 5,
)

#
# To enable the multi-grid integral algorithm, we can overwrite the _numint
# attribute of the DFT object
#
mf = cell.RKS(xc='pbe').to_gpu()
mf._numint = MultiGridNumInt(cell)
mf.run()

# Build a 2x2x2 super cell, its energy is equal to 8x of the k-point calculation
# below
#    kpts = cell.make_kpts([2,2,2])
#    mf = cell.KRKS(xc='pbe', kpts=kpts).run()

from pyscf.pbc.tools.pbc import super_cell
cell = super_cell(cell, [2,2,2])
mf = cell.RKS(xc='pbe').to_gpu()
mf._numint = MultiGridNumInt(cell)
mf.run()
