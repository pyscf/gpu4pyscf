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

#############################################################
#  Example of transition state with geomeTRIC
#############################################################

import numpy as np
import pyscf

from pyscf import lib
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf.dft import rks

lib.num_threads(8)

atom='''
   C          3.21659       -1.41022       -0.26053
   C          2.16708       -0.35258       -0.59607
   N          1.21359       -0.16703        0.41640
   C          0.11616        0.82394        0.50964
   C         -1.19613        0.03585        0.74226
   N         -2.18193       -0.02502       -0.18081
   C         -3.43891       -0.74663        0.01614
   O          2.19596        0.25708       -1.63440
   C          0.11486        1.96253       -0.53088
   O         -1.29658       -0.59392        1.85462
   H          3.25195       -2.14283       -1.08721
   H          3.06369       -1.95423        0.67666
   H          4.20892       -0.93714       -0.22851
   H          1.24786       -0.78278        1.21013
   H          0.25990        1.31404        1.47973
   H         -2.02230        0.38818       -1.10143
   H         -3.60706       -1.48647       -0.76756
   H         -4.29549       -0.06423        0.04327
   H         -3.36801       -1.25875        0.98106
   H         -0.68664        2.66864       -0.27269
   H          0.01029        1.65112       -1.56461
   H          1.06461        2.50818       -0.45885'''


mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf_GPU = rks.RKS(mol, xc='b3lyp').density_fit()
mf_GPU.disp = 'd3bj'
mf_GPU.grids.level = 3
mf_GPU.conv_tol = 1e-10
mf_GPU.max_cycle = 50


# Transition state search in geomeTRIC will need Hessian matrix
import time
start_time = time.time()
mf_GPU.kernel()
h = mf_GPU.Hessian()
h.auxbasis_response = 2

h_dft = h.kernel()
natm = h_dft.shape[0]
h_dft = h_dft.transpose([0,2,1,3]).reshape([3*natm,3*natm])

from tempfile import NamedTemporaryFile
outfile = NamedTemporaryFile()
np.savetxt(outfile.name, h_dft)

mol_eq = optimize(mf_GPU, maxsteps=100, transition=True, hessian='file:'+outfile.name)
print("Optimized coordinate:")
print(mol_eq.atom_coords())
print('transition state search takes', time.time() - start_time, 's')
