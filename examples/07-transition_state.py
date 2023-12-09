# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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


xc = 'B3LYP'
bas = 'def2-tzvpp'
auxbasis = 'def2-tzvpp-jkfit'
scf_tol = 1e-10
max_scf_cycles = 50
screen_tol = 1e-14
grids_level = 3
mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)

mol.verbose = 1
mf_GPU = rks.RKS(mol, xc=xc, disp='d3bj').density_fit(auxbasis=auxbasis)
mf_GPU.grids.level = grids_level
mf_GPU.conv_tol = scf_tol
mf_GPU.max_cycle = max_scf_cycles
mf_GPU.screen_tol = screen_tol

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