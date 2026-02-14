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
# Use the multigrid_numint to enable the multigrid integration algorithm.
# This method overwrites the _numint attribute of the DFT object with the
# multigrid integrator.
#
mf = cell.RKS(xc='pbe').to_gpu()
mf = mf.multigrid_numint()
mf.run()
mf.get_bands

kpts = cell.make_kpts([2,2,2])
mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
mf = mf.multigrid_numint()
mf.run()

#
# The multigrid integrator also supports constructing the Fock matrix for
# arbitrary k-points along a band path. This can be used to compute
# band structures. The band path can be generated using the ASE package.
#
import cupy as cp
from gpu4pyscf.tools.ase_interface import bandpath, plot_band_structure

bp = bandpath(cell)
band_kpts = cell.get_abs_kpts(bp.kpts)
e_kn = mf.get_bands(band_kpts)[0]
e_kn = cp.asnumpy(cp.asarray(e_kn))

nocc = cell.nelectron // 2
e_kn = (e_kn - mf.get_fermi()) * HARTREE2EV

plot_band_structure(cell, e_kn, ax)
plt.tight_layout()
plt.show()
