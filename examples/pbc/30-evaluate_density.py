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
Evaluate electron density in real space and reciprocal space
'''

import numpy as np
import cupy as cp
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
    basis = 'gth-dzv',
    pseudo = 'gth-pbe',
)

#
# 1. Electron density in real space.
#
#
# The SCF object provides the function `get_rho()` to evaluate the electron
# density on the integration grids associated with the SCF calculation.
# The default grid for PBC DFT is a uniform real-space grid in the unit cell.
#
kpts = cell.make_kpts([4,4,4])
mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
mf = mf.multigrid_numint()
mf.run()

density = mf.get_rho()
print(density.shape)

#
# The same function can also evaluate the density corresponding to a different
# density matrix.
#
dm_custom = mf.get_init_guess()
density = mf.get_rho(dm=dm_custom)
print(density.shape)

#
# To evaluate the density on a different grid, pass a custom grid object to
# `mf.get_rho()`.
#
from gpu4pyscf.pbc.dft import UniformGrids
grids = UniformGrids(cell)
grids.mesh = [10, 10, 10]
density = mf.get_rho(grids=grids)
print(density.shape)

#
# Alternatively, the electron density can be evaluated using the functions provided
# by `pbc.dft.numint.NumInt` or `pbc.dft.numint.multigrid.MultiGridNumInt` directly.
# For example, use the multigrid integrator to evaluate the density
#
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt
ni = MultiGridNumInt(cell)
ni.mesh = [10, 10, 10]
dm = mf.make_rdm1()
density = ni.get_rho(dm, kpts)
print(density.shape)

#
# 2. Electron density in reciprocal space
#
#
# The reciprocal-space density can be obtained by applying a Fourier transform
# to the real-space density. The normalization convention follows
# numpy.fft. The forward transform is unscaled and the inverse transform
# includes a factor of cell.vol/N, where N is the number of grid points.
#
from gpu4pyscf.pbc.tools import fft, ifft
density = mf.get_rho()
rhoG_FFT = fft(density, mf.grids.mesh)

#
# Alternatively, the multigrid integrator provides a specialized routine to
# compute the reciprocal-space density directly, which is more efficient.
#
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt, _eval_rhoG
ni = MultiGridNumInt(cell)
dm = mf.make_rdm1()
rhoG_direct = _eval_rhoG(ni, dm, kpts=kpts)
fac = cell.vol / np.prod(ni.mesh)
print(abs(rhoG_direct - rhoG_FFT * fac).max())

#
# 3. density derivatives.
#
#
# Density derivatives can be obtained by evaluating derivatives of AOs on the
# grid and contracting them with the density matrix.
#
# The function `eval_ao_kpts` can compute AO values and their derivatives up
# to a specified order.
#
from gpu4pyscf.pbc.dft.numint import eval_ao_kpts
grids = UniformGrids(cell)
grids.mesh = [10, 10, 10]

# ao is an array of the shape [nkpts, deriv-components, Ngrids, Nao]
ao = eval_ao_kpts(cell, grids.coords, kpts, deriv=1)
ao_0 = ao[:,0]      # AO values
ao_1 = ao[:,1:4]    # AO derivatives (dx, dy, dz)

# regular density
nkpts = len(kpts)
rho_0 = cp.einsum('kpq,kgq,kgp->g', dm, ao_0, ao_0) / nkpts

# first order density
rho_1 = cp.einsum('kpq,kgq,kxgp->g', dm, ao_0, ao_1)
rho_1 += cp.einsum('kpq,kxgq,kgp->g', dm, ao_1, ao_0)
rho_1 /= nkpts
