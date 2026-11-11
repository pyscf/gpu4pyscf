# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Fixed-charge gate for a dimension=2 slab, without a periodic dipole correction.
Lengths are Bohr; barrier_height is Hartree.
This small CH+ layer illustrates the API, not a converged material calculation.

It should be noted that multigrid and density-fit cannot work with dimension=2.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge

# Use the asymmetric cell and tilted bond of the SCF regression model to
# avoid an artificial orbital degeneracy in this small Gamma-only example.
cell = gto.M(
    a=np.diag([7., 8., 60.]), unit='Bohr', dimension=2,
    atom='C 2.1 2.8 0.; H 2.8 3.3 1.7',
    basis='gth-szv', pseudo='gth-pade', mesh=[31, 33, 249],
    charge=1, spin=0,  # the gate carries -1.
    precision=1e-10, verbose=4)

# Keep low_dim_ft_type at its default, not 'inf_vacuum'. The default uniform
# grid spans [-L/2, L/2) along z, so center the slab near z=0. Gate/barrier
# coordinates are used without wrapping; choose sufficient vacuum yourself.
# For Nk k-points, cell.charge must be Nk times the charge per primitive cell.
# Use only in-plane k-points. The default FFTDF and UniformGrids are retained.
mf = dft.RKS(cell, xc='lda,vwn')
mf = gate_charge(mf, zgate=-5., barrier=True,
                 barrier_start=-7., barrier_end=-3.,
                 barrier_width=.5, barrier_height=.05)
mf.chkfile = None
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

print('Total energy (Hartree):', mf.e_tot)
print('Gate energy (Hartree):', mf.scf_summary['egate'])
print('Barrier energy (Hartree):', mf.scf_summary['ebarrier'])

# There is no edip term in 2D. The gate self-energy is zero in the PySCF
# finite-part reference; its electronic and ionic terms are both included.
# Full 2D atomic gradients are not yet available. Do not interpret the
# explicit gate_grad contribution as a total force or use it for relaxation.
# Same-dimension mf.reset(new_cell) keeps gate positions and widths in Bohr;
# construct a new SCF object when changing between dimension=2 and 3.
