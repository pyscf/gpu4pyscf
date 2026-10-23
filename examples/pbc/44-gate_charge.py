#!/usr/bin/env python
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

'''Field-effect doping with a charged sheet, barrier and dipole correction.

Refs: Brumme et al., Phys. Rev. B 89, 245406 (2014).
      https://mcresearch.github.io/abacus-user-guide/abacus-surface6.html
Lengths below are Bohr; barrier_height is Hartree. ABACUS fractional
positions and widths are converted by multiplying the cell length L.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge, slab_dipole_correction

cell = gto.M(
    a=np.diag([8., 8., 28.]), unit='Bohr',
    atom='F 3 3 6.; H 3 3 7.7',
    basis='gth-szv', pseudo='gth-pade', mesh=[41, 41, 141],
    charge=-1, spin=1,  # One extra electron; the gate carries +1.
    precision=1e-10, verbose=4)

# For Nk k-points, scale cell.charge and cell.spin by Nk to keep the same
# charge and spin difference per primitive cell.
length = cell.lattice_vectors()[2, 2]
mf = dft.UKS(cell, xc='pbe')
gate_charge(mf, zgate=.5*length, barrier=True,
            barrier_start=.45*length, barrier_end=.55*length,
            barrier_height=.05, barrier_width=.01*length)
# Both monopole and dipole potentials use Brumme's zero-cell-integral
# reference. Do not set z0 when composing with gate_charge.
# Keep the sheet and the entire linear return region in the vacuum.
# Fix cut for force calculations/scanners to stay on the same branch.
slab_dipole_correction(mf, cut=.75*length, width=.1*length)
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

print('Total energy (Hartree):', mf.e_tot)
print('Gate energy (Hartree):', mf.scf_summary['egate'])
print('Barrier energy (Hartree):', mf.scf_summary['ebarrier'])
print('Dipole energy (Hartree):', mf.get_veff().edip)
print('Atomic forces (Hartree/Bohr):')
print(-mf.nuc_grad_method().kernel())
