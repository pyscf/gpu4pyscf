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

'''
Field-effect with a charged sheet, barrier and dipole correction.

Refs: Brumme et al., Phys. Rev. B 89, 245406 (2014).
Lengths below are Bohr; barrier_height is Hartree.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge

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

# gate_charge adds the dipole automatically, with
# cut = zgate + gate_side*(epsilon + d_dip/2), modulo L.
# Here gate_side=+1 puts the gate above the slab. The dipole is farther
# toward +z; the default barrier extends toward the slab on the lower-z side.
# z0 = (cut + L/2) % L. Explicit positions persist in force scanners.
# For a lower gate, use gate_side=-1 and zgate=.1*length instead.
gate_charge(mf, zgate=.5*length, gate_side=1, dipole_width=.01*length,
            dipole_gap=1e-4, barrier=True, barrier_height=.05)
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

print('Total energy (Hartree):', mf.e_tot)
print('Gate energy (Hartree):', mf.scf_summary['egate'])
print('Barrier energy (Hartree):', mf.scf_summary['ebarrier'])
print('Dipole energy (Hartree):', mf.get_veff().edip)
print('Atomic forces (Hartree/Bohr):')
print(-mf.nuc_grad_method().kernel())
