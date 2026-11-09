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

# This legacy example does not set cell.nelectron. For Nk k-points, scale
# cell.charge and cell.spin by Nk to keep the same charge and spin difference
# per primitive cell. The explicit-electron example below uses a different convention.
length = cell.lattice_vectors()[2, 2]
mf = dft.UKS(cell, xc='pbe')

# gate_charge adds the dipole automatically, with d_dip=.01*length and
# cut = zgate + gate_side*(epsilon + d_dip/2), modulo L.
# Here gate_side=+1 puts the gate above the slab. The dipole is farther
# toward +z; the default barrier extends toward the slab on the lower-z side.
# z0 = (cut + L/2) % L. Explicit positions persist in force scanners.
# For a lower gate, use gate_side=-1 and zgate=.1*length instead.
gate_charge(mf, zgate=.5*length, gate_side=1,
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


# Fractional electrons: explicit per-cell charge and electron count.
# The F/H pseudopotentials give 8 valence electrons in the neutral cell.
# Set these three parameters AFTER building the cell and BEFORE gate_charge:
fractional_cell = cell.copy()  # Keep the built geometry, basis and pseudopotentials.
fractional_cell.charge = 0.3   # Remove 0.3 electrons; the gate carries -0.3.
fractional_cell.nelectron = 7.7  # Explicit target PER PRIMITIVE CELL.
fractional_cell.spin = 0       # Explicit gate electron counts require RKS/KRKS.
# Required: nelectron + charge == sum of effective ionic charges.
# To add 0.3 electrons instead, use charge=-0.3 and nelectron=8.3.
# Unlike the legacy example, DO NOT multiply charge or nelectron by Nk.

kpts = fractional_cell.make_kpts([2, 1, 1])
fractional_mf = dft.KRKS(fractional_cell, kpts, xc='pbe')
fractional_mf = gate_charge(
    fractional_mf, zgate=.5*length, gate_side=1, dipole_width=.01*length,
    dipole_gap=1e-4, barrier=True, barrier_height=.05)

# Without smearing, get_occ automatically fills the lowest states across k-points
# and shares any remaining electrons over the frontier group. No manual mo_occ.
# For Fermi smearing instead, uncomment this line; sigma is in Hartree:
# fractional_mf = fractional_mf.smearing(sigma=.01, method='fermi')
fractional_mf.conv_tol = 1e-10
fractional_mf.kernel()
assert fractional_mf.converged

nelectron = float(fractional_mf.mo_occ.sum()) / len(kpts)
print('Fractional gate total energy (Hartree):', fractional_mf.e_tot)
print('Electrons per primitive cell:', nelectron)
assert abs(nelectron - fractional_cell.nelectron) < 1e-9
# Nuclear gradients are not yet supported for this explicit-electron gate mode.
