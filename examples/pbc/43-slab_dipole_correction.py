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

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import slab_dipole_correction

'''
Self-consistent dipole correction for a neutral polar slab.

The slab normal is z. Automatic selection places the sawtooth discontinuity
inside a low-density vacuum window; the cut remains fixed during SCF.
The default potential gauge has zero volume-weighted average on the grid.
Energies, the KS potential and nuclear gradients are supported, but stress is
not implemented.
'''

cell = gto.M(
    a=np.diag([4., 4., 12.]),
    atom='F 2 2 5.00; H 2 2 5.92',
    basis='gth-szv', pseudo='gth-pade', verbose=4)
kpts = cell.make_kpts([2, 2, 1])
mf = dft.KRKS(cell, kpts, xc='pbe').multigrid_numint()
slab_dipole_correction(mf)

# IMPORTANT: z0 and cut are always in Bohr, regardless of cell.unit.
# slab_dipole_correction(mf, z0=10.0)          # automatic cut, v_dip(10) = 0
# slab_dipole_correction(mf, cut=20.0)         # manual cut, zero-average potential
# slab_dipole_correction(mf, z0=10.0, cut=20.0)
# The cut must lie in vacuum; changing z0 leaves this cut fixed.
# To match PySCF's slab_center=10 Bohr convention:
# slab_dipole_correction(mf, z0=10.0, cut=10.0+cell.lattice_vectors()[2, 2]/2)

mf.kernel()
veff = mf.get_veff()
print('Sawtooth cut (Bohr):', mf.dipole_cut)
print('Potential zero (Bohr):', mf.dipole_z0)
print('Charge dipole (e Bohr):', veff.dipole)
print('Dipole energy correction (Hartree):', veff.edip)

# Analytic nuclear gradient of the dipole-corrected total energy.
# For geometry optimization, set cut explicitly in Bohr in the vacuum so that
# the sawtooth branch remains fixed when the cell is reset.
gradient = mf.nuc_grad_method().kernel()
print('Nuclear gradient (Hartree/Bohr):')
print(gradient)
