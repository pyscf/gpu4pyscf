#!/usr/bin/env python
'''Self-consistent dipole correction for a neutral polar slab.

The slab normal is z. Automatic selection places the sawtooth discontinuity
inside a low-density vacuum window; the cut remains fixed during SCF.
The default potential gauge has zero volume-weighted average on the grid.
Only energies and the KS potential are supported, not forces or stress.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import slab_dipole_correction

cell = gto.M(
    a=np.diag([4., 4., 12.]),
    atom='F 2 2 5.00; H 2 2 5.92',
    basis='gth-szv', pseudo='gth-pade', verbose=4)
kpts = cell.make_kpts([2, 2, 1])
mf = dft.KRKS(cell, kpts, xc='pbe').multigrid_numint()
slab_dipole_correction(mf)

# The zero point and the discontinuity can be set independently (both in Bohr):
# slab_dipole_correction(mf, z0=10.0)           # automatic cut, v_dip(10) = 0
# slab_dipole_correction(mf, cut=20.0)         # manual cut, zero-average potential
# slab_dipole_correction(mf, z0=10.0, cut=20.0)
# The cut must lie in vacuum; changing z0 leaves this cut fixed.
# To match PySCF's slab_center=10 convention:
# slab_dipole_correction(mf, z0=10.0, cut=10.0+cell.lattice_vectors()[2, 2]/2)

mf.kernel()
veff = mf.get_veff()
print('Sawtooth cut (Bohr):', mf.dipole_cut)
print('Potential zero (Bohr):', mf.dipole_z0)
print('Charge dipole (e Bohr):', veff.dipole)
print('Dipole energy correction (Hartree):', veff.edip)
