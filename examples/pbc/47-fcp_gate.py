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
FCP with two k-points, a planar countercharge and Fermi smearing.

This H2 layer demonstrates the API, not a converged material calculation.
Lengths are Bohr; chemical potentials, smearing width and energies are Hartree.
Only equal-weight KRKS in 3D without a barrier is supported.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge
from gpu4pyscf.pbc.scf.fcp import FCP, HARTREE2EV

cell = gto.M(
    a=np.diag([7., 8., 18.]), unit='Bohr',
    atom='H 3 3 7.; H 3.2 3 8.4',
    basis='gth-szv', pseudo='gth-pade',
    mesh=[21, 23, 55], precision=1e-9, verbose=4)

kpts = cell.make_kpts([2, 1, 1])
mf = dft.KRKS(cell, kpts, xc='lda,vwn')
# Add a planar countercharge.
mf = gate_charge(mf, zgate=12., gate_side=1, barrier=False)
mf.conv_tol = 1e-9
mf.conv_tol_grad = 1e-6
mf.max_cycle = 200
mf.chkfile = None
# Set mf.verbose = 5 to see fit coefficients, N, eta, Lambda and Newton steps.

# FCP enables fixed-N Fermi smearing. sigma=None or sigma=0 raises ValueError.
calc = FCP(mf, mu=0., sigma=.1/HARTREE2EV)
neutral = float(cell.atom_charges().sum())
mu0 = calc.evaluate(neutral)

# Keep this neutral reference fixed when changing the target or geometry.
delta_u = .5  # volts relative to the neutral model, not SHE
calc.mu = mu0 - delta_u/HARTREE2EV
print('mu0:', mu0)
calc.nelec_bounds = (neutral-.5, neutral+.5)
calc.kernel(nelec=neutral)
assert calc.converged

print('N per cell:', calc.nelec)
print('mu, target (Ha):', calc.mu_eff, calc.mu)
print('eta, Lambda (Ha):', calc.eta, calc.shift)
print('Helmholtz free energy (Ha):', calc.e_free)
print('Grand potential (Ha):', calc.e_grand)
