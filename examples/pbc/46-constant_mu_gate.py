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

'''Fixed-mu SCF with a compensating gate and no barrier.

This small H2 layer demonstrates the API, not a converged material calculation.
Lengths are Bohr; mu, sigma and energies are Hartree. The chemical potential
uses the variational gate reference, not an independently chosen vacuum zero.
'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf import gcscf
from gpu4pyscf.pbc.scf.addons import gate_charge

cell = gto.M(
    a=np.diag([7., 8., 18.]), unit='Bohr', dimension=3,
    atom='H 3 3 7.; H 3.2 3 8.4', basis='gth-szv', pseudo='gth-pade',
    mesh=[21, 23, 55], precision=1e-9, verbose=4)

# The gate is above the slab; put the dipole return region farther toward +z.
mf = gate_charge(dft.RKS(cell, xc='lda,vwn'), zgate=12.,
                 gate_side=1, barrier=False)
mf = gcscf.constant_mu(mf, mu=-.5, sigma=.2)
# Use only ordinary Fock damping throughout this demonstration.
# The addon itself retains the ordinary SCF default DIIS settings.
mf.diis = False
mf.damp = .8
mf.max_cycle = 150
mf.diis_start_cycle = mf.max_cycle + 1
mf.chkfile = None
mf.conv_tol = 1e-9
mf.conv_tol_grad = 1e-6
mf.kernel()
assert mf.converged

# cell.charge sets the initial guess only; the gate follows the actual density.
print('Electrons per cell:', mf.gcscf_nelec)
print('Gate charge per cell:', mf.scf_summary['gate_charge'])
print('Internal energy (Hartree):', mf.e_tot)
print('Helmholtz free energy (Hartree):', mf.e_free)
print('Grand potential (Hartree):', mf.e_grand)
