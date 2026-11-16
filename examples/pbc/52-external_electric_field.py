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

'''Homogeneous external electric field in diamond silicon.'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc.dft import KRKS
from gpu4pyscf.pbc.properties.berry import polarization_difference
from gpu4pyscf.pbc.properties.external_field import apply


lattice_constant = 10.26
cell = gto.Cell(
    a=np.array([
        [0., lattice_constant / 2, lattice_constant / 2],
        [lattice_constant / 2, 0., lattice_constant / 2],
        [lattice_constant / 2, lattice_constant / 2, 0.],
    ]),
    atom=[
        ['Si', [0., 0., 0.]],
        ['Si', [lattice_constant / 4] * 3],
    ],
    unit='Bohr',
    basis='gth-szv',
    pseudo='gth-pbe',
    precision=1e-8,
    verbose=4,
)
kmesh = [7, 7, 7]

# First converge the insulating, integer-occupied zero-field reference.
mf = KRKS(cell, cell.make_kpts(kmesh), xc='pbe')
# MultiGrid accelerates the numerical integration of the XC functional.
mf = mf.multigrid_numint()
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-7
mf.kernel()
if not mf.converged:
    raise RuntimeError('Zero-field KRKS did not converge')
zero_field = apply(mf, field=np.zeros(3)).run()
polarization_zero = zero_field.polarization()

# Cartesian electric field in Hartree atomic units: 1 a.u. = 51.422 V/Angstrom.
field = np.array([0., 0., 1e-4])
print('Zero-field KS energy (Hartree):', mf.e_tot)
print('Electric field (a.u.):', field)

# solver='cg' is also supported, but usually requires more iterations.
ef = apply(mf, field=field, solver='scf')
ef.conv_tol = 1e-10
ef.conv_tol_grad = 1e-8
ef.conv_tol_dm = 1e-7
ef.max_cycle = 150
ef.kernel()
if not ef.converged:
    raise RuntimeError('External-field SCF did not converge')

# e_tot is the electric enthalpy; polarization includes electrons and ions.
print('KS energy       (Hartree):', ef.e_ks)
print('Field energy    (Hartree):', ef.e_field)
print('Electric enthalpy (Hartree):', ef.e_tot)
print('Total polarization (e/Bohr^2):', ef.polarization())
# Compare nearby states on the same Berry branch; Si's raw polarization
# can contain a quantum offset even though its zero branch is nonpolar.
delta_p = polarization_difference(ef.polarization(), polarization_zero, cell)
print('Induced polarization (e/Bohr^2):', delta_p)
