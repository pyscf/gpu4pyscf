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

'''Wannier-center Berry-phase polarization of diamond silicon.'''

import numpy as np
from pyscf.pbc import gto
from gpu4pyscf.pbc.properties import berry


lattice_constant = 10.26  # Bohr
cell = gto.Cell(
    a=np.asarray([
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
    precision=1e-10,
    verbose=4,
)
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
mf.conv_tol = 1e-10
mf.kernel()

result = berry.eval_polarization(
    mf, kmesh=kmesh, unit='C/m^2', return_details=True)

# Polarization is lattice-valued. Diamond Si is inversion symmetric, so the
# branch nearest zero must vanish as the k mesh and basis are converged.
polarization_near_zero, branch = berry.unwrap_polarization(
    result.total, np.zeros(3), cell, unit=result.unit, return_branch=True)

print('Electronic polarization (C/m^2):', result.electronic)
print('Ionic polarization    (C/m^2):', result.ionic)
print('Raw total polarization (C/m^2):', result.total)
print('Branch indices:', branch)
print('Total polarization nearest zero (C/m^2):', polarization_near_zero)

for direction, name in enumerate('xyz'):
    centers = result.wannier.centers[0][direction]
    print(f'Hybrid Wannier centers along {name} (fractional):')
    print(centers)
