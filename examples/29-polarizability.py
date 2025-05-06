#!/usr/bin/env python
# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
Static polarizability (unit Bohr^3)
'''

import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf.properties import polarizability

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='631g'

mol = pyscf.M(atom=atom, basis=bas)

# Analytical

mf = mol.RKS(xc='b3lyp').to_gpu()
e_gpu = mf.kernel() #  -76.3849465432042
polar_gpu = polarizability.eval_polarizability(mf)
print('------------------- Polarizability -----------------------------')
print(polar_gpu)
"""
[[ 6.96413065e+00  9.60315894e-18 -2.25792304e-13]
 [ 9.60315894e-18  1.48264155e+00 -6.84920815e-15]
 [-2.25792304e-13 -6.84920815e-15  4.81230498e+00]]
"""

# Numerical

def apply_electric_field(mol, E):
    mf = mol.RKS(xc = 'b3lyp').to_gpu()
    mf.verbose = 0
    mf.conv_tol = 1e-14

    dipole_integral = cp.asarray(mol.intor('cint1e_r_sph', comp=3))
    E = cp.asarray(E)
    Hcore = mf.get_hcore() + cp.einsum('d,dij->ij', E, dipole_integral)

    mf.get_hcore = lambda *args: Hcore
    energy = mf.kernel()

    # The electric field - nuclei interaction energy is not necessary for polarizability calculation,
    # But is necessary for other purposes.
    nuclear_charge = -mol.atom_charges()
    nuclear_coords = mol.atom_coords()
    origin = np.zeros(3)
    nuclear_dipole = nuclear_charge @ (nuclear_coords - origin[None, :])
    energy += nuclear_dipole @ E.get()

    dipole = mf.dip_moment(unit = "au", verbose = 0)

    return energy, dipole

delta_E = 1e-4
polarizability_numerical = np.zeros((3,3))
for i_xyz in range(3):
    E_1p = np.zeros(3)
    E_1p[i_xyz] = delta_E
    e_1p, d_1p = apply_electric_field(mol, E_1p)

    E_1m = np.zeros(3)
    E_1m[i_xyz] = -delta_E
    e_1m, d_1m = apply_electric_field(mol, E_1m)

    polarizability_numerical[i_xyz, :] = (d_1p - d_1m) / (2 * delta_E)

print('---------------- Numerical Polarizability ----------------------')
print(polarizability_numerical)
