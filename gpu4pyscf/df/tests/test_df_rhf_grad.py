# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import pyscf
import cupy
import numpy as np
import unittest
from gpu4pyscf import scf

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0 = 'def2-tzvpp'
auxbasis0 = 'def2-tzvpp-jkfit'

def setUpModule():
    global mol_cart, mol_sph
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)
eps = 1e-3

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, grid_response=False, tol=1e-6):
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-14
    mf.direct_scf_tol = 1e-20
    mf.verbose = 1
    mf.kernel()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.grid_response = grid_response

    g_scanner = g.as_scanner()
    g_analy = g_scanner(mol)[1]
    print('analytical gradient:')
    print(g_analy)

    f_scanner = mf.as_scanner()
    coords = mol.atom_coords()
    grad_fd = np.zeros_like(coords)
    for i in range(len(coords)):
        for j in range(3):
            coords = mol.atom_coords()
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e0 = f_scanner(mol)

            coords[i,j] -= 2.0 * eps
            mol.set_geom_(coords, unit='Bohr')
            mol.build()
            e1 = f_scanner(mol)

            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            grad_fd[i,j] = (e0-e1)/2.0/eps
    grad_fd = np.array(grad_fd).reshape(-1,3)
    print('finite difference gradient:')
    print(grad_fd)
    print('difference between analytical and finite difference gradient:', cupy.linalg.norm(g_analy - grad_fd))
    assert(cupy.linalg.norm(g_analy - grad_fd) < tol)

def _vs_cpu(mol, grid_response=False, tol=1e-9):
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis0)
    mf.verbose = 1
    mf.kernel()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.grid_response = grid_response
    g_analy = g.kernel()
    
    g_cpu = g.to_cpu()
    ref = g_cpu.kernel()
    print('CPU - GPU:', abs(g_analy - ref).max())
    assert abs(g_analy - ref).max() < tol

class KnownValues(unittest.TestCase):

    def test_grad_sph(self):
        _vs_cpu(mol_sph)
    
    def test_grad_cart(self):
        _vs_cpu(mol_cart)
    
if __name__ == "__main__":
    print("Full Tests for DF RHF Gradient")
    unittest.main()
