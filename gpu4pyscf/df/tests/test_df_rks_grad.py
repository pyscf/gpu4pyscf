# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pyscf
import cupy
import numpy as np
import unittest
from pyscf import lib
from gpu4pyscf.dft import rks

lib.num_threads(8)

'''
test density fitting for dft
1. energy
2. gradient
3. hessian
'''

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc0 = 'B3LYP'
bas0 = 'def2-tzvpp'
auxbasis0 = 'def2-tzvpp-jkfit'
disp0 = 'd3bj'
grids_level = 6
nlcgrids_level = 3
def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol.output = '/dev/null'
    mol.verbose = 1
    mol.build()

eps = 1.0/1024

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_grad(grid_response=False, xc=xc0, disp=disp0, tol=1e-6):
    mf = rks.RKS(mol, xc=xc, disp=disp).density_fit(auxbasis=auxbasis0)
    mf.grids.level = grids_level
    mf.nlcgrids.level = nlcgrids_level
    mf.conv_tol = 1e-10
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

class KnownValues(unittest.TestCase):

    def test_grad_with_grids_response(self):
        print("-----testing DF DFT gradient with grids response----")
        _check_grad(grid_response=True)

    def test_grad_without_grids_response(self):
        print('-----testing DF DFT gradient without grids response----')
        _check_grad(grid_response=False)

    def test_grad_lda(self):
        print("-----LDA testing-------")
        _check_grad(xc='LDA', disp=None, tol=1e-6)

    def test_grad_gga(self):
        print('-----GGA testing-------')
        _check_grad(xc='PBE', disp=None, tol=1e-6)

    def test_grad_hybrid(self):
        print('------hybrid GGA testing--------')
        _check_grad(xc='B3LYP', disp=None, tol=1e-6)

    def test_grad_mgga(self):
        print('-------mGGA testing-------------')
        _check_grad(xc='m06', disp=None, tol=1e-4)

    def test_grad_rsh(self):
        print('--------RSH testing-------------')
        _check_grad(xc='wb97', disp=None, tol=1e-4)

    def test_grad_nlc(self):
        print('--------nlc testing-------------')
        _check_grad(xc='HYB_MGGA_XC_WB97M_V', disp=None, tol=1e-6)

if __name__ == "__main__":
    print("Full Tests for DF Gradient")
    unittest.main()
