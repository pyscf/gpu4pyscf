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
import numpy as np
import unittest
import pytest
from gpu4pyscf.dft import uks
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0='def2-tzvpp'
grids_level = 5
nlcgrids_level = 3
def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, grid_response=False, xc='B3LYP', disp=None, tol=1e-6):
    mf = uks.UKS(mol, xc=xc)
    mf.disp = disp
    mf.grids.level = grids_level
    mf.grids.prune = None
    mf.small_rho_cutoff = 1e-30
    mf.direct_scf_tol = 1e-20
    if mf._numint.libxc.is_nlc(mf.xc):
        mf.nlcgrids.level = nlcgrids_level
    mf.kernel()
    gpu_gradient = mf.Gradients()
    gpu_gradient.grid_response = grid_response
    g_gpu = gpu_gradient.kernel()

    cpu_gradient = gpu_gradient.to_cpu()
    g_cpu = cpu_gradient.kernel()
    print('|| CPU - GPU ||:', np.linalg.norm(g_cpu - g_gpu))
    assert(np.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):

    def test_grad_with_grids_response(self):
        print("-----testing unrestricted DFT gradient with grids response----")
        _check_grad(mol_sph, grid_response=True)

    def test_grad_without_grids_response(self):
        print('-----testing unrestricted DFT gradient without grids response----')
        _check_grad(mol_sph, grid_response=False)

    def test_grad_lda(self):
        print("-----LDA testing-------")
        _check_grad(mol_sph, xc='LDA', disp=None)

    def test_grad_gga(self):
        print('-----GGA testing-------')
        _check_grad(mol_sph, xc='PBE', disp=None)

    def test_grad_hybrid(self):
        print('------hybrid GGA testing--------')
        _check_grad(mol_sph, xc='B3LYP', disp=None)

    def test_grad_mgga(self):
        print('-------mGGA testing-------------')
        _check_grad(mol_sph, xc='tpss', disp=None)

    def test_grad_rsh(self):
        print('--------RSH testing-------------')
        _check_grad(mol_sph, xc='wb97', disp=None)

    def test_grad_nlc(self):
        print('--------nlc testing-------------')
        _check_grad(mol_sph, xc='HYB_MGGA_XC_WB97M_V', disp=None)

    def test_grad_cart(self):
        print('------hybrid GGA Cart testing--------')
        _check_grad(mol_cart, xc='B3LYP', disp=None)

    def test_grad_d3bj(self):
        print('------hybrid GGA with D3(BJ) testing--------')
        _check_grad(mol_sph, xc='B3LYP', disp='d3bj')

    def test_grad_d4(self):
        print('------hybrid GGA with D4 testing--------')
        _check_grad(mol_sph, xc='B3LYP', disp='d4')

if __name__ == "__main__":
    print("Full Tests for UKS Gradient")
    unittest.main()
