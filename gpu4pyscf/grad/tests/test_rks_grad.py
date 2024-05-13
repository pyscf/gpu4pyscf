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
import unittest
import pytest
from pyscf.dft import rks as cpu_rks
from gpu4pyscf.dft import rks as gpu_rks
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

bas0='def2-tzvpp'
grids_level = 5
nlcgrids_level = 3
def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.build()
    mol_sph.verbose = 1

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1)
    mol_cart.output = '/dev/null'
    mol_cart.build()
    mol_cart.verbose = 1

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, grid_response=False, xc='B3LYP', disp=None, tol=1e-6):
    mf = cpu_rks.RKS(mol, xc=xc)
    mf.disp = disp
    mf.direct_scf_tol = 1e-14
    mf.grids.level = grids_level
    mf.grids.prune = None
    mf.grids.small_rho_cutoff = 1e-30
    if mf._numint.libxc.is_nlc(mf.xc):
        mf.nlcgrids.level = nlcgrids_level
    mf.kernel()
    cpu_gradient = pyscf.grad.RKS(mf)
    cpu_gradient.grid_response = grid_response
    g_cpu = cpu_gradient.kernel()

    gpu_gradient = cpu_gradient.to_gpu()
    g_gpu = gpu_gradient.kernel()
    print('|| CPU - GPU ||:', cupy.linalg.norm(g_cpu - g_gpu))
    assert(cupy.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):

    def test_grad_with_grids_response(self):
        print("-----testing DFT gradient with grids response----")
        _check_grad(mol_sph, grid_response=True, tol=1e-5)

    def test_grad_without_grids_response(self):
        print('-----testing DFT gradient without grids response----')
        _check_grad(mol_sph, grid_response=False, tol=1e-5)

    def test_grad_lda(self):
        print("-----LDA testing-------")
        _check_grad(mol_sph, xc='LDA', disp=None, tol=1e-5)

    def test_grad_gga(self):
        print('-----GGA testing-------')
        _check_grad(mol_sph, xc='PBE', disp=None, tol=1e-5)

    def test_grad_hybrid(self):
        print('------hybrid GGA testing--------')
        _check_grad(mol_sph, xc='B3LYP', disp=None, tol=1e-5)

    def test_grad_mgga(self):
        print('-------mGGA testing-------------')
        _check_grad(mol_sph, xc='tpss', disp=None, tol=1e-4)

    def test_grad_rsh(self):
        print('--------RSH testing-------------')
        _check_grad(mol_sph, xc='wb97', disp=None, tol=1e-4)

    def test_grad_nlc(self):
        print('--------nlc testing-------------')
        _check_grad(mol_sph, xc='HYB_MGGA_XC_WB97M_V', disp=None, tol=1e-5)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d3bj(self):
        print('--------- testing RKS with D3BJ ------')
        _check_grad(mol_sph, xc='b3lyp', disp='d3bj', tol=1e-5)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d4(self):
        print('--------- testing RKS with D4 ------')
        _check_grad(mol_sph, xc='b3lyp', disp='d4', tol=1e-5)

    def test_grad_cart(self):
        print('------hybrid GGA Cart testing--------')
        _check_grad(mol_cart, xc='B3LYP', disp=None, tol=1e-5)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu(self):
        mf = gpu_rks.RKS(mol_sph, xc='b3lyp')
        mf.direct_scf_tol = 1e-14
        mf.disp = 'd3bj'
        mf.kernel()

        gpu_gradient = mf.nuc_grad_method()
        g_gpu = gpu_gradient.kernel()
        cpu_gradient = gpu_gradient.to_cpu()
        g_cpu = cpu_gradient.kernel()
        assert cupy.linalg.norm(g_gpu - g_cpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for RKS Gradient")
    unittest.main()
