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
from pyscf.dft import rks
import gpu4pyscf
from gpu4pyscf.dft import numint

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0='def2-tzvpp'
grids_level = 5
nlcgrids_level = 3
def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_grad(grid_response=False, xc='B3LYP', disp='d3bj', tol=1e-6):
    mf = rks.RKS(mol, xc=xc)
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


    # TODO: use to_gpu functionality
    mf.__class__ = gpu4pyscf.dft.rks.RKS
    mf._numint = numint.NumInt(xc=xc)
    mf.grids = gpu4pyscf.dft.gen_grid.Grids(mol)
    mf.grids.level = grids_level
    mf.grids.prune = None
    mf.grids.small_rho_cutoff = 1e-30
    mf.grids.build()
    if mf._numint.libxc.is_nlc(mf.xc):
        mf.nlcgrids = gpu4pyscf.dft.gen_grid.Grids(mol)
        mf.nlcgrids.level = nlcgrids_level
        mf.nlcgrids.build()

    gpu_gradient = gpu4pyscf.grad.RKS(mf)
    gpu_gradient.grid_response = grid_response
    g_gpu = gpu_gradient.kernel()
    assert(cupy.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):

    def test_grad_with_grids_response(self):
        print("-----testing DFT gradient with grids response----")
        _check_grad(grid_response=True, tol=1e-5)

    def test_grad_without_grids_response(self):
        print('-----testing DFT gradient without grids response----')
        _check_grad(grid_response=False, tol=1e-5)

    def test_grad_lda(self):
        print("-----LDA testing-------")
        _check_grad(xc='LDA', disp=None, tol=1e-5)

    def test_grad_gga(self):
        print('-----GGA testing-------')
        _check_grad(xc='PBE', disp=None, tol=1e-5)

    def test_grad_hybrid(self):
        print('------hybrid GGA testing--------')
        _check_grad(xc='B3LYP', disp=None, tol=1e-5)

    def test_grad_mgga(self):
        print('-------mGGA testing-------------')
        _check_grad(xc='m06', disp=None, tol=1e-4)

    def test_grad_rsh(self):
        print('--------RSH testing-------------')
        _check_grad(xc='wb97', disp=None, tol=1e-4)

    def test_grad_nlc(self):
        print('--------nlc testing-------------')
        _check_grad(xc='HYB_MGGA_XC_WB97M_V', disp=None, tol=1e-5)

if __name__ == "__main__":
    print("Full Tests for Gradient")
    unittest.main()
