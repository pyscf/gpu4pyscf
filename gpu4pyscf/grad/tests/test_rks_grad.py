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
import numpy
import cupy
import unittest
import pytest
from pyscf.dft import rks as cpu_rks
from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.dft import gen_grid
try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

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

def _check_grad(mol, grid_response=False, xc='B3LYP', disp=None, tol=1e-9):
    mf = cpu_rks.RKS(mol, xc=xc).to_gpu()
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
    print('|| CPU - GPU ||:', cupy.linalg.norm(g_cpu - g_gpu))
    assert(cupy.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):

    def test_grad_with_grids_response(self):
        print("-----testing DFT gradient with grids response----")
        _check_grad(mol_sph, grid_response=True, tol=1e-6)
    
    def test_grad_without_grids_response(self):
        print('-----testing DFT gradient without grids response----')
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
        _check_grad(mol_sph, xc='HYB_MGGA_XC_WB97M_V', disp=None, tol=1e-7)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_grad_d3bj(self):
        print('--------- testing RKS with D3BJ ------')
        _check_grad(mol_sph, xc='b3lyp', disp='d3bj')

    def test_grad_d4(self):
        print('--------- testing RKS with D4 ------')
        _check_grad(mol_sph, xc='b3lyp', disp='d4')

    def test_grad_cart(self):
        print('------hybrid GGA Cart testing--------')
        _check_grad(mol_cart, xc='B3LYP', disp=None)
    
    def test_grad_hf(self):
        print('------HF testing--------')
        _check_grad(mol_sph, xc='hf', disp=None)

    def test_grid_response_stratmann(self):
        mol = pyscf.M(
            atom = '''
            Na 0 0 0
            F 1 0 0.1''',
            basis = "def2-svp",
            verbose = 0,
        )
        mf = mol.RKS(xc = "r2scan").to_gpu()
        mf.grids.atom_grid = (10,14)
        mf.grids.prune = None
        mf.grids.becke_scheme = gen_grid.stratmann
        mf.small_rho_cutoff = 1e-30
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # ref_gradient = numpy.empty([mol.natm, 3])
        # def get_e(mol):
        #     mf = mol.RKS(xc = "r2scan")
        #     mf.grids.atom_grid = (10,14)
        #     mf.grids.prune = None
        #     from pyscf.dft import gen_grid as gen_grid_cpu
        #     mf.grids.becke_scheme = gen_grid_cpu.stratmann
        #     mf.small_rho_cutoff = 1e-30
        #     mf.conv_tol = 1e-12
        #     e = mf.kernel()
        #     assert mf.converged
        #     return e

        # dx = 4e-5
        # mol_copy = mol.copy()
        # for i_atom in range(mol.natm):
        #     for i_xyz in range(3):
        #         xyz_p = mol.atom_coords()
        #         xyz_p[i_atom, i_xyz] += dx
        #         mol_copy.set_geom_(xyz_p, unit='Bohr')
        #         mol_copy.build()
        #         e_p = get_e(mol_copy)

        #         xyz_m = mol.atom_coords()
        #         xyz_m[i_atom, i_xyz] -= dx
        #         mol_copy.set_geom_(xyz_m, unit='Bohr')
        #         mol_copy.build()
        #         e_m = get_e(mol_copy)

        #         ref_gradient[i_atom, i_xyz] = (e_p - e_m) / (2 * dx)
        # print(repr(ref_gradient))

        ref_gradient = numpy.array([
            [ 2.03163616e+00, -3.55271368e-09,  3.42115604e-01],
            [-2.03163615e+00,  5.68434189e-09, -3.42115600e-01],
        ])

        assert numpy.max(numpy.abs(test_gradient - ref_gradient)) < 1e-7

    def test_grid_response_no_radii_adjustment(self):
        mol = pyscf.M(
            atom = '''
            Na 0 0 0
            F 1 0 0.1''',
            basis = "def2-svp",
            verbose = 0,
        )
        mf = mol.RKS(xc = "r2scan").to_gpu()
        mf.grids.atom_grid = (10,14)
        mf.grids.prune = None
        mf.grids.becke_scheme = gen_grid.stratmann
        mf.grids.radii_adjust = None
        mf.small_rho_cutoff = 1e-30
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # ref_gradient = numpy.empty([mol.natm, 3])
        # def get_e(mol):
        #     mf = mol.RKS(xc = "r2scan")
        #     mf.grids.atom_grid = (10,14)
        #     mf.grids.prune = None
        #     from pyscf.dft import gen_grid as gen_grid_cpu
        #     mf.grids.becke_scheme = gen_grid_cpu.stratmann
        #     mf.grids.radii_adjust = None
        #     mf.small_rho_cutoff = 1e-30
        #     mf.conv_tol = 1e-12
        #     e = mf.kernel()
        #     assert mf.converged
        #     return e

        # dx = 4e-5
        # mol_copy = mol.copy()
        # for i_atom in range(mol.natm):
        #     for i_xyz in range(3):
        #         xyz_p = mol.atom_coords()
        #         xyz_p[i_atom, i_xyz] += dx
        #         mol_copy.set_geom_(xyz_p, unit='Bohr')
        #         mol_copy.build()
        #         e_p = get_e(mol_copy)

        #         xyz_m = mol.atom_coords()
        #         xyz_m[i_atom, i_xyz] -= dx
        #         mol_copy.set_geom_(xyz_m, unit='Bohr')
        #         mol_copy.build()
        #         e_m = get_e(mol_copy)

        #         ref_gradient[i_atom, i_xyz] = (e_p - e_m) / (2 * dx)
        # print(repr(ref_gradient))

        ref_gradient = numpy.array([
            [ 2.60052754e+00, -2.13162821e-09,  2.66351520e-01],
            [-2.60052754e+00,  0.00000000e+00, -2.66351523e-01],
        ])

        assert numpy.max(numpy.abs(test_gradient - ref_gradient)) < 1e-7

if __name__ == "__main__":
    print("Full Tests for RKS Gradient")
    unittest.main()
