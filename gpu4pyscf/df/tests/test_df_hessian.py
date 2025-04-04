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

import numpy as np
import pyscf
from gpu4pyscf import dft, scf
import unittest

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
eps = 1e-3

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _make_rhf(mol, disp=None):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.kernel()
    return mf

def _make_uhf(mol, disp=None):
    mf = scf.UHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.kernel()
    return mf

def _make_rks(mol, xc, disp=None):
    mf = dft.rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    return mf

def _make_uks(mol, xc, disp=None):
    mf = dft.uks.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    return mf

def _check_rhf_hessian(mf, h, ix=0, iy=0, tol=1e-3):
    pmol = mf.mol.copy()
    pmol.build()

    g = mf.nuc_grad_method()
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    pmol.build()
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    pmol.build()
    _, g1 = g_scanner(pmol)
    h_fd = (g0 - g1)/2.0/eps

    print(f'Norm of analytical - finite difference Hessian H({ix},{iy})', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

def _check_dft_hessian(mf, h, ix=0, iy=0, tol=1e-3):
    pmol = mf.mol.copy()
    pmol.build()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    pmol.build()
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    pmol.build()
    _, g1 = g_scanner(pmol)

    h_fd = (g0 - g1)/2.0/eps
    print(f'Norm of analytical - finite difference Hessian H({ix},{iy})', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

class KnownValues(unittest.TestCase):
    def test_hessian_rhf(self, disp=None):
        print('-----testing DF RHF Hessian----')
        mf = _make_rhf(mol_sph)
        mf.disp = disp
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_rhf_hessian(mf, h, ix=0, iy=0)
        _check_rhf_hessian(mf, h, ix=0, iy=1)
    
    def test_hessian_lda(self, disp=None):
        print('-----testing DF LDA Hessian----')
        mf = _make_rks(mol_sph, 'LDA')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_gga(self):
        print('-----testing DF PBE Hessian----')
        mf = _make_rks(mol_sph, 'PBE')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_hybrid(self):
        print('-----testing DF B3LYP Hessian----')
        mf = _make_rks(mol_sph, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_uks_hessian_hybrid(self):
        print('-----testing DF-UKS B3LYP Hessian----')
        mf = _make_uks(mol_sph, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_mgga(self):
        print('-----testing DF M06 Hessian----')
        mf = _make_rks(mol_sph, 'm06')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
    
    def test_hessian_rsh(self):
        print('-----testing DF wb97 Hessian----')
        mf = _make_rks(mol_sph, 'wb97')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
    
    def test_hessian_rhf_D3(self):
        print('----- testing DFRHF with D3BJ ------')
        mf = _make_rhf(mol_sph, disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rhf_D4(self):
        print('------ testing DFRKS with D4 --------')
        mf = _make_rhf(mol_sph, disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uhf_D3(self):
        print('-------- testing DFUHF with D3BJ ------')
        mf = _make_uhf(mol_sph, disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uhf_D4(self):
        print('--------- testing DFUHF with D4 -------')
        mf = _make_uhf(mol_sph, disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_D3(self):
        print('--------- testing DFRKS with D3BJ -------')
        mf = _make_rks(mol_sph, 'b3lyp', disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_D4(self):
        print('----------- testing DFRKS with D4 --------')
        mf = _make_rks(mol_sph, 'b3lyp', disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_D3(self):
        print('------------ testing DFUKS with D3BJ -------')
        mf = _make_uks(mol_sph, 'b3lyp', disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_D4(self):
        print('------------- testing DFUKS with D4 ---------')
        mf = _make_uks(mol_sph, 'b3lyp', disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_wb97m_d3bj(self):
        print('----------- testing DFRKS, wb97m-d3bj --------')
        mf = _make_rks(mol_sph, 'wb97m-d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_wb97m_d3bj(self):
        print('------------- testing DFUKS, wb97m-d3bj ---------')
        mf = _make_uks(mol_sph, 'wb97m-d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_cart(self):
        print('-----testing DF Hessian (cartesian)----')
        mf = _make_rks(mol_cart, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_uks_cart(self):
        print('-----testing DF Hessian (cartesian)----')
        mf = _make_uks(mol_cart, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
    
    def test_hessian_qz(self):
        mol = pyscf.M(atom=atom, basis='def2-qzvpp', max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-14
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()

        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
        mol.stdout.close()
        
if __name__ == "__main__":
    print("Full Tests for DF Hessian")
    unittest.main()
