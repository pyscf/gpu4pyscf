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

import numpy as np
import pyscf
from pyscf import scf, dft
from gpu4pyscf import dft, scf
import unittest

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc0='B3LYP'
bas0='def2-tzvpp'
auxbasis0='def2-tzvpp-jkfit'
disp0='d3bj'
grids_level = 6
eps = 1e-3

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol.build(output='/dev/null')
    mol.verbose = 1

def tearDownModule():
    global mol
    del mol

def _check_rhf_hessian(ix=0, iy=0, tol=1e-3):
    pmol = mol.copy()
    pmol.build()
    
    mf = scf.RHF(pmol).density_fit(auxbasis='ccpvtz-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    g = mf.nuc_grad_method()
    g.kernel()
    g_scanner = g.as_scanner()
    hobj = mf.Hessian()
    hobj.set(auxbasis_response=2)
    
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

    pmol.set_geom_(coords, unit='Bohr')
    h = hobj.kernel()
    print(f"analytical Hessian H({ix},{iy})")
    print(h[ix,:,iy,:])
    print(f"finite different Hessian H({ix},{iy})")
    print(h_fd)
    print('Norm of diff', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

def _check_dft_hessian(xc=xc0, disp=None, ix=0, iy=0, tol=1e-3):
    pmol = mol.copy()
    pmol.build()

    mf = dft.rks.RKS(pmol, xc=xc, disp=disp).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-12
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    
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
    pmol.set_geom_(coords, unit='Bohr')
    pmol.build()

    mf = dft.rks.RKS(pmol, xc=xc, disp=disp).density_fit(auxbasis=auxbasis0)
    mf.grids.level = grids_level
    mf.kernel()
    hobj = mf.Hessian()
    hobj.set(auxbasis_response=2)
    hobj.verbose=0
    h = hobj.kernel()
    
    print(f"analytical Hessian H({ix},{iy})")
    print(h[ix,:,iy,:])
    print(f"finite different Hessian H({ix},{iy})")
    print(h_fd)
    print('Norm of diff', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)
    
class KnownValues(unittest.TestCase):
    def test_hessian_rhf(self):
        print('-----testing DF RHF Hessian----')
        _check_rhf_hessian(ix=0, iy=0)
        _check_rhf_hessian(ix=0, iy=1)
    
    def test_hessian_lda(self):
        print('-----testing DF LDA Hessian----')
        _check_dft_hessian(xc='LDA', disp=None, ix=0,iy=0)
        _check_dft_hessian(xc='LDA', disp=None, ix=0,iy=1)
    
    def test_hessian_gga(self):
        print('-----testing DF PBE Hessian----')
        _check_dft_hessian(xc='PBE', disp=None, ix=0,iy=0)
        _check_dft_hessian(xc='PBE', disp=None, ix=0,iy=1)
    
    def test_hessian_hybrid(self):
        print('-----testing DF B3LYP Hessian----')
        _check_dft_hessian(xc='B3LYP', disp=None, ix=0,iy=0)
        _check_dft_hessian(xc='B3LYP', disp=None, ix=0,iy=1)

    def test_hessian_mgga(self):
        print('-----testing DF M06 Hessian----')
        _check_dft_hessian(xc='m06', disp=None, ix=0,iy=0)
        _check_dft_hessian(xc='m06', disp=None, ix=0,iy=1)
    
    def test_hessian_rsh(self):
        print('-----testing DF wb97 Hessian----')
        _check_dft_hessian(xc='wb97', disp=None, ix=0,iy=0)
        _check_dft_hessian(xc='wb97', disp=None, ix=0,iy=1)
    
    def test_hessian_D3(self):
        pmol = mol.copy()
        pmol.build()

        mf = dft.rks.RKS(pmol, xc='B3LYP', disp='d3bj').density_fit(auxbasis=auxbasis0)
        mf.conv_tol = 1e-12
        mf.grids.level = grids_level
        mf.verbose = 1
        mf.kernel()
        
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        hobj.verbose=0
        h = hobj.kernel()

    def test_hessian_D4(self):
        pmol = mol.copy()
        pmol.build()

        mf = dft.rks.RKS(pmol, xc='B3LYP', disp='d4').density_fit(auxbasis=auxbasis0)
        mf.conv_tol = 1e-12
        mf.grids.level = grids_level
        mf.verbose = 1
        mf.kernel()

        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        hobj.verbose=0
        h = hobj.kernel()

if __name__ == "__main__":
    print("Full Tests for DF Hessian")
    unittest.main()
