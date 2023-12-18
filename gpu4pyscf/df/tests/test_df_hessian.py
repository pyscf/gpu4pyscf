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
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol.build(output='/dev/null')
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _make_rhf():
    mf = scf.RHF(mol).density_fit(auxbasis='ccpvtz-jkfit')
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf

def _make_rks(xc, disp=None):
    mf = dft.rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis0)
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
    def test_hessian_rhf(self):
        print('-----testing DF RHF Hessian----')
        mf = _make_rhf()
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_rhf_hessian(mf, h, ix=0, iy=0)
        _check_rhf_hessian(mf, h, ix=0, iy=1)

    def test_hessian_lda(self):
        print('-----testing DF LDA Hessian----')
        mf = _make_rks('LDA')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_gga(self):
        print('-----testing DF PBE Hessian----')
        mf = _make_rks('PBE')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_hybrid(self):
        print('-----testing DF B3LYP Hessian----')
        mf = _make_rks('b3lyp')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_mgga(self):
        print('-----testing DF M06 Hessian----')
        mf = _make_rks('m06')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_rsh(self):
        print('-----testing DF wb97 Hessian----')
        mf = _make_rks('wb97')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

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
        hobj.kernel()

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
        hobj.kernel()

if __name__ == "__main__":
    print("Full Tests for DF Hessian")
    unittest.main()
