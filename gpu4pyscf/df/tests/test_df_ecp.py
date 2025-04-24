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

import unittest
import numpy as np
import pyscf
from gpu4pyscf.dft import rks

xc='pbe0'
bas='def2-tzvpp'
auxbasis='def2-tzvpp-jkfit'
grids_level = 6
eps=0.001

def setUpModule():
    global mol
    mol = pyscf.M(
        verbose = 0,
        atom = 'C 0 0 0; O 0.1 1 1.5',
        basis = {'C': 'crenbl', 'O': 'ccpvdz'},
        ecp = {'C': 'crenbl', 'O': 'crenbl'},
        output = '/dev/null'
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    return e_dft

def _make_mf():
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-12
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    return mf

def _check_grad(grid_response=False, tol=1e-5):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.grids.level = grids_level
    mf.conv_tol = 1e-12
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
    print('norm of analytical - finite difference gradient:', np.linalg.norm(g_analy - grad_fd))
    assert(np.linalg.norm(g_analy - grad_fd) < tol)


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
    pmol.set_geom_(coords, unit='Bohr')
    pmol.build()

    print(f'Norm of finite difference - analytical Hessian({ix},{iy})', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

class KnownValues(unittest.TestCase):
    def test_rks_scf(self):
        mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
        mf.conv_tol = 1e-12
        e_tot = mf.kernel()
        assert np.allclose(e_tot, -21.29853214867972)

    def test_rks_gradient(self):
        _check_grad()

    def test_rks_hessian(self):
        mf = _make_mf()
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        hobj.verbose=0
        h = hobj.kernel()

        _check_dft_hessian(mf, h, ix=0, iy=0)
        _check_dft_hessian(mf, h, ix=0, iy=1)

if __name__ == "__main__":
    print("Full Tests for DF ECP")
    unittest.main()
