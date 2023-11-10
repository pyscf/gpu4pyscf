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

import unittest
import numpy as np
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.solvent import pcm
from gpu4pyscf.solvent.grad import pcm as pcm_grad

def setUpModule():
    global mol, epsilon, lebedev_order, eps, xc, tol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 35.9
    lebedev_order = 29
    eps = 1e-4
    xc = 'B3LYP'
    tol = 1e-4

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_hess_cpcm(self):
        pmol = mol.copy()
        pmol.build()

        mf = dft.rks.RKS(pmol, xc=xc).density_fit().PCM()
        mf.with_solvent.eps = epsilon
        mf.with_solvent.lebedev_order = lebedev_order
        mf.conv_tol = 1e-12
        mf.grids.atom_grid = (99,590)
        mf.verbose = 0
        mf.kernel()

        g = mf.nuc_grad_method()
        g.auxbasis_response = True
        g.kernel()
        g_scanner = g.as_scanner()

        ix = 0
        iy = 1
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

        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()

        print(f"analytical Hessian H({ix},{iy})")
        print(h[ix,:,iy,:])
        print(f"finite different Hessian H({ix},{iy})")
        print(h_fd)
        print('Norm of diff', np.linalg.norm(h[ix,:,iy,:] - h_fd))
        assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

    def test_hess_iefpcm(self):
        pmol = mol.copy()
        pmol.build()

        mf = dft.rks.RKS(pmol, xc=xc).density_fit().PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = epsilon
        mf.with_solvent.lebedev_order = lebedev_order
        mf.conv_tol = 1e-12
        mf.grids.atom_grid = (99,590)
        mf.verbose = 0
        mf.kernel()

        g = mf.nuc_grad_method()
        g.auxbasis_response = True
        g.kernel()
        g_scanner = g.as_scanner()

        ix = 0
        iy = 1
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

        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()

        print(f"analytical Hessian H({ix},{iy})")
        print(h[ix,:,iy,:])
        print(f"finite different Hessian H({ix},{iy})")
        print(h_fd)
        print('Norm of diff', np.linalg.norm(h[ix,:,iy,:] - h_fd))
        assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)
if __name__ == "__main__":
    print("Full Tests for Hessian of PCMs")
    unittest.main()