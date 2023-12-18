# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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
import pyscf
import cupy
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.dft import numint

def setUpModule():
    global mol_sph, mol_cart
    atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
'''
    mol_sph = pyscf.M(
        atom=atom,
        basis='ccpvdz',
        spin=None,
        cart = 0,
        output = '/dev/null')

    mol_cart = pyscf.M(
        atom=atom,
        basis='ccpvqz',
        spin=None,
        cart=1,
        output = '/dev/null')

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

class KnownValues(unittest.TestCase):

    # sph mol
    def test_ao_sph_deriv0(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv0', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_sph, coords, deriv=0)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv1(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv1', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_sph, coords, deriv=1)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv2(self):
        coords = np.random.random((4,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv2', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_sph, coords, deriv=2)
        #idx = cupy.argwhere(cupy.abs(ao_gpu - ao_cpu) > 1e-10)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv3(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv3', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_sph, coords, deriv=3)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv4(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv4', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_sph, coords, deriv=4)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    # cart mol
    def test_ao_cart_deriv0(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv0', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_cart, coords, deriv=0)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv1(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv1', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_cart, coords, deriv=1)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv2(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv2', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_cart, coords, deriv=2)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv3(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv3', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = ni.eval_ao(mol_cart, coords, deriv=3)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv4(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv4', coords)
        ao_cpu = cupy.asarray(ao)
        ni = NumInt(xc='LDA')
        ao_gpu = numint.eval_ao(ni, mol_cart, coords, deriv=4)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
