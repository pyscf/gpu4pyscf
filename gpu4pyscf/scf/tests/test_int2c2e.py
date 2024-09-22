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
from pyscf import lib, df
from pyscf.scf import _vhf
from pyscf.gto.moleintor import getints, make_cintopt
from pyscf.df.grad.rhf import _int3c_wrapper
import numpy as np
import cupy
import unittest

from gpu4pyscf.scf import int2c2e
from gpu4pyscf.lib.cupy_helper import load_library
libgint = load_library('libgint')

'''
compare int2c2e by pyscf and gpu4pyscf
'''

def setUpModule():
    global mol_cart, mol_sph
    mol_cart = pyscf.M(atom='''
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis= 'ccpvdz',
    verbose=1,
    output = '/dev/null')
    mol_cart.build()

    mol_sph = pyscf.M(atom='''
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis= 'ccpvdz',
    verbose=1,
    output = '/dev/null')
    mol_sph.build()

omega = 0.2

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

class KnownValues(unittest.TestCase):
    def test_int2c2e_sph(self):
        nao = mol_sph.nao
        eri = mol_sph.intor('int2c2e_sph').reshape((nao,)*2)
        int2c = int2c2e.get_int2c2e(mol_sph)
        assert np.linalg.norm(eri - int2c.get()) < 1e-10

    def test_int2c2e_cart(self):
        nao = mol_cart.nao
        eri = mol_cart.intor('int2c2e_cart').reshape((nao,)*2)
        int2c = int2c2e.get_int2c2e(mol_cart)
        assert np.linalg.norm(eri - int2c.get()) < 1e-10

if __name__ == "__main__":
    print("Full Tests for int2c")
    unittest.main()
