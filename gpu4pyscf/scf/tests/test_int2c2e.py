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
