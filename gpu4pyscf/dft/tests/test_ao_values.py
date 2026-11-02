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
import cupy
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.dft import numint

def setUpModule():
    global mol_sph, mol_cart
    atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
'''

    basis = ('ccpvqz', 
             [[5, [1.2, 1.]], 
               [6, [1.2, 1.]], 
               [7, [1.6, 1.]],
               [8, [1.3, 1.]]
               ])
    
    mol_sph = pyscf.M(
        atom=atom,
        basis=basis,
        spin=None,
        cart = 0,
        output = '/dev/null')

    mol_cart = pyscf.M(
        atom=atom,
        basis=basis,
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
        ao_gpu = numint.eval_ao(mol_sph, coords, deriv=0)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8
        
    def test_ao_sph_deriv1(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv1', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_sph, coords, deriv=1)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv2(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv2', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_sph, coords, deriv=2)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv3(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv3', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_sph, coords, deriv=3)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_sph_deriv4(self):
        coords = np.random.random((100,3))
        ao = mol_sph.eval_gto('GTOval_sph_deriv4', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_sph, coords, deriv=4)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    # cart mol
    def test_ao_cart_deriv0(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv0', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_cart, coords, deriv=0)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv1(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv1', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_cart, coords, deriv=1)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv2(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv2', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_cart, coords, deriv=2)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv3(self):
        coords = np.random.random((1000,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv3', coords)
        ao_cpu = cupy.asarray(ao)
        #ni = NumInt()
        ao_gpu = numint.eval_ao(mol_cart, coords, deriv=3)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

    def test_ao_cart_deriv4(self):
        coords = np.random.random((100,3))
        ao = mol_cart.eval_gto('GTOval_cart_deriv4', coords)
        ao_cpu = cupy.asarray(ao)
        ao_gpu = numint.eval_ao(mol_cart, coords, deriv=4)
        assert cupy.linalg.norm(ao_cpu - ao_gpu) < 1e-8

if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
