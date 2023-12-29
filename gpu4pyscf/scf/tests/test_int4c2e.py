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

from gpu4pyscf.scf import int4c2e
from gpu4pyscf.lib.cupy_helper import load_library
libgint = load_library('libgint')

'''
compare int4c2e by pyscf and gpu4pyscf
'''

def setUpModule():
    global mol, ao_labels
    mol = pyscf.M(atom='''
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis= 'sto3g',
    verbose=1,
    output = '/dev/null')
    mol.build()
    ao_labels = mol.ao_labels()

omega = 0.2

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_int4c2e(self):
        nao = mol.nao
        eri = mol.intor('int2e_sph').reshape((nao,)*4)
        int4c = int4c2e.get_int4c2e(mol)
        int4c = int4c + int4c.transpose([0,1,3,2])
        int4c = int4c + int4c.transpose([1,0,2,3])
        assert np.linalg.norm(eri - int4c.get()) < 1e-10

    def test_int4c2e_jk(self):
        nao = mol.nao
        dm = cupy.random.rand(nao,nao)
        dm = dm + dm.T
        eri = mol.intor('int2e_sph').reshape((nao,)*4)
        vj0 = np.einsum('ijkl,kl->ij', eri, dm.get())
        vk0 = np.einsum('ijkl,jl->ik', eri, dm.get())

        vj, vk = int4c2e.get_int4c2e_jk(mol, dm)
        assert np.linalg.norm(vj.get() - vj0) < 1e-8
        assert np.linalg.norm(vk.get() - vk0) < 1e-8

if __name__ == "__main__":
    print("Full Tests for int4c")
    unittest.main()