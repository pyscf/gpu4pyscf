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

import numpy
import cupy as cp
import unittest
from pyscf import gto
from pyscf import scf as pyscf_scf
from gpu4pyscf import scf
from pyscf import lib
import scipy.linalg
from pyscf.x2c import x2c as pyscf_x2c
from gpu4pyscf.x2c import x2c as x2c

def setUpModule():
    global mol
    mol = gto.M(
        verbose = 5,
        # output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    # def test_x2c1e_ghf(self):
    #     myx2c = scf.GHF(mol).x2c1e()
    #     myx2c.with_x2c.xuncontract = False
    #     e_gpu = myx2c.kernel()

    #     myx2c_cpu = pyscf_scf.GHF(mol).x2c1e()
    #     myx2c_cpu.with_x2c.xuncontract = False
    #     e_cpu = myx2c_cpu.kernel()

    #     self.assertAlmostEqual(e_gpu, -76.08176796102066, 9)
    #     self.assertAlmostEqual(e_cpu, e_gpu, 9)

    #     myx2c.with_x2c.xuncontract = True
    #     e_gpu = myx2c.kernel()

    #     myx2c_cpu.with_x2c.xuncontract = True
    #     e_cpu = myx2c_cpu.kernel()
    #     self.assertAlmostEqual(e_gpu, -76.075431226329414, 9)
    #     self.assertAlmostEqual(e_cpu, e_gpu, 9)

    #     myx2c.with_x2c.xuncontract = True
    #     myx2c.with_x2c.approx = 'ATOM1E'
    #     e_gpu = myx2c.kernel()

    #     myx2c_cpu.with_x2c.xuncontract = True
    #     myx2c_cpu.with_x2c.approx = 'ATOM1E'
    #     e_cpu = myx2c_cpu.kernel()
    #     self.assertAlmostEqual(e_gpu, -76.0761147360038, 9)
    #     self.assertAlmostEqual(e_cpu, e_gpu, 9)

    #     myx2c.with_x2c.basis = 'aug-cc-pvqz'
    #     e_gpu = myx2c.kernel()

    #     myx2c_cpu.with_x2c.basis = 'aug-cc-pvqz'
    #     e_cpu = myx2c_cpu.kernel()
    #     self.assertAlmostEqual(e_gpu, -76.0896170536646, 9)
    #     self.assertAlmostEqual(e_cpu, e_gpu, 9)

    # def test_get_xmat_routine_and_get_hcore(self):
    #     myx2c = scf.GHF(mol).x2c1e()
    #     def get_xmat_test(xmol):
    #         zero_matrix = cp.zeros((xmol.nao*2, xmol.nao*2))
    #         return zero_matrix
    #     myx2c.with_x2c.get_xmat = get_xmat_test
    #     h1 = myx2c.with_x2c.get_hcore()
    #     ref = mol.intor('int1e_nuc')
    #     ref = scipy.linalg.block_diag(ref, ref)
    #     self.assertAlmostEqual(abs(h1.get() - ref).max(), 0, 12)

    # def test_undo_x2c(self):
    #     pass

    def test_to_cpu(self):
        myx2c = scf.GHF(mol).x2c1e()
        e_gpu = myx2c.kernel()

        mfx2c_cpu = myx2c.to_cpu()
        e_cpu = mfx2c_cpu.kernel()
        self.assertAlmostEqual(e_cpu, e_gpu, 9)
        

if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()
