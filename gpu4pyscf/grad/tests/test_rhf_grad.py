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
import numpy as np
import unittest
import gpu4pyscf
from pyscf import scf

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0='cc-pvtz'

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol
    
def _check_grad(tol=1e-6):
    mf = scf.hf.RHF(mol)
    mf.direct_scf_tol = 1e-10
    mf.kernel()

    cpu_gradient = pyscf.grad.RHF(mf)
    g_cpu = cpu_gradient.kernel()
    
    # TODO: use to_gpu function
    mf.__class__ = gpu4pyscf.scf.hf.RHF
    gpu_gradient = gpu4pyscf.grad.RHF(mf)
    g_gpu = gpu_gradient.kernel()
    assert(np.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):
    def test_grad_rhf(self):
        _check_grad(tol=1e-6)
    
if __name__ == "__main__":
    print("Full Tests for Gradient")
    unittest.main()
