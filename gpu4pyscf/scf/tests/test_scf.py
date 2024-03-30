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
import cupy
import pyscf
from pyscf import lib
from pyscf import scf as cpu_scf
from pyscf import dft as cpu_dft
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import dft as gpu_dft

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-qzvpp'
def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.verbose = 0
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rhf(self):
        mf = gpu_scf.RHF(mol)
        mf.max_cycle = 10
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.allclose(e_tot, -76.0667232412)

    def test_to_cpu(self):
        mf = gpu_scf.RHF(mol)
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_scf.hf.RHF)
        assert np.allclose(e_cpu, e_gpu)

        mf = gpu_dft.rks.RKS(mol)
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_dft.rks.RKS)
        assert 'gpu' not in mf.grids.__module__
        assert np.allclose(e_cpu, e_gpu)

    def test_to_gpu(self):
        mf = cpu_scf.RHF(mol)
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, gpu_scf.hf.RHF)
        assert np.allclose(e_cpu, e_gpu)

        mf = cpu_dft.rks.RKS(mol)
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, gpu_dft.rks.RKS)
        assert 'gpu' in mf.grids.__module__
        assert np.allclose(e_cpu, e_gpu)

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()
