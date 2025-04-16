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
import cupy
import pyscf
from pyscf import lib
from pyscf import scf as cpu_scf
from pyscf import dft as cpu_dft
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import dft as gpu_dft

def setUpModule():
    global mol_sph, mol_cart, mol2
    atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
    bas='def2-qzvpp'

    mol_sph = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.verbose = 0
    mol_sph.build()

    mol_cart = pyscf.M(atom=atom, basis=bas, max_memory=32000, cart=1)
    mol_cart.output = '/dev/null'
    mol_cart.verbose = 0
    mol_cart.build()

    atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
O       0.0000000000    100.0000000000     0.1174000000
H      -0.7570000000    100.0000000000    -0.4696000000
H       0.7570000000    100.0000000000    -0.4696000000
'''
    mol2 = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol2.output = '/dev/null'
    mol2.verbose = 0
    mol2.build()

def tearDownModule():
    global mol_sph, mol_cart, mol2
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    mol2.stdout.close()
    del mol_sph, mol_cart, mol2

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rhf(self):
        mf = gpu_scf.RHF(mol_sph)
        mf.max_cycle = 50
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.abs(e_tot - -76.0667232412) < 1e-5

    def test_rhf_cart(self):
        mf = gpu_scf.RHF(mol_cart)
        mf.max_cycle = 50
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.abs(e_tot - -76.0668120924) < 1e-5

    def test_uhf(self):
        mf = gpu_scf.UHF(mol_sph)
        mf.max_cycle = 50
        mf.conv_tol = 1e-9
        e_gpu = mf.kernel()

        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_uhf_cart(self):
        mf = gpu_scf.UHF(mol_cart)
        mf.max_cycle = 50
        mf.conv_tol = 1e-9
        e_gpu = mf.kernel()

        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_screening(self):
        mf = gpu_scf.RHF(mol2)
        mf.max_cycle = 50
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.abs(e_tot - -76.0667232412 * 2.0) < 1e-5

    def test_to_cpu(self):
        mf = gpu_scf.RHF(mol_sph)
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_scf.hf.RHF)
        assert np.abs(e_cpu - e_gpu) < 1e-5

        mf = gpu_dft.rks.RKS(mol_sph)
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_dft.rks.RKS)
        assert 'gpu' not in mf.grids.__module__
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_to_gpu(self):
        mf = cpu_scf.RHF(mol_sph)
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, gpu_scf.hf.RHF)
        assert np.abs(e_cpu - e_gpu) < 1e-5

        mf = cpu_dft.rks.RKS(mol_sph)
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, gpu_dft.rks.RKS)
        assert 'gpu' in mf.grids.__module__
        assert np.abs(e_cpu - e_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()
