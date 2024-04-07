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
import pytest
from pyscf import scf as cpu_scf
from gpu4pyscf import scf as gpu_scf
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

bas0='cc-pvtz'

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.build()
    mol_sph.verbose = 1

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1)
    mol_cart.output = '/dev/null'
    mol_cart.build()
    mol_cart.verbose = 1

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, tol=1e-6, disp=None):
    mf = cpu_scf.hf.RHF(mol)
    mf.direct_scf_tol = 1e-10
    mf.disp = disp
    mf.kernel()

    cpu_gradient = mf.nuc_grad_method()
    g_cpu = cpu_gradient.kernel()

    gpu_gradient = cpu_gradient.to_gpu()
    g_gpu = gpu_gradient.kernel()
    print('|| CPU - GPU ||:', np.linalg.norm(g_cpu - g_gpu))
    assert(np.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):
    def test_grad_rhf(self):
        _check_grad(mol_sph, tol=1e-6)

    def test_grad_cart(self):
        _check_grad(mol_cart, tol=1e-6)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d3bj(self):
        _check_grad(mol_sph, tol=1e-6, disp='d3bj')

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d4(self):
        _check_grad(mol_sph, tol=1e-6, disp='d4')

    def test_to_cpu(self):
        mf = gpu_scf.hf.RHF(mol_sph)
        mf.direct_scf_tol = 1e-10
        mf.disp = 'd3bj'
        mf.kernel()

        gpu_gradient = mf.nuc_grad_method()
        g_gpu = gpu_gradient.kernel()
        cpu_gradient = gpu_gradient.to_cpu()
        g_cpu = cpu_gradient.kernel()
        assert np.linalg.norm(g_gpu - g_cpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for RHF Gradient")
    unittest.main()
