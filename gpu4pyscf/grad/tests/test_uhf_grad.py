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
import numpy as np
import unittest
import pytest
from pyscf import lib, gto
from gpu4pyscf import scf
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
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1, spin=2,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, tol=1e-6, disp=None):
    mf = scf.uhf.UHF(mol)
    mf.direct_scf_tol = 1e-14
    mf.disp = disp
    mf.kernel()

    gpu_gradient = mf.nuc_grad_method()
    g_gpu = gpu_gradient.kernel()

    cpu_gradient = gpu_gradient.to_cpu()
    g_cpu = cpu_gradient.kernel()
    print('|| CPU - GPU ||:', np.linalg.norm(g_cpu - g_gpu))
    assert(np.linalg.norm(g_cpu - g_gpu) < tol)

class KnownValues(unittest.TestCase):
    def test_grad_uhf(self):
        print('---- testing UHF -------')
        _check_grad(mol_sph, tol=1e-10)

    def test_grad_cart(self):
        print('---- testing UHF Cart -------')
        _check_grad(mol_cart, tol=1e-10)

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d3bj(self):
        print('---- testing UHF with D3(BJ) ----')
        _check_grad(mol_sph, tol=1e-6, disp='d3bj')

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_grad_d4(self):
        print('------- UHF with D4 -----')
        _check_grad(mol_sph, tol=1e-6, disp='d4')

    def test_grad_ecp(self):
        mol = gto.M(atom=' H 0 0 1.5; Cu 0 0 0', basis='lanl2dz',
                    ecp='lanl2dz', verbose=0)
        mf = scf.RHF(mol)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(mol.atom)[1]
        self.assertAlmostEqual(lib.fp(g), 0.012310573162997052, 7)
        
        mfs = mf.as_scanner()
        e1 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0 -0.001'))
        e2 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0  0.001'))
        self.assertAlmostEqual(g[1,2], (e2-e1)/0.002*lib.param.BOHR, 6)

    def test_to_cpu(self):
        mf = scf.uhf.UHF(mol_sph)
        mf.direct_scf_tol = 1e-14
        mf.disp = 'd3bj'
        mf.kernel()

        gpu_gradient = mf.nuc_grad_method()
        g_gpu = gpu_gradient.kernel()
        cpu_gradient = gpu_gradient.to_cpu()
        g_cpu = cpu_gradient.kernel()
        assert np.linalg.norm(g_gpu - g_cpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for UHF Gradient")
    unittest.main()
