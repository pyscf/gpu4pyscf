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

import numpy as np
import cupy as cp
import unittest
import pytest
import pyscf
from pyscf import lib, gto
from pyscf import scf as cpu_scf
from gpu4pyscf import scf as gpu_scf
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.grad import rhf as rhf_grad_gpu
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

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, tol=1e-6, disp=None):
    mf = cpu_scf.hf.RHF(mol)
    mf.direct_scf_tol = 1e-14
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
        mf.direct_scf_tol = 1e-14
        mf.disp = 'd3bj'
        mf.kernel()

        gpu_gradient = mf.nuc_grad_method()
        g_gpu = gpu_gradient.kernel()
        cpu_gradient = gpu_gradient.to_cpu()
        g_cpu = cpu_gradient.kernel()
        assert np.linalg.norm(g_gpu - g_cpu) < 1e-5

    def test_jk_energy_per_atom(self):
        mol = pyscf.M(
            atom = '''
            O   0.000   -0.    0.1174
            H  -0.757    4.   -0.4696
            H   0.757    4.   -0.4696
            C   3.      1.    0.
            ''',
            basis='def2-tzvp',
            unit='B',)
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao) - .5
        dm = cp.asarray(dm.dot(dm.T))
        ejk = rhf_grad_gpu._jk_energy_per_atom(mol, dm).get()
        self.assertAlmostEqual(ejk.sum(), 0, 9)
        self.assertAlmostEqual(lib.fp(ejk), 2710.490337642, 9)

        dm = dm.get()
        vj, vk = rhf_grad_cpu.get_jk(mol, dm)
        veff = vj - vk * .5
        ref = np.empty_like(ejk)
        for n, (i0, i1) in enumerate(mol.aoslice_by_atom()[:,2:]):
            ref[n] = np.einsum('xpq,pq->x', veff[:,i0:i1], dm[i0:i1])
        self.assertAlmostEqual(abs(ejk - ref).max(), 0, 9)

    def test_ecp_grad(self):
        mol = gto.M(atom=' H 0 0 1.5; Cu 0 0 0', basis='lanl2dz',
                    ecp='lanl2dz', verbose=0)
        mf = gpu_scf.RHF(mol)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(mol.atom)[1]
        self.assertAlmostEqual(lib.fp(g), 0.012310573162997052, 7)
        
        mfs = mf.as_scanner()
        e1 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0 -0.001'))
        e2 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0  0.001'))
        self.assertAlmostEqual(g[1,2], (e2-e1)/0.002*lib.param.BOHR, 6)

if __name__ == "__main__":
    print("Full Tests for RHF Gradient")
    unittest.main()
