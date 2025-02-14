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
from pyscf import scf, dft, tdscf
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

bas0='cc-pvtz'

benchmark_results = {}

def setUpModule():
    global mol_sph, mol_cart
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def finite_difference_test(mol, tda=False):

    delta=1.0E-6

    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    grad = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            coords_new = coords*1.0
            coords_new[i, j] += delta
            mol.set_geom_(coords_new, unit='Ang')
            mol.build()

            mf_add = scf.RHF(mol).run().to_gpu()
            if tda:
                td_add = gpu4pyscf.tdscf.rhf.TDA(mf_add)
            else:
                td_add = gpu4pyscf.tdscf.rhf.TDHF(mf_add)
            td_add.nroots = 5
            e1 = td_add.kernel()[0]
            e_add = e1[0] + mf_add.e_tot

            coords_new = coords*1.0
            coords_new[i, j] -= delta
            mol.set_geom_(coords_new, unit='Ang')
            mol.build()

            mf_minus = scf.RHF(mol).run().to_gpu()
            if tda:
                td_minus = gpu4pyscf.tdscf.rhf.TDA(mf_minus)
            else:
                td_minus = gpu4pyscf.tdscf.rhf.TDHF(mf_minus)
            td_minus.nroots = 5
            e1 = td_minus.kernel()[0]
            e_minus = e1[0] + mf_minus.e_tot

            grad[i, j] = (e_add - e_minus)/(delta*2.0)*0.52917721092
    return grad
    

def calculate_benchmark(mol, tol=1e-6, disp=None, method='cpu', tda=False):
    
    key = (id(mol), tol, disp, method.lower(), tda)
    if key in benchmark_results:
        return benchmark_results[key]
    
    g_cpu = 0.0
    if method.lower() == 'cpu':
        mf_cpu = scf.hf.RHF(mol)
        mf_cpu.direct_scf_tol = 1e-14
        mf_cpu.disp = disp
        mf_cpu.kernel()
        if tda:
            postmf = tdscf.rhf.TDA(mf_cpu).run()
        else:
            postmf = tdscf.rhf.TDHF(mf_cpu).run()
        g = postmf.nuc_grad_method()
        g_cpu = g.kernel(state=1)
    else:
        g_cpu = finite_difference_test(mol, tda=tda)

    mf_gpu = gpu_scf.hf.RHF(mol)
    mf_gpu.direct_scf_tol = 1e-14
    mf_gpu.disp = disp
    mf_gpu.kernel()
    if tda:
        postmf = gpu4pyscf.tdscf.rhf.TDA(mf_gpu).run()
    else:
        postmf = gpu4pyscf.tdscf.rhf.TDHF(mf_gpu).run()
    g_gpu = postmf.kernel()

    norm_diff = np.linalg.norm(g_cpu - g_gpu)
    print('|| CPU - GPU ||:', norm_diff)

    benchmark_results[key] = norm_diff
    return norm_diff

def _check_grad(mol, tol=1e-6, disp=None, method='cpu'):
    norm_diff = calculate_benchmark(mol, tol, disp, method)
    assert norm_diff < tol

class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_cpu(self):
        _check_grad(mol_sph, tol=1e-6)

    def test_grad_tda_singlet_numerical(self):
        _check_grad(mol_sph, tol=1e-6)

    def test_grad_tddft_cart_singlet_cpu(self):
        _check_grad(mol_cart, tol=1e-6)

    def test_grad_tddft_cart_singlet_numerical(self):
        _check_grad(mol_cart, tol=1e-6)

if __name__ == "__main__":
    print("Full Tests for TD-RHF Gradient")
    unittest.main()
