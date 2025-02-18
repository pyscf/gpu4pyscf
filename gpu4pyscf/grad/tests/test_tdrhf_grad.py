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

def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e, xy = np.linalg.eig(np.asarray(h))
    sorted_indices = np.argsort(e)
    
    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]
    
    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e, xy = np.linalg.eig(np.asarray(a))
    sorted_indices = np.argsort(e)
    
    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]
    
    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

def tearDownModule():
    global mol
    del mol


def benchmark_with_cpu(mol, nstates=3, lindep=1.0E-12, tda=False):
    mf = scf.RHF(mol).run()
    if tda:
        td = mf.TDA()
    else:
        td = mf.TDHF()
    td.lindep=lindep
    td.nstates=nstates
    td.kernel()

    tdgrad_cpu = pyscf.grad.tdrhf.Gradients(td)
    tdgrad_cpu.kernel()

    td_gpu = td.to_gpu()
    tdgrad_gpu = gpu4pyscf.grad.tdrhf.Gradients(td_gpu)
    tdgrad_gpu.kernel()

    return tdgrad_cpu.de, tdgrad_gpu.de


def benchmark_with_finite_diff(mol_input, delta=0.1, nstates=3, lindep=1.0E-12, tda=False):
    mol = mol_input.copy()
    mf = scf.RHF(mol).to_gpu()
    mf.run()
    if tda:
        td = gpu4pyscf.tdscf.rhf.TDA(mf)
    else:
        td = gpu4pyscf.tdscf.rhf.TDHF(mf)
    td.nstates = nstates
    td.lindep = lindep
    assert td.device == 'gpu'
    td.kernel()

    tdgrad = gpu4pyscf.grad.tdrhf.Gradients(td)
    gradient_ana = tdgrad.kernel()
    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    grad = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            coords_new = coords*1.0
            coords_new[i, j] += delta
            mol.set_geom_(coords_new, unit='Ang')
            mol.build()
            mf_add = scf.RHF(mol).to_gpu()
            mf_add.run()

            if tda:
                td_add = gpu4pyscf.tdscf.rhf.TDA(mf_add)
                a, b = td_add.get_ab()
                e1 = diagonalize_tda(a)[0]
            else:
                td_add = gpu4pyscf.tdscf.rhf.TDHF(mf_add)
                a, b = td_add.get_ab()
                e1 = diagonalize(a, b)[0]
                
            e_add = e1[0] + mf_add.e_tot

            coords_new = coords*1.0
            coords_new[i, j] -= delta
            mol.set_geom_(coords_new, unit='Ang')
            mol.build()
            mf_minus = scf.RHF(mol).to_gpu()
            mf_minus.run()
            
            if tda:
                td_minus = gpu4pyscf.tdscf.rhf.TDA(mf_minus)
                a, b = td_minus.get_ab()
                e1 = diagonalize_tda(a)[0]
            else:
                td_minus = gpu4pyscf.tdscf.rhf.TDHF(mf_minus)
                a, b = td_minus.get_ab()
                e1 = diagonalize(a, b)[0]
                
            e_minus = e1[0] + mf_minus.e_tot
            grad[i, j] = (e_add - e_minus)/(delta*2.0)*0.52917721092

    return gradient_ana, grad


def _check_grad(mol, tol=1e-6, disp=None, tda=False, method='cpu'):
    if method == 'cpu':
        gradi_cpu, grad_gpu = benchmark_with_cpu(mol, nstates=5, lindep=1.0E-12, tda=tda)
        norm_diff = np.linalg.norm(gradi_cpu - grad_gpu)
    elif method == 'numerical':
        grad_ana, grad = benchmark_with_finite_diff(mol, nstates=5, lindep=1.0E-12, tda=tda)
        norm_diff = np.linalg.norm(grad_ana - grad)
    print(f'{tda}  {method}  norm_diff = {norm_diff}')
    assert norm_diff < tol

class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_cpu(self):
        _check_grad(mol, tol=1e-11, tda=True, method='cpu')

    def test_grad_tda_singlet_numerical(self):
        _check_grad(mol, tol=1e-3, tda=True, method='numerical')

    def test_grad_tddft_singlet_cpu(self):
        _check_grad(mol, tol=1e-11, tda=False, method='cpu')

    def test_grad_tddft_singlet_numerical(self):
        _check_grad(mol, tol=1e-3, tda=False, method='numerical')


if __name__ == "__main__":
    print("Full Tests for TD-RHF Gradient")
    unittest.main()
