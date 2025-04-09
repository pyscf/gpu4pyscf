# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

bas0 = "cc-pvdz"

def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a, b], 
                  [-b.conj(), -a.conj()]])
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


def cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, tda, singlet=True):
    a, b = td.get_ab()

    if tda:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_tda(a)
        x = xy_diag[:, 0].reshape(nocc, nvir) * np.sqrt(0.5)
        de_td = tdgrad.grad_elec((x, 0), singlet)
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)
    else:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize(a, b)
        nsize = xy_diag.shape[0] // 2
        norm_1 = np.linalg.norm(xy_diag[:nsize, 0])
        norm_2 = np.linalg.norm(xy_diag[nsize:, 0])
        x = xy_diag[:nsize, 0] * np.sqrt(0.5 / (norm_1**2 - norm_2**2))
        y = xy_diag[nsize:, 0] * np.sqrt(0.5 / (norm_1**2 - norm_2**2))
        x = x.reshape(nocc, nvir)
        y = y.reshape(nocc, nvir)

        de_td = tdgrad.grad_elec((x, y), singlet)
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)

    return gradient_ana


def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def benchmark_with_cpu(mol, nstates=3, lindep=1.0e-12, tda=False):
    mf = scf.RHF(mol).to_gpu().run()
    if tda:
        td = mf.TDA()
    else:
        td = mf.TDHF()

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ>0).sum())
    nvir = nmo - nocc

    tdgrad_gpu = gpu4pyscf.grad.tdrhf.Gradients(td)
    gpu_gradient = cal_analytic_gradient(mol, td, tdgrad_gpu, nocc, nvir, tda)

    tdgrad_cpu = tdgrad_gpu.to_cpu()
    cpu_gradient = cal_analytic_gradient(mol, td, tdgrad_cpu, nocc, nvir, tda)

    return cpu_gradient, gpu_gradient


def benchmark_with_finite_diff(
        mol_input, delta=0.1, nstates=3, lindep=1.0e-12, tda=False):
    mol = mol_input.copy()
    mf = scf.RHF(mol).to_gpu()
    mf.run()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    if tda:
        td = gpu4pyscf.tdscf.rhf.TDA(mf)
    else:
        td = gpu4pyscf.tdscf.rhf.TDHF(mf)
    assert td.device == "gpu"
    tdgrad = td.Gradients()
    gradient_ana = cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, tda)

    coords = mol.atom_coords(unit="Ang") * 1.0
    natm = coords.shape[0]
    grad = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            coords_new = coords * 1.0
            coords_new[i, j] += delta
            mol.set_geom_(coords_new, unit="Ang")
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

            coords_new = coords * 1.0
            coords_new[i, j] -= delta
            mol.set_geom_(coords_new, unit="Ang")
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
            grad[i, j] = (e_add - e_minus) / (delta * 2.0) * 0.52917721092

    return gradient_ana, grad


def _check_grad(mol, tol=1e-6, lindep=1.0E-12, disp=None, tda=False, method="cpu"):
    if method == "cpu":
        gradi_cpu, grad_gpu = benchmark_with_cpu(
            mol, nstates=5, lindep=lindep, tda=tda
        )
        norm_diff = np.linalg.norm(gradi_cpu - grad_gpu)
    elif method == "numerical":
        grad_gpu, grad = benchmark_with_finite_diff(
            mol, delta=0.005, nstates=5, lindep=lindep, tda=tda
        )
        norm_diff = np.linalg.norm(grad_gpu - grad)
    assert norm_diff < tol
    return grad_gpu


class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_cpu(self):
        grad_gpu = _check_grad(mol, tol=1e-10, tda=True, method="cpu")

        grad_cpu, grad_gpu = benchmark_with_cpu(
            mol, nstates=5, lindep=1e-12, tda=True
        )
        ref = np.array([[ 6.1708732035531e-16,  5.4527883622350e-15, 1.0228273246502e-01],
                        [-5.6967766850923e-16,  6.9672215647416e-02, -5.1141366232508e-02],
                        [-4.7409651846241e-17, -6.9672215647420e-02, -5.1141366232510e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_tda_singlet_numerical(self):
        _check_grad(mol, tol=1e-4, tda=True, method="numerical")

    def test_grad_tdhf_singlet_cpu(self):
        grad_gpu = _check_grad(mol, tol=1e-10, lindep=1.0E-6, tda=False, method="cpu")
        ref = np.array([[-3.4653829069609e-16,  2.3748317799310e-14, 1.0506609371536e-01],
                        [ 5.6250300822602e-16,  7.1515265578103e-02, -5.2533046857670e-02],
                        [-2.1596471752992e-16, -7.1515265578123e-02, -5.2533046857686e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5


if __name__ == "__main__":
    print("Full Tests for TD-RHF Gradient")
    unittest.main()
