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

bas0 = "def2svpd"


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


def cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, grad_elec, tda):
    assert hasattr(td._scf, 'with_df')
    a, b = td.get_ab()

    if tda:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_tda(a)
        x = xy_diag[:, 0].reshape(nocc, nvir)*np.sqrt(0.5)
        de_td = grad_elec(tdgrad, (x, 0))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)
    else:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize(a, b)
        nsize = xy_diag.shape[0]//2
        norm_1 = np.linalg.norm(xy_diag[:nsize,0])
        norm_2 = np.linalg.norm(xy_diag[nsize:,0])
        x = xy_diag[:nsize,0]*np.sqrt(0.5/(norm_1**2-norm_2**2))
        y = xy_diag[nsize:,0]*np.sqrt(0.5/(norm_1**2-norm_2**2))
        x = x.reshape(nocc, nvir)
        y = y.reshape(nocc, nvir)
    
        de_td = grad_elec(tdgrad, (x, y))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)

    return gradient_ana


def cal_td(td, tda):
    assert hasattr(td._scf, 'with_df')
    a, b = td.get_ab()
    if tda:
        e1 = diagonalize_tda(a)[0]
    else:
        e1 = diagonalize(a, b)[0]
    return e1


def cal_mf(mol, xc):
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').to_gpu()
    else:
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit').to_gpu()
        mf.grids.level=9
        mf.grids.prune = None
    mf.run()
    return mf


def get_new_mf(mol, coords, i, j, factor, delta, xc):
    coords_new = coords*1.0
    coords_new[i, j] += delta*factor
    mol.set_geom_(coords_new, unit='Ang')
    mol.build()
    mf = cal_mf(mol, xc)
    return mf


def get_td(mf, tda, xc):
    if xc == 'hf':
        if tda:
            td = mf.TDA()
        else:
            td = mf.TDHF()
    else:
        if tda:
            td = mf.TDA()
        else:
            td = mf.TDDFT()

    return td


def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def benchmark_with_finite_diff(mol_input, delta=0.1, xc='b3lyp', tda=False):
    mol = mol_input.copy()
    mf = cal_mf(mol, xc)
    td = get_td(mf, tda, xc)
    tdgrad = td.nuc_grad_method()
    assert hasattr(tdgrad.base._scf, 'with_df')

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ>0).sum())
    nvir = nmo - nocc
    if xc == 'hf':
        grad_elec = gpu4pyscf.grad.tdrhf.grad_elec
    else:
        grad_elec = gpu4pyscf.grad.tdrks.grad_elec
    gradient_ana = cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, grad_elec, tda)

    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    grad = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            mf_add = get_new_mf(mol, coords, i, j, 1.0, delta, xc)
            td_add = get_td(mf_add, tda, xc)
            e1 = cal_td(td_add, tda)
            if e1 is None:
                return None, None
            e_add = e1[0] + mf_add.e_tot

            mf_minus = get_new_mf(mol, coords, i, j, -1.0, delta, xc)
            td_minus = get_td(mf_minus, tda, xc)
            e1 = cal_td(td_minus, tda)
            if e1 is None:
                return None, None
            e_minus = e1[0] + mf_minus.e_tot
            grad[i, j] = (e_add - e_minus)/(delta*2.0)*0.52917721092
    return gradient_ana, grad


def _check_grad(mol, tol=1e-6, disp=None, tda=False, method="numerical"):
    if method == "cpu":
        raise NotImplementedError("Only benchmark with finite difference")
    elif method == "numerical":
        grad_ana, grad = benchmark_with_finite_diff(
            mol, delta=0.005, xc="hf", tda=tda
        )
        norm_diff = np.linalg.norm(grad_ana - grad)
    assert norm_diff < tol


class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_numerical(self):
        _check_grad(mol, tol=1e-4, tda=True, method="numerical")

    def test_grad_tdhf_singlet_numerical(self):
        _check_grad(mol, tol=1e-4, tda=False, method="numerical")


if __name__ == "__main__":
    print("Full Tests for DF TD-RHF Gradient")
    unittest.main()
