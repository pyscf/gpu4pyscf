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
import cupy as cp
import unittest
import pytest
from pyscf import lib
import gpu4pyscf
from gpu4pyscf.df import int3c2e_bdiv as int3c2e
from gpu4pyscf.df.grad.tdrhf import _jk_energy_per_atom
from gpu4pyscf.df.grad import rhf as rhf_grad

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

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
        mf = mol.RHF().density_fit(auxbasis='def2-universal-jkfit').to_gpu()
    else:
        mf = mol.RKS(xc=xc).density_fit(auxbasis='def2-universal-jkfit').to_gpu()
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
    global mol1, mol, auxmol
    mol1 = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=6)

    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
                O3  -.5   -.14     0.5
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]
                     ),
               'C2': 'ccpvdz',
               'O3': 'ccpvdz'}
    )
    auxmol = mol.copy()
    auxmol.basis = {
        'C1':'''
C    S
 50.0000000000           1.0000000000
C    S
 18.338091700            0.60189974570
C    S
  9.5470634000           0.19165883840
C    S
  5.1584143000           1.0000000
C    S
  2.8816701000           1.0000000
C    S
  1.6573522000           1.0000000
C    S
  0.97681020000          1.0000000
C    S
  0.35779270000          1.0000000
C    S
  0.21995500000          1.0000000
C    S
  0.13560770000          1.0000000
C    P
102.9917624900           1.0000000000
 28.1325940100           1.0000000000
  9.8364318200           1.0000000000
C    P
  3.3490545000           1.0000000000
C    P
  1.4947618600           1.0000000000
C    P
  0.4000000000           1.0000000000
C    D
  0.1995412500           1.0000000000 ''',
        'C2':'unc-weigend',
        'O3': [[0, [9.5, 1.]],
              [0, [3.5, 1.]],
              [0, [1.5, 1.]],
              [0, [.8, 1.]],
              [0, [.5, 1.]],
              [0, [.3, 1.]],
              [0, [.2, 1.]],
              [0, [.1, 1.]]
             ],
    }
    auxmol.build()

def tearDownModule():
    global mol1
    mol1.stdout.close()
    del mol1


def benchmark_with_finite_diff(mol_input, delta=0.1, xc='b3lyp', tda=False,
                               tol=1e-5, coords_indices=None):
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
    if coords_indices is None:
        coords_indices = [[0, 2], [2, 1]]
    for i, j in coords_indices:
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

        grad_fdiff = (e_add - e_minus)/(delta*2.0)*0.52917721092
        assert abs(gradient_ana[i, j] - grad_fdiff) < tol
    return gradient_ana


def _check_grad(mol, tol=1e-5, disp=None, tda=False, method="numerical"):
    if method == "cpu":
        raise NotImplementedError("Only benchmark with finite difference")
    elif method == "numerical":
        grad_ana = benchmark_with_finite_diff(
            mol, delta=0.005, xc="hf", tda=tda, tol=tol
        )
    return grad_ana


class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_numerical(self):
        _check_grad(mol1, tol=1e-4, tda=True, method="numerical")

    def test_grad_tdhf_singlet_numerical(self):
        _check_grad(mol1, tol=1e-4, tda=False, method="numerical")

    def test_j_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = 4
        mo_coeff = cp.asarray(np.random.rand(6, nao, nocc)) - .5
        dm = cp.einsum('spi,sqi->spq', mo_coeff, mo_coeff)
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        j_factor = [1, -1, -1, 1, -.5, .5]
        ej = _jk_energy_per_atom(opt, dm, j_factor=j_factor)
        assert abs(ej.sum(axis=0)).max() < 1e-12
        ref = 0
        for i, jfac in enumerate(j_factor):
            ref += rhf_grad._jk_energy_per_atom(opt, dm[i], j_factor=jfac, k_factor=0)
        assert abs(ej - ref).max() < 1e-12
        assert abs(lib.fp(ej) - -5.7379651745047555) < 1e-12

    def test_jk_energy_per_atom(self):
        cp.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.random.rand(3, nao, nocc) - .5
        dm = cp.einsum('spi,sqi->spq', mo_coeff, mo_coeff)
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        j_factor = [1, -1,  0]
        k_factor = [1, -1, -1]
        ejk = _jk_energy_per_atom(opt, dm, j_factor=j_factor, k_factor=k_factor)
        assert abs(ejk.sum(axis=0)).max() < 1e-11
        ref = 0
        for i in range(len(dm)):
            ref += rhf_grad._jk_energy_per_atom(
                opt, dm[i], j_factor=j_factor[i], k_factor=k_factor[i])
        assert abs(ejk - ref).max() < 1e-11

if __name__ == "__main__":
    print("Full Tests for DF TD-RHF Gradient")
    unittest.main()
