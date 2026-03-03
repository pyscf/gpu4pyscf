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
from pyscf import scf, dft, tdscf, lib
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
import gpu4pyscf.tdscf.ris as ris

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a, b], [-b.conj(), -a.conj()]])
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


def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def benchmark_with_finite_diff(
        mol_input, xc, delta=0.1, nstates=3, lindep=1.0e-12, tda=False,
        with_df=False, tol=1e-5, coords_indices=None):
    
    mol = mol_input.copy()
    
    if with_df:
        mf = dft.RKS(mol, xc=xc).density_fit().to_gpu()
    else:
        mf = dft.RKS(mol, xc=xc).to_gpu()
    mf.grids.level=9
    mf.grids.prune = None
    mf.run()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ>0).sum())
    nvir = nmo - nocc
    if tda:
        td = ris.TDA(mf=mf.to_gpu(), nstates=5, single=False)
    else:
        td = ris.TDDFT(mf=mf.to_gpu(), nstates=5, single=False)
    td.conv_tol = 1.0E-8
    td.lindep=lindep
    td.Ktrunc = 0.0
    td.single = False
    td.nstates = nstates
    tdgrad = td.nuc_grad_method()
    gradient_ana = cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, gpu4pyscf.grad.tdrks_ris.grad_elec, tda)

    coords = mol.atom_coords(unit='Ang')*1.0
    if coords_indices is None:
        coords_indices = [[0, 2], [2, 1]]
    for i, j in coords_indices:
        coords_new = coords*1.0
        coords_new[i, j] += delta
        mol.set_geom_(coords_new, unit='Ang')
        mol.build()
        if with_df:
            mf_add = dft.RKS(mol, xc=xc).density_fit().to_gpu()
        else:
            mf_add = dft.RKS(mol, xc=xc).to_gpu()
        mf_add.grids.level=9
        mf_add.grids.prune = None
        mf_add.run()
        if tda:
            td_add = ris.TDA(mf=mf_add.to_gpu(), nstates=5, single=False)
        else:
            td_add = ris.TDDFT(mf=mf_add.to_gpu(), nstates=5, single=False)
        td_add.conv_tol = 1.0E-8
        td_add.single = False
        td_add.Ktrunc = 0.0
        a, b = td_add.get_ab()
        if tda:
            e1 = diagonalize_tda(a)[0]
        else:
            e1 = diagonalize(a, b)[0]
        e_add = e1[0] + mf_add.e_tot

        coords_new = coords*1.0
        coords_new[i, j] -= delta
        mol.set_geom_(coords_new, unit='Ang')
        mol.build()
        if with_df:
            mf_minus = dft.RKS(mol, xc=xc).density_fit().to_gpu()
        else:
            mf_minus = dft.RKS(mol, xc=xc).to_gpu()
        mf_minus.grids.level=9
        mf_minus.grids.prune = None
        mf_minus.run()
        if tda:
            td_minus = ris.TDA(mf=mf_minus.to_gpu(), nstates=5, single=False)
        else:
            td_minus = ris.TDDFT(mf=mf_minus.to_gpu(), nstates=5, single=False)
        td_minus.conv_tol = 1.0E-8
        td_minus.single = False
        td_minus.Ktrunc = 0.0
        a, b = td_minus.get_ab()
        if tda:
            e1 = diagonalize_tda(a)[0]
        else:
            e1 = diagonalize(a, b)[0]
        e_minus = e1[0] + mf_minus.e_tot

        grad_fdiff = (e_add - e_minus)/(delta*2.0)*0.52917721092
        assert abs(gradient_ana[i, j] - grad_fdiff) < tol
    return gradient_ana


def _check_grad(mol, xc, tol=1e-5, lindep=1.0e-12, disp=None, tda=False, with_df=False):
    grad_gpu = benchmark_with_finite_diff(
        mol, xc, delta=0.005, nstates=5, lindep=lindep, tda=tda,
        with_df=with_df, tol=tol)
    return grad_gpu


class KnownValues(unittest.TestCase):

    @pytest.mark.slow
    def test_grad_b3lyp_tda_singlet(self):
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()

        td = ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()
        g = td.nuc_grad_method()
        g.kernel()
        assert abs(lib.fp(g.de) - -0.16752744269930692) < 1e-4

    def test_grad_b3lyp_tda_singlet_df(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()

        td2 = ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td2.conv_tol = 1.0E-4
        td2.Ktrunc = 0.0
        td2.kernel()
        g2 = td2.nuc_grad_method()
        g2.kernel()

        assert abs(lib.fp(g2.de) - -0.16752744269930692) < 1e-4

    @pytest.mark.slow
    def test_grad_b3lyp_tda_singlet_df_num(self):
        _check_grad(mol, xc="b3lyp", tol=1e-4, tda=True, with_df=True)


if __name__ == "__main__":
    print("Full Tests for TD-RKS RIS Gradient")
    unittest.main()
