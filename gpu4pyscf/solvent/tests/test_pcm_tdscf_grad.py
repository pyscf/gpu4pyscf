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

import unittest
import numpy as np
import cupy as cp
import pyscf
import gpu4pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "def2svp"
bas1 = "631g"


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


def diagonalize_u(a, b, nroots=5):
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    b_aa = b_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    b_ab = b_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    b_bb = b_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = np.block([[ a_aa  , a_ab],
                     [ a_ab.T, a_bb]])
    b = np.block([[ b_aa  , b_ab],
                     [ b_ab.T, b_bb]])
    abba = np.asarray(np.block([[a        , b       ],
                                      [-b.conj(),-a.conj()]]))
    e, xy = np.linalg.eig(abba)
    sorted_indices = np.argsort(e)
    
    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]
    
    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def diagonalize_tda_u(a, nroots=5):
    a_aa, a_ab, a_bb = a
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = np.block([[ a_aa  , a_ab],
                     [ a_ab.T, a_bb]])
    e, xy = np.linalg.eig(a)
    sorted_indices = np.argsort(e)
    
    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]
    
    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, grad_elec, tda):
    a, b = td.get_ab()
    assert hasattr(tdgrad.base._scf, 'with_solvent')

    if tda:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_tda(a)
        x = xy_diag[:, 0].reshape(nocc, nvir)*np.sqrt(0.5)
        de_td = grad_elec((x, 0))
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
    
        de_td = grad_elec((x, y))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)

    return gradient_ana


def cal_analytic_gradient_u(mol, td, tdgrad, nocc_a, nvir_a, nocc_b, nvir_b, grad_elec, tda):
    a, b = td.get_ab()

    if tda:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_tda_u(a)
        nsize = nocc_a*nvir_a
        norm1 = np.linalg.norm(xy_diag[:nsize, 0])
        norm2 = np.linalg.norm(xy_diag[nsize:, 0])
        x_aa = xy_diag[:nsize, 0].reshape(nocc_a, nvir_a)*np.sqrt(1/(norm1**2+norm2**2))
        x_bb = xy_diag[nsize:, 0].reshape(nocc_b, nvir_b)*np.sqrt(1/(norm1**2+norm2**2))
        de_td = grad_elec(((x_aa, x_bb), (0, 0)))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)
    else:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_u(a, b)
        nsize1 = nocc_a*nvir_a
        nsize2 = nocc_b*nvir_b
        norm_1 = np.linalg.norm(xy_diag[: nsize1,0])
        norm_2 = np.linalg.norm(xy_diag[nsize1: nsize1+nsize2,0])
        norm_3 = np.linalg.norm(xy_diag[nsize1+nsize2: nsize1+nsize2+nsize1,0])
        norm_4 = np.linalg.norm(xy_diag[nsize1+nsize2+nsize1: ,0])
        norm_factor = np.sqrt(1/(norm_1**2 + norm_2**2 - norm_3**2 - norm_4**2))
        x_aa = xy_diag[: nsize1,0]*norm_factor
        x_bb = xy_diag[nsize1: nsize1+nsize2,0]*norm_factor
        y_aa = xy_diag[nsize1+nsize2: nsize1+nsize2+nsize1,0]*norm_factor
        y_bb = xy_diag[nsize1+nsize2+nsize1: ,0]*norm_factor
        x_aa = x_aa.reshape(nocc_a, nvir_a)
        x_bb = x_bb.reshape(nocc_b, nvir_b)
        y_aa = y_aa.reshape(nocc_a, nvir_a)
        y_bb = y_bb.reshape(nocc_b, nvir_b)
        x = (x_aa, x_bb)
        y = (y_aa, y_bb)
    
        de_td = grad_elec((x, y))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)

    return gradient_ana


def cal_td(td, tda, unrestrict):
    a, b = td.get_ab()
    if unrestrict:
        if tda:
            e1 = diagonalize_tda_u(a)[0]
        else:
            e1 = diagonalize_u(a, b)[0]
    else:
        if tda:
            e1 = diagonalize_tda(a)[0]
        else:
            e1 = diagonalize(a, b)[0]
    return e1


def cal_mf(mol, xc, solvent, unrestrict):
    if unrestrict:
        if xc == 'hf':
            mf = scf.UHF(mol).PCM().to_gpu()
        else:
            mf = dft.UKS(mol, xc=xc).PCM().to_gpu()
            mf.grids.atom_grid = (99,590)
            mf.grids.prune = None
    else:
        if xc == 'hf':
            mf = scf.RHF(mol).PCM().to_gpu()
        else:
            mf = dft.RKS(mol, xc=xc).PCM().to_gpu()
            mf.grids.atom_grid = (99,590)
            mf.grids.prune = None
    mf.with_solvent.method = solvent
    mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
    mf.with_solvent.eps = 78
    mf.run()
    return mf


def get_new_mf(mol, coords, i, j, factor, delta, xc, solvent, unrestrict):
    coords_new = coords*1.0
    coords_new[i, j] += delta*factor
    mol.set_geom_(coords_new, unit='Ang')
    mol.build()
    mf = cal_mf(mol, xc, solvent, unrestrict)
    return mf


def get_td(mf, tda, xc):
    if xc == 'hf':
        if tda:
            td = mf.TDA(equilibrium_solvation=True)
        else:
            td = mf.TDHF(equilibrium_solvation=True)
    else:
        if tda:
            td = mf.TDA(equilibrium_solvation=True)
        else:
            td = mf.TDDFT(equilibrium_solvation=True)

    return td


def setUpModule():
    global mol, molu
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)
    molu = pyscf.M(
        atom=atom, charge=1, spin=1, basis=bas1, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol, molu
    del mol, molu


def benchmark_with_finite_diff(mol_input, delta=0.1, xc='b3lyp', tda=False, solvent='CPCM', unrestrict=False, num=True):
    mol = mol_input.copy()
    mf = cal_mf(mol, xc, solvent, unrestrict)
    td = get_td(mf, tda, xc)
    tdgrad = td.nuc_grad_method()
    grad_elec = tdgrad.grad_elec

    if unrestrict:
        mo_occ = mf.mo_occ
        occidxa = np.where(mo_occ[0]>0)[0]
        occidxb = np.where(mo_occ[1]>0)[0]
        viridxa = np.where(mo_occ[0]==0)[0]
        viridxb = np.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        gradient_ana =  cal_analytic_gradient_u(mol, td, tdgrad, nocca, nvira, noccb, nvirb, grad_elec, tda)
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        nocc = int((mo_occ>0).sum())
        nvir = nmo - nocc
        gradient_ana = cal_analytic_gradient(mol, td, tdgrad, nocc, nvir, grad_elec, tda)
    if not num:
        return gradient_ana, None
    if num:
        coords = mol.atom_coords(unit='Ang')*1.0
        natm = coords.shape[0]
        grad = np.zeros((natm, 3))
        for i in range(natm):
            for j in range(3):
                mf_add = get_new_mf(mol, coords, i, j, 1.0, delta, xc, solvent, unrestrict)
                td_add = get_td(mf_add, tda, xc)
                e1 = cal_td(td_add, tda, unrestrict)
                e_add = e1[0] + mf_add.e_tot

                mf_minus = get_new_mf(mol, coords, i, j, -1.0, delta, xc, solvent, unrestrict)
                td_minus = get_td(mf_minus, tda, xc)
                e1 = cal_td(td_minus, tda, unrestrict)
                e_minus = e1[0] + mf_minus.e_tot
                grad[i, j] = (e_add - e_minus)/(delta*2.0)*0.52917721092
        return gradient_ana, grad
    


def _check_grad_numerical(mol, tol=1e-6, xc='hf', disp=None, tda=False, solvent='CPCM', unrestrict=False, num=True):
    grad_ana, grad = benchmark_with_finite_diff(
        mol, delta=0.005, xc=xc, tda=tda, solvent=solvent, unrestrict=unrestrict, num=num)
    if num:
        norm_diff = np.linalg.norm(grad_ana - grad)
        assert norm_diff < tol
    return grad_ana


class KnownValues(unittest.TestCase):
    def test_grad_tda_singlet_hf_CPCM(self):
        """
        $rem
        JOBTYPE              force          
        METHOD               hf     
        BASIS                def2-svp             
        CIS_N_ROOTS          5      
        CIS_STATE_DERIV      1 
        CIS_SINGLETS         TRUE        
        CIS_TRIPLETS         FALSE       
        SYMMETRY             FALSE       
        SYM_IGNORE           TRUE   
        ! RPA 2
        BASIS_LIN_DEP_THRESH 12
        SOLVENT_METHOD PCM
        $end

        $PCM
        Theory CPCM
        HeavyPoints 302
        HPoints 302
        $end

        $solvent
        dielectric 78
        $end
         -- total gradient after adding PCM contribution --
        ---------------------------------------------------
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       0.000000       0.089607
        2      -0.000000       0.067883      -0.044803
        3       0.000000      -0.067883      -0.044803
        ---------------------------------------------------
        """
        grad_pyscf = _check_grad_numerical(mol, tol=1e-4, xc='hf', tda=True, solvent='CPCM')
        ref = np.array([[-0.000000,  0.000000,  0.089607],
                        [ 0.000000,  0.067883, -0.044803],
                        [ 0.000000, -0.067883, -0.044803]])
        norm_diff = np.linalg.norm(grad_pyscf - ref)
        assert norm_diff < 1e-5

    def test_grad_tda_singlet_b3lyp_CPCM(self):
        grad_pyscf = _check_grad_numerical(mol, tol=1e-4, xc='b3lyp', tda=True, solvent='CPCM')
        ref = np.array([[-0.000000,  0.000000,  0.106714],
                        [ 0.000000,  0.073132, -0.053357],
                        [ 0.000000, -0.073132, -0.053357]])
        norm_diff = np.linalg.norm(grad_pyscf - ref)
        assert norm_diff < 2e-5

    def test_grad_tda_singlet_b3lyp_IEPPCM(self):
        _check_grad_numerical(mol, tol=1e-4, xc='b3lyp', tda=True, solvent='IEFPCM')

    def test_grad_tda_singlet_b3lyp_COSMO(self):
        _check_grad_numerical(mol, tol=1e-4, xc='b3lyp', tda=True, solvent='COSMO')

    def test_grad_tda_singlet_b3lyp_ssvpe(self):
        _check_grad_numerical(mol, tol=5e-4, xc='b3lyp', tda=True, solvent='ss(v)pe')

    def test_grad_tda_unrestrict_hf_CPCM(self):
        grad_pyscf = _check_grad_numerical(molu, tol=1e-4, xc='hf', tda=True, unrestrict=True, solvent='CPCM')
        ref = np.array([[-0.000000,  0.000000, -0.066532],
                        [ 0.000000,  0.073344,  0.033266],
                        [ 0.000000, -0.073344,  0.033266]])
        norm_diff = np.linalg.norm(grad_pyscf - ref)
        assert norm_diff < 2e-5

    def test_grad_tda_unrestrict_b3lyp_CPCM(self):
        grad_pyscf = _check_grad_numerical(molu, tol=1e-4, xc='b3lyp', tda=True, unrestrict=True, solvent='CPCM')
        ref = np.array([[-0.000000,  0.000000, -0.037576],
                        [ 0.000000,  0.083399,  0.018788],
                        [ 0.000000, -0.083399,  0.018788]])
        norm_diff = np.linalg.norm(grad_pyscf - ref)
        assert norm_diff < 2e-5

    def test_grad_tda_unrestrict_b3lyp_IEFPCM(self):
        _check_grad_numerical(molu, tol=1e-4, xc='b3lyp', tda=True, unrestrict=True, solvent='IEFPCM')

    def test_grad_tda_unrestrict_b3lyp_ssvpe(self):
        _check_grad_numerical(molu, tol=8e-4, xc='b3lyp', tda=True, unrestrict=True, solvent='ss(v)pe')


if __name__ == "__main__":
    print("Full Tests for TDHF and TDDFT Gradient with PCM")
    unittest.main()
