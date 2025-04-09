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


def cal_analytic_gradient(td, tdgrad):
    td.kernel(nstates=5)
    tdgrad.base._scf.with_solvent.tdscf = True
    tdgrad.kernel()
    return tdgrad.de


def cal_td(td, tda):
    td._scf.with_solvent.tdscf = True
    td.kernel()
    return td.e


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
    mf.with_solvent.tdscf = True
    mf.with_solvent.equilibrium_solvation = True
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

    gradient_ana = cal_analytic_gradient(td, tdgrad)
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
                e1 = cal_td(td_add, tda)
                e_add = e1[0] + mf_add.e_tot

                mf_minus = get_new_mf(mol, coords, i, j, -1.0, delta, xc, solvent, unrestrict)
                td_minus = get_td(mf_minus, tda, xc)
                e1 = cal_td(td_minus, tda)
                e_minus = e1[0] + mf_minus.e_tot
                grad[i, j] = (e_add - e_minus)/(delta*2.0)*0.52917721092
        return gradient_ana, grad
    


def _check_grad_numerical(mol, tol=1e-6, xc='hf', disp=None, tda=False, solvent='CPCM', unrestrict=False, num=True):
    grad_ana, grad = benchmark_with_finite_diff(
        mol, delta=0.005, xc=xc, tda=tda, solvent=solvent, unrestrict=unrestrict, num=num)
    if num:
        norm_diff = np.linalg.norm(grad_ana - grad)
        print(norm_diff)
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

    def test_grad_tda_unrestrict_hf_CPCM(self):
        grad_pyscf = _check_grad_numerical(molu, tol=1e-4, xc='hf', tda=True, unrestrict=True, solvent='CPCM')
        print(grad_pyscf)
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


if __name__ == "__main__":
    print("Full Tests for TDHF and TDDFT Gradient with PCM")
    unittest.main()
