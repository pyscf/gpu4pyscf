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
import pytest
import numpy as np
import cupy as cp
import pyscf
import gpu4pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "321g"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output='/dev/null', verbose=1)


def tearDownModule():
    global mol 
    del mol


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


def get_mf(mol, mf, s, mo_coeff):
    if isinstance(mf, dft.rks.RKS):
        mf_new = dft.RKS(mol)
        mf_new.xc = mf.xc
        if len(mf.grids.atom_grid) > 0:
            mf_new.grids.atom_grid = mf.grids.atom_grid
        else:
            mf_new.grids.level = mf.grids.level
    else:
        mf_new = scf.RHF(mol).PCM().to_gpu()
        mf_new.with_solvent.method = 'CPCM'
        mf_new.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_new.with_solvent.eps = 1.78
    if getattr(mf, 'with_df', None) is not None:
        mf_new = mf_new.density_fit()
    mf_new.conv_tol = mf.conv_tol
    mf_new.conv_tol_cpscf = mf.conv_tol_cpscf
    mf_new.max_cycle = mf.max_cycle
    mf_new.kernel()
    assert mf_new.converged
    mo_coeff_new, _ = nac.finite_diff.match_and_reorder_mos(s, mo_coeff, mf_new.mo_coeff)
    mf_new.mo_coeff = mo_coeff_new

    return mf_new


def get_nacv_ge(td_nac, x_yI, delta=0.001, with_ris=False, singlet=True, atmlst=None):
    mf = td_nac.base._scf
    mol = mf.mol

    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    nac_fdiff = np.zeros((natm, 3))
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir)
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = yI.reshape(nocc, nvir)

    gamma = np.block([[np.zeros((nocc, nocc)), xI.get()], 
                      [(xI.T*0.0).get(), np.zeros((nvir, nvir))]])
    gamma = cp.asarray(gamma)*2
    gamma_ao = mo_coeff @ gamma @ mo_coeff.T
    s = mol.intor('int1e_ovlp')
    s = cp.asarray(s)

    for iatm in range(natm):
        for icart in range(3):
            mol_add = nac.finite_diff.get_new_mol(mol, coords, delta, iatm, icart)
            mf_add = get_mf(mol_add, mf, s, mo_coeff)
            mol_minus = nac.finite_diff.get_new_mol(mol, coords, -delta, iatm, icart)
            mf_minus = get_mf(mol_minus, mf, s, mo_coeff)

            mo_diff = (mf_add.mo_coeff - mf_minus.mo_coeff)/(delta*2.0)*0.52917721092
            dpq = mo_coeff.T @ s @ mo_diff
            nac_fdiff[iatm, icart] = (gamma*dpq).sum()

    nac2 = np.zeros((natm, 3))
    atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    s12_deriv = mol.intor('int1e_ipovlp')
    s12_deriv = cp.asarray(s12_deriv)
    for k, ia in enumerate(atmlst): 
        shl0, shl1, p0, p1 = offsetdic[ia]
        s12_deriv_tmp = s12_deriv*1.0
        ds1_tmp = s12_deriv_tmp.transpose(0,2,1)
        ds1_tmp[:,:,:p0] = 0
        ds1_tmp[:,:,p1:] = 0
        nac2[k] = cp.einsum('xij,ij->x', ds1_tmp, gamma_ao).get()
    return nac_fdiff - nac2


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet_qchem(self):
        """
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'CPCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 1.78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=False).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[ 0.0227621423,  0.0000000000, 0.0000000000],
                        [-0.0475441111, -0.0000000000, 0.0000000000],
                        [-0.0475441111,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[ 0.0624782171,  0.0000000000, 0.0000000000],
                               [-0.1305005151, -0.0000000000, 0.0000000000],
                               [-0.1305005151,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[  0.1870579287,  0.0000000000, 0.0000000000],
                            [ -0.0935289644, -0.0000000000, 0.0000000000],
                            [ -0.0935289644,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[ 0.5134422640,  0.0000000000, 0.0000000000],
                                   [-0.2567211320, -0.0000000000, 0.0000000000],
                                   [-0.2567211320,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    # @pytest.mark.slow
    def test_nac_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'CPCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 1.78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=False).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)

        td.kernel()
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5


    # def test_nac_tdhf_singlet_qchem(self):
    #     """
    #     benchmark from Qchem
    #     $rem
    #     JOBTYPE              sp          
    #     METHOD               hf       
    #     BASIS                cc-pvdz     
    #     CIS_N_ROOTS          5       
    #     CIS_SINGLETS         TRUE        
    #     CIS_TRIPLETS         FALSE       
    #     SYMMETRY             FALSE       
    #     SYM_IGNORE           TRUE   
    #     SCF_CONVERGENCE      14
    #     XC_GRID 000099000590
    #     RPA True
    #     BASIS_LIN_DEP_THRESH 12
    #     CIS_DER_NUMSTATE   3
    #     CALC_NAC           true
    #     $end

    #     $derivative_coupling
    #     0 is the reference state
    #     0 1 2
    #     $end
    #     """
    #     mf = scf.RHF(mol).to_gpu()
    #     mf.kernel()
    #     td = mf.TDHF().set(nstates=5)
    #     td.kernel()
    #     nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
    #     nac1.states=(1,0)
    #     nac1.kernel()
    #     ref = np.array([[ -0.037645,  0.000000,  0.000000],
    #                     [ -0.093950, -0.000000, -0.000000],
    #                     [ -0.093950,  0.000000, -0.000000]])
    #     ref_etf_scaled = np.array([[ 0.399489,  0.000000,  0.000000],
    #                                [-0.199744, -0.000000, -0.000000],
    #                                [-0.199744,  0.000000, -0.000000]])
    #     ref_etf = np.array([[ 0.134429,  0.000000,  0.000000],
    #                         [-0.067214, -0.000000, -0.000000],
    #                         [-0.067214,  0.000000, -0.000000]])
    #     assert abs(np.abs(nac1.de/td.e[0]) - np.abs(ref)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    #     nac1.states=(2,0)
    #     nac1.kernel()
    #     ref = np.array([[ -0.000000,  0.000000,  0.000000],
    #                     [  0.095909, -0.000000, -0.000000],
    #                     [ -0.095909,  0.000000, -0.000000]])
    #     ref_etf_scaled = np.array([[ 0.000000,  0.000000,  0.000000],
    #                                [ 0.262906, -0.000000, -0.000000],
    #                                [-0.262906,  0.000000, -0.000000]])
    #     ref_etf = np.array([[ 0.000000,  0.000000,  0.000000],
    #                         [ 0.105513, -0.000000, -0.000000],
    #                         [-0.105513,  0.000000, -0.000000]])
    #     assert abs(np.abs(nac1.de/td.e[1]) - np.abs(ref)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
    #     assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4


if __name__ == "__main__":
    print("Full Tests for TD-RHF nonadiabatic coupling vectors between ground and excited states in LR-PCM.")
    unittest.main()
