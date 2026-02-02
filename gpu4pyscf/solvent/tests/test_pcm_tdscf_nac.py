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
from pyscf import lib, gto, scf
from gpu4pyscf import tdscf, nac, dft
from gpu4pyscf.solvent.tdscf.pcm import WithSolventTDSCFNacMethod

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


def get_mf(mol, mf, s, mo_coeff, method='CPCM'):
    if isinstance(mf, dft.rks.RKS):
        mf_new = dft.RKS(mol).to_gpu()
        if getattr(mf, 'with_df', None) is not None:
            mf_new = mf_new.density_fit()
        mf_new = mf_new.PCM()
        mf_new.xc = mf.xc
        if len(mf.grids.atom_grid) > 0:
            mf_new.grids.atom_grid = mf.grids.atom_grid
        else:
            mf_new.grids.level = mf.grids.level
        mf_new.with_solvent.method = method
        mf_new.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_new.with_solvent.eps = 78
    else:
        mf_new = scf.RHF(mol).to_gpu()
        if getattr(mf, 'with_df', None) is not None:
            mf_new = mf_new.density_fit()
        mf_new = mf_new.PCM()
        mf_new.with_solvent.method = method
        mf_new.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_new.with_solvent.eps = 78
    mf_new.conv_tol = mf.conv_tol
    mf_new.conv_tol_cpscf = mf.conv_tol_cpscf
    mf_new.max_cycle = mf.max_cycle
    mf_new.kernel()
    assert mf_new.converged
    mo_coeff_new, _ = nac.finite_diff.match_and_reorder_mos(s, mo_coeff, mf_new.mo_coeff)
    mf_new.mo_coeff = mo_coeff_new

    return mf_new


def get_mf_td(mol, mf, s, mo_coeff, method='CPCM'):
    mf_new = get_mf(mol, mf, s, mo_coeff, method=method)
    td_new = mf_new.TDA(equilibrium_solvation=True)
    a, b = td_new.get_ab()
    e_diag, xy_diag = diagonalize_tda(a)

    return mf_new, xy_diag


def get_nacv_ge(td_nac, x_yI, delta=0.001, singlet=True, atmlst=None, method='CPCM'):
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
            mf_add = get_mf(mol_add, mf, s, mo_coeff, method=method)
            mol_minus = nac.finite_diff.get_new_mol(mol, coords, -delta, iatm, icart)
            mf_minus = get_mf(mol_minus, mf, s, mo_coeff, method=method)

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


def get_nacv_ee(td_nac, x_yI, x_yJ, nJ, delta=0.001, singlet=True, atmlst=None, method='CPCM'):
    mf = td_nac.base._scf
    mol = mf.mol
    coords = mol.atom_coords(unit='Ang')*1.0
    natm = coords.shape[0]
    nac_num = np.zeros((natm, 3))
    nac3 = np.zeros((natm, 3))
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    xI, yI = x_yI
    xJ, yJ = x_yJ
    xI = cp.asarray(xI).reshape(nocc, nvir)
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir)
    xJ = cp.asarray(xJ).reshape(nocc, nvir)
    if not isinstance(yJ, np.ndarray) and not isinstance(yJ, cp.ndarray):
        yJ = cp.zeros_like(xJ)
    yJ = cp.asarray(yJ).reshape(nocc, nvir)
    gamma = np.block([[(-xJ@xI.T).get(), np.zeros((nocc, nvir))], 
                    [np.zeros((nvir, nocc)), (xI.T@xJ).get()]]) * 2
    gamma = cp.asarray(gamma)
    gamma_ao = mo_coeff @ gamma @ mo_coeff.T
    s = mol.intor('int1e_ovlp')
    s = cp.asarray(s)
    for iatm in range(natm):
        for icart in range(3):
            mol_add = nac.finite_diff.get_new_mol(mol, coords, delta, iatm, icart)
            mf_add, xy_diag_add = get_mf_td(mol_add, mf, s, mo_coeff, method=method)
            mol_minus = nac.finite_diff.get_new_mol(mol, coords, -delta, iatm, icart)
            mf_minus, xy_diag_minus = get_mf_td(mol_minus, mf, s, mo_coeff, method=method)

            sign1 = 1.0
            sign2 = 1.0
            xJ_add = cp.asarray(xy_diag_add[:, nJ]).reshape(nocc, nvir)*cp.sqrt(0.5)
            xJ_minus = cp.asarray(xy_diag_minus[:, nJ]).reshape(nocc, nvir)*cp.sqrt(0.5)
            if (xJ*xJ_add).sum() < 0.0:
                sign1 = -1.0
            if (xJ*xJ_minus).sum() < 0.0:
                sign2 = -1.0
            
            mo_diff = (mf_add.mo_coeff - mf_minus.mo_coeff)/(delta*2.0)*0.52917721092
            dpq = mo_coeff.T @ s @ mo_diff
            nac_num[iatm, icart] = (gamma*dpq).sum()

            t_diff = (xJ_add*sign1 - xJ_minus*sign2)/(delta*2.0)*0.52917721092
            nac3[iatm, icart] = (xI*t_diff).sum()*2 # for double occupancy
    
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
    return nac_num - nac2 + nac3


class KnownValues(unittest.TestCase):
    def test_nac_tda_singlet_ge_ref_CPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'CPCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0169025342,  0.0000000000, 0.0000000000],
                        [ 0.0465754741, -0.0000000000, 0.0000000000],
                        [ 0.0465754741,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0446961004,  0.0000000000, 0.0000000000],
                               [ 0.1231615358, -0.0000000000, 0.0000000000],
                               [ 0.1231615358,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1891076576,  0.0000000000, 0.0000000000],
                            [ 0.0945538288, -0.0000000000, 0.0000000000],
                            [ 0.0945538288,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5000655383,  0.0000000000, 0.0000000000],
                                   [ 0.2500327691, -0.0000000000, 0.0000000000],
                                   [ 0.2500327691,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_singlet_ee_ref_CPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'CPCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        ref = np.array([[-0.0000000000,  0.1975254546, -0.0000000000],
                        [ 0.0000000000, -0.1052458447,  0.0746679020],
                        [ 0.0000000000, -0.1052458447, -0.0746679020]])
        ref_scaled = np.array([[-0.0000000000,  2.8603062535, -0.0000000000],
                               [ 0.0000000000, -1.5240331860,  1.0812432626],
                               [ 0.0000000000, -1.5240331860, -1.0812432626]])
        ref_etf = np.array([[-0.0000000000,  0.2048354179, -0.0000000000],
                            [ 0.0000000000, -0.1024177089,  0.0756340377],
                            [-0.0000000000, -0.1024177089, -0.0756340377]])
        ref_etf_scaled = np.array([[-0.0000000000,  2.9661596164, -0.0000000000],
                                   [ 0.0000000000, -1.4830798082,  1.0952335814],
                                   [-0.0000000000, -1.4830798082, -1.0952335814]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    @pytest.mark.slow
    def test_nac_tda_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0169683811,  0.0000000000, 0.0000000000],
                        [ 0.0465867533, -0.0000000000, 0.0000000000],
                        [ 0.0465867533,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0448882510,  0.0000000000, 0.0000000000],
                               [ 0.1232408596, -0.0000000000, 0.0000000000],
                               [ 0.1232408596,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1890863180,  0.0000000000, 0.0000000000],
                            [ 0.0945431590, -0.0000000000, 0.0000000000],
                            [ 0.0945431590,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5002100106,  0.0000000000, 0.0000000000],
                                   [ 0.2501050053, -0.0000000000, 0.0000000000],
                                   [ 0.2501050053,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    @pytest.mark.slow
    def test_nac_tda_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        ref = np.array([[-0.0000000000,  0.1975150856,  0.0000000000],
                        [ 0.0000000000, -0.1052419102,  0.0746604968],
                        [ 0.0000000000, -0.1052419102, -0.0746604968]])
        ref_scaled = np.array([[-0.0000000000,  2.8596724344,  0.0000000000],
                               [ 0.0000000000, -1.5237184981,  1.0809532036],
                               [ 0.0000000000, -1.5237184981, -1.0809532036]])
        ref_etf = np.array([[-0.0000000000,  0.2048270177,  0.0000000000],
                            [ 0.0000000000, -0.1024135088,  0.0756260076],
                            [-0.0000000000, -0.1024135088, -0.0756260076]])
        ref_etf_scaled = np.array([[-0.0000000000,  2.9655364013,  0.0000000000],
                                   [ 0.0000000000, -1.4827682006,  1.0949321091],
                                   [-0.0000000000, -1.4827682006, -1.0949321091]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_b3lyp_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='B3LYP').PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[ 0.0256543933,  0.0000000000, 0.0000000000],
                        [-0.0419852335, -0.0000000000, 0.0000000000],
                        [-0.0419852335,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[ 0.0814018791,  0.0000000000, 0.0000000000],
                               [-0.1332199461, -0.0000000000, 0.0000000000],
                               [-0.1332199461,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[ 0.1628978311,  0.0000000000, 0.0000000000],
                            [-0.0814491476, -0.0000000000, 0.0000000000],
                            [-0.0814491476,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[ 0.5168779231,  0.0000000000, 0.0000000000],
                                   [-0.2584396977, -0.0000000000, 0.0000000000],
                                   [-0.2584396977,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_b3lyp_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='B3LYP').PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,4)
        nac1.kernel()
        ref = np.array([[ 0.0000000000, 0.0000000000, 0.0000000000],
                        [ 0.0022087355, 0.0000000000, 0.0000000000],
                        [-0.0022087355, 0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[ 0.0000000000, 0.0000000000, 0.0000000000],
                               [ 0.0130172487, 0.0000000000, 0.0000000000],
                               [-0.0130172487, 0.0000000000, 0.0000000000]])
        ref_etf = np.array([[ 0.0000000000, 0.0000000000, 0.0000000000],
                            [ 0.0019431043, 0.0000000000, 0.0000000000],
                            [-0.0019431043, 0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[ 0.0000000000, 0.0000000000, 0.0000000000],
                                   [ 0.0114517433, 0.0000000000, 0.0000000000],
                                   [-0.0114517433, 0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0169681226,  0.0000000000, 0.0000000000],
                        [ 0.0465865958, -0.0000000000, 0.0000000000],
                        [ 0.0465865958,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0448869136,  0.0000000000, 0.0000000000],
                               [ 0.1232386489, -0.0000000000, 0.0000000000],
                               [ 0.1232386489,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1890906137,  0.0000000000, 0.0000000000],
                            [ 0.0945453068, -0.0000000000, 0.0000000000],
                            [ 0.0945453068,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5002140924,  0.0000000000, 0.0000000000],
                                   [ 0.2501070462, -0.0000000000, 0.0000000000],
                                   [ 0.2501070462,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = scf.RHF(mol).density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        ref = np.array([[-0.0000000000, -0.1974929000,  0.0000000000],
                        [ 0.0000000000,  0.1052307850, -0.0746525780],
                        [ 0.0000000000,  0.1052307850,  0.0746525780]])
        ref_scaled = np.array([[-0.0000000000, -2.8593682346,  0.0000000000],
                               [ 0.0000000000,  1.5235664872, -1.0808449828],
                               [ 0.0000000000,  1.5235664872,  1.0808449828]])
        ref_etf = np.array([[-0.0000000000, -0.2048057004,  0.0000000000],
                            [ 0.0000000000,  0.1024028502, -0.0756178747],
                            [-0.0000000000,  0.1024028502,  0.0756178747]])
        ref_etf_scaled = np.array([[-0.0000000000, -2.9652454034,  0.0000000000],
                                   [ 0.0000000000,  1.4826227017, -1.0948208712],
                                   [-0.0000000000,  1.4826227017,  1.0948208712]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    @pytest.mark.slow
    def test_nac_tda_df_b3lyp_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0256435589,  0.0000000000, 0.0000000000],
                        [ 0.0419823862, -0.0000000000, 0.0000000000],
                        [ 0.0419823862,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0813658125,  0.0000000000, 0.0000000000],
                               [ 0.1332081469, -0.0000000000, 0.0000000000],
                               [ 0.1332081469,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1628922129,  0.0000000000, 0.0000000000],
                            [ 0.0814463385, -0.0000000000, 0.0000000000],
                            [ 0.0814463385,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5168493691,  0.0000000000, 0.0000000000],
                                   [ 0.2584254207, -0.0000000000, 0.0000000000],
                                   [ 0.2584254207,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    @pytest.mark.slow
    def test_nac_tda_df_b3lyp_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,4)
        nac1.kernel()
        ref = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                        [-0.0021934952, 0.0000000000, 0.0000000000],
                        [ 0.0021934952, 0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                               [-0.0129286386, 0.0000000000, 0.0000000000],
                               [ 0.0129286386, 0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                            [-0.0019302468, 0.0000000000, 0.0000000000],
                            [ 0.0019302468, 0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                                   [-0.0113770312, 0.0000000000, 0.0000000000],
                                   [ 0.0113770312, 0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_camb3lyp_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='camb3lyp').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0230996727,  0.0000000000, 0.0000000000],
                        [ 0.0417184380, -0.0000000000, 0.0000000000],
                        [ 0.0417184380,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0726231757,  0.0000000000, 0.0000000000],
                               [ 0.1311588045, -0.0000000000, 0.0000000000],
                               [ 0.1311588045,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1629925985,  0.0000000000, 0.0000000000],
                            [ 0.0814964926, -0.0000000000, 0.0000000000],
                            [ 0.0814964926,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5124332403,  0.0000000000, 0.0000000000],
                                   [ 0.2562172279, -0.0000000000, 0.0000000000],
                                   [ 0.2562172279,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_camb3lyp_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='camb3lyp').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,4)
        nac1.kernel()
        ref = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                        [ 0.0021378449, 0.0000000000, 0.0000000000],
                        [-0.0021378449, 0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                               [ 0.0124442494, 0.0000000000, 0.0000000000],
                               [-0.0124442494, 0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                            [ 0.0018784309, 0.0000000000, 0.0000000000],
                            [-0.0018784309, 0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                                   [ 0.0109342184, 0.0000000000, 0.0000000000],
                                   [-0.0109342184, 0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_tpss_singlet_ge_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='tpss').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()
        ref = np.array([[-0.0278102547,  0.0000000000, 0.0000000000],
                        [ 0.0433664781, -0.0000000000, 0.0000000000],
                        [ 0.0433664781,  0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0868864710,  0.0000000000, 0.0000000000],
                               [ 0.1354881601, -0.0000000000, 0.0000000000],
                               [ 0.1354881601,  0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.1670293601,  0.0000000000, 0.0000000000],
                            [ 0.0835126330, -0.0000000000, 0.0000000000],
                            [ 0.0835126330,  0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.5218431763,  0.0000000000, 0.0000000000],
                                   [ 0.2609151925, -0.0000000000, 0.0000000000],
                                   [ 0.2609151925,  0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    def test_nac_tda_df_tpss_singlet_ee_ref_IEFPCM(self):
        """
        Compared with the reference values.
        """
        mf = dft.RKS(mol, xc='tpss').density_fit().PCM().to_gpu()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        td.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,4)
        nac1.kernel()
        ref = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                        [-0.0001376736, 0.0000000000, 0.0000000000],
                        [ 0.0001376736, 0.0000000000, 0.0000000000]])
        ref_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                               [-0.0007960973, 0.0000000000, 0.0000000000],
                               [ 0.0007960973, 0.0000000000, 0.0000000000]])
        ref_etf = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                            [-0.0002108306, 0.0000000000, 0.0000000000],
                            [ 0.0002108306, 0.0000000000, 0.0000000000]])
        ref_etf_scaled = np.array([[-0.0000000000, 0.0000000000, 0.0000000000],
                                   [-0.0012191278, 0.0000000000, 0.0000000000],
                                   [ 0.0012191278, 0.0000000000, 0.0000000000]])
        assert abs(np.abs(nac1.de)-np.abs(ref)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1.0E-6
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1.0E-6

    @pytest.mark.slow
    def test_nac_tda_singlet_ge_fdiff_CPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_singlet_ee_fdiff_CPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_b3lyp_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="b3lyp").PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_b3lyp_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="b3lyp").PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 3
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,4)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    def test_nac_tda_df_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_df_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = scf.RHF(mol).density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_df_b3lyp_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="b3lyp").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_df_b3lyp_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="b3lyp").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 3
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,4)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_df_camb3lyp_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="camb3lyp").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    def test_nac_tda_df_camb3lyp_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="camb3lyp").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 3
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,4)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_df_tpss_singlet_ge_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="tpss").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ge(nac1, (xI, xI*0.0), delta=delta, method=method)

        td.kernel()
        nac1.states=(0,1)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-5

    @pytest.mark.slow
    def test_nac_tda_df_tpss_singlet_ee_fdiff_IEFPCM(self):
        """
        compare with finite difference
        """
        method = "IEF-PCM"
        mf = dft.RKS(mol, xc="tpss").density_fit().PCM().to_gpu()
        mf.with_solvent.method = method
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf.kernel()
        td = mf.TDA(equilibrium_solvation=True).set(nstates=5)
        nac1 = td.nac_method()
        assert getattr(td, 'with_solvent', None) is not None

        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 3
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        delta = 0.0005
        fdiff_nac = get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, method=method)

        td.kernel()
        nac1.states=(1,4)
        nac1.kernel()
        assert abs(np.abs(np.abs(nac1.de_scaled) - np.abs(fdiff_nac))).max() < 1e-4

    @unittest.skip('PCM-TDA-NAC not available in PySCF')
    def test_from_cpu(self):
        mol = gto.M(atom='H  0.  0.  1.804; F  0.  0.  0.', verbose=0, unit='B')
        nac_cpu = mol.RHF().PCM().TDA(equilibrium_solvation=True).nac_method()
        nac_gpu = nac_cpu.to_gpu()
        assert isinstance(nac_gpu, WithSolventTDSCFNacMethod)
        assert not hasattr(nac_gpu, 'xy')


if __name__ == "__main__":
    print("Full Tests for TD-RHF nonadiabatic coupling vectors between ground and excited states in LR-PCM.")
    unittest.main()
