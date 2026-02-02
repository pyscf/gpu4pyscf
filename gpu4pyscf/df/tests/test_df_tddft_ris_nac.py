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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf import tdscf, nac
import pytest

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "def2tzvp"

def setUpModule():
    global mol, molpbe
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=6)
    molpbe = pyscf.M(
        atom=atom, basis="ccpvdz", max_memory=32000, output="/dev/null", verbose=6)


def tearDownModule():
    global mol
    global molpbe
    mol.stdout.close()
    molpbe.stdout.close()
    del mol, molpbe


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


class KnownValues(unittest.TestCase):
    def test_nac_pbe_tdaris_singlet_vs_ref_ge(self):
        mf = molpbe.RKS(xc="pbe").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.Ktrunc = 0.0
        nac_ris = td_ris.nac_method()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[nstate])

        ref_e = np.array([0.25933835, 0.33439277, 0.35638221, 0.42592415, 0.51762665])
        ref_de = np.array(
            [[-5.90903417e-05,  7.14375913e-17, -2.26866400e-15],
             [ 2.60385843e-02,  9.51909004e-16, -2.78277910e-16],
             [ 2.60385843e-02, -1.05669301e-15, -1.34098234e-15],])
        ref_de_etf = np.array(
            [[-1.06809321e-01, -2.75316895e-17, -1.66754809e-15],
             [ 5.34045261e-02,  8.11192182e-16, -2.83779869e-16],
             [ 5.34045261e-02, -9.49822786e-16, -1.37984533e-15],])

        # compare with previous calculation resusts
        assert np.linalg.norm(e_diag - ref_e) < 1.0E-8
        assert np.linalg.norm(np.abs(ana_nac[0]) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(ana_nac[2]) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe0_tddftris_singlet_vs_ref_ge(self):
        mf = mol.RKS(xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 7.10616608e-04,  1.67751530e-17,  2.43082576e-15],
             [-2.01248594e-02, -1.27707824e-15,  2.20490394e-15],
             [-2.01248611e-02,  5.34947467e-16,  1.53819879e-15],])
        ref_de_etf = np.array(
            [[ 1.06135079e-01,  2.11139079e-16,  2.05232306e-15],
             [-5.30675522e-02,  3.70362138e-16,  1.20929274e-15],
             [-5.30675585e-02, -7.92538762e-16,  5.86483960e-16],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_camb3lyp_tdaris_singlet_vs_ref_ge(self):
        mf = mol.RKS(xc="camb3lyp").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[-4.89770318e-04, -4.82312005e-16,  1.17207473e-15],
             [ 1.90733178e-02, -8.52029551e-16, -2.58105953e-16],
             [ 1.90733197e-02,  1.54498000e-15,  2.32205946e-15],])
        ref_de_etf = np.array(
            [[-1.01768244e-01, -1.60217510e-16,  4.40680529e-16],
             [ 5.08839991e-02, -8.82169442e-16,  4.39921371e-17],
             [ 5.08840061e-02,  1.56981811e-15,  2.64021576e-15],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe_tda_singlet_vs_ref_ee(self):
        mf = molpbe.RKS(xc="pbe").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.Ktrunc = 0.0
        nac_ris = td_ris.nac_method()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # excited-excited state
        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])

        ref_e = np.array([0.25933835, 0.33439277, 0.35638221, 0.42592415, 0.51762665])
        ref_de = np.array(
            [[ 8.34605908e-17, -1.18122143e-01, -1.38959236e-14],
             [ 8.91217037e-16,  6.74132293e-02, -4.46138124e-02],
             [-9.93589447e-16,  6.74132293e-02,  4.46138124e-02],])
        ref_de_etf = np.array(
            [[ 8.84485527e-17, -1.23821678e-01, -1.40671554e-14],
             [ 8.27342873e-16,  6.19105543e-02, -4.58616923e-02],
             [-9.17992771e-16,  6.19105543e-02,  4.58616923e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(e_diag - ref_e) < 1.0E-8
        assert np.linalg.norm(np.abs(ana_nac[0]) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(ana_nac[2]) - np.abs(ref_de_etf)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_pbe_tda_singlet_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = mol.RKS(xc="pbe").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[nstate])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4.0E-3
        
        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_pbe0_tda_singlet_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = mol.RKS(xc="pbe0").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[nstate])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4.0E-3

        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5

    def test_nac_pbe0_tddft_singlet_vs_ref_ee(self):
        mf = mol.RKS(xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()

        ref_de = np.array(
            [[-1.28630229e-16, -1.01018827e-01, -1.39860434e-09],
             [-4.70672025e-16,  5.75872007e-02, -3.81043336e-02],
             [-1.16131845e-16,  5.75872029e-02,  3.81043350e-02],])
        ref_de_etf = np.array(
            [[-1.35556786e-16, -1.00688286e-01, -1.46281252e-09],
             [-3.65451480e-16,  5.03441246e-02, -3.95286117e-02],
             [-1.79485516e-16,  5.03441269e-02,  3.95286131e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_camb3lyp_tddft_singlet_vs_ref_ee(self):
        mf = mol.RKS(xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 1.39495980e-14, -9.05064872e-02, -2.38916839e-09],
             [ 6.45116755e-16,  5.26412496e-02, -3.38915479e-02],
             [ 1.62215844e-15,  5.26412531e-02,  3.38915503e-02],])
        ref_de_etf = np.array(
            [[ 1.40478550e-14, -9.01595513e-02, -2.48094447e-09],
             [ 6.25487158e-16,  4.50797680e-02, -3.54158761e-02],
             [ 1.58165609e-15,  4.50797717e-02,  3.54158785e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5


if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RKS-ris nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
