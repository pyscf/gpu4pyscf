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
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf
import pytest


atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "ccpvdz"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
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


class KnownValues(unittest.TestCase):
    def test_nac_pbe_tda_singlet_qchem(self):
        """
        $rem
        JOBTYPE              sp
        METHOD               pbe
        BASIS                cc-pvdz
        CIS_N_ROOTS          5
        CIS_SINGLETS         TRUE
        CIS_TRIPLETS         FALSE
        SYMMETRY             FALSE
        SYM_IGNORE           TRUE
        XC_GRID 000099000590
        ! RPA 2
        BASIS_LIN_DEP_THRESH 12
        CIS_DER_NUMSTATE   4
        CALC_NAC           true
        $end

        $derivative_coupling
        0 is the reference state
        0 1 2 4
        $end

        ---------------------------------------------------
        CIS derivative coupling without ETF
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       1.581396       0.000000
        2       0.000000      -0.898277       0.592227
        3       0.000000      -0.898277      -0.592227
        ---------------------------------------------------
        ---------------------------------------------------
        CIS Force Matrix Element
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       0.113255       0.000000
        2       0.000000      -0.056628       0.042045
        3       0.000000      -0.056628      -0.042045
        ---------------------------------------------------
        ---------------------------------------------------
        CIS derivative coupling with ETF
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       1.648487       0.000000
        2       0.000000      -0.824243       0.611981
        3       0.000000      -0.824243      -0.611981
        ---------------------------------------------------
        """
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        nac1.states=(1,2)
        nac1.kernel()
        ref_scaled = np.array([[-0.000000,  1.581396,  0.000000],
                               [ 0.000000, -0.898277,  0.592227],
                               [ 0.000000, -0.898277, -0.592227]])
        ref_etf = np.array([[-0.000000,  0.113255,  0.000000],
                            [ 0.000000, -0.056628,  0.042045],
                            [ 0.000000, -0.056628, -0.042045]])
        ref_etf_scaled = np.array([[-0.000000,  1.648487,  0.000000],
                                   [ 0.000000, -0.824243,  0.611981],
                                   [ 0.000000, -0.824243, -0.611981]])
        assert abs(np.abs(nac1.de/(td.e[1] - td.e[0])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

        nac1.states=(2,4)
        nac1.kernel()
        ref_scaled = np.array([[-0.898200, -0.000000, -0.000000],
                               [ 0.428663, -0.000000, -0.000000],
                               [ 0.428663,  0.000000,  0.000000]])
        ref_etf = np.array([[-0.070461, -0.000000, -0.000000],
                            [ 0.035231, -0.000000, -0.000000],
                            [ 0.035231,  0.000000,  0.000000]])
        ref_etf_scaled = np.array([[-0.776323, -0.000000, -0.000000],
                                   [ 0.388162, -0.000000, -0.000000],
                                   [ 0.388162,  0.000000,  0.000000]])
        assert abs(np.abs(nac1.de/(td.e[3] - td.e[1])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    @pytest.mark.slow
    def test_nac_pbe_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = td.nac_method()
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4e-3

        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    @pytest.mark.slow
    def test_nac_pbe0_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = td.nac_method()
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-3

        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    @pytest.mark.slow
    def test_nac_b3lyp_tddft_singlet_qchem(self):
        """
        benchmark from qchem
        """
        mf = dft.rks.RKS(mol, xc="b3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        nac1.states=(1,2)
        nac1.kernel()
        ref_scaled = np.array([[-0.000000,  1.666697, -0.000000],
                               [ 0.000000, -0.939801,  0.624083],
                               [ 0.000000, -0.939801, -0.624083]])
        ref_etf = np.array([[-0.000000,  0.118801, -0.000000],
                            [ 0.000000, -0.059400,  0.044126],
                            [ 0.000000, -0.059400, -0.044126]])
        ref_etf_scaled = np.array([[-0.000000,  1.734397, -0.000000],
                                   [ 0.000000, -0.867198,  0.644200],
                                   [ 0.000000, -0.867198, -0.644200]])
        assert abs(np.abs(nac1.de/(td.e[1] - td.e[0])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    def test_nac_df_b3lyp_tddft_singlet_qchem(self):
        mf = dft.rks.RKS(mol, xc="b3lyp").to_gpu().density_fit()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(2,3)
        nac1.kernel()
        ref_scaled = np.array([[ 0.000000, -0.000000, -0.000000],
                               [-0.024525, -0.000000, -0.000000],
                               [ 0.024525,  0.000000,  0.000000]])
        ref_etf = np.array([[ 0.000000, -0.000000, -0.000000],
                            [-0.000371, -0.000000, -0.000000],
                            [ 0.000371,  0.000000,  0.000000]])
        ref_etf_scaled = np.array([[ 0.000000, -0.000000, -0.000000],
                                   [-0.021625, -0.000000, -0.000000],
                                   [ 0.021625,  0.000000,  0.000000]])
        assert abs(np.abs(nac1.de/(td.e[2] - td.e[1])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    def test_nac_camb3lyp_tddft_singlet_qchem(self):
        """
        benchmark from qchem
        """
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        nac1.states=(1,2)
        nac1.kernel()
        ref_scaled = np.array([[-0.000000,  1.573532,  0.000000],
                               [ 0.000000, -0.892200,  0.587221],
                               [ 0.000000, -0.892200, -0.587221]])
        ref_etf = np.array([[-0.000000,  0.114350,  0.000000],
                            [ 0.000000, -0.057175,  0.042404],
                            [ 0.000000, -0.057175, -0.042404]])
        ref_etf_scaled = np.array([[-0.000000,  1.641129,  0.000000],
                                   [ 0.000000, -0.820565,  0.608576],
                                   [ 0.000000, -0.820565, -0.608576]])
        assert abs(np.abs(nac1.de/(td.e[1] - td.e[0])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    def test_nac_df_camb3lyp_tddft_singlet_qchem_1(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu().density_fit()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(2,3)
        nac1.kernel()
        ref_scaled = np.array([[ 0.000000, -0.000000, -0.000000],
                               [ 0.028623, -0.000000, -0.000000],
                               [-0.028623,  0.000000,  0.000000]])
        ref_etf = np.array([[ 0.000000, -0.000000, -0.000000],
                            [ 0.000442, -0.000000, -0.000000],
                            [-0.000442,  0.000000,  0.000000]])
        ref_etf_scaled = np.array([[ 0.000000, -0.000000, -0.000000],
                                   [ 0.025503, -0.000000, -0.000000],
                                   [-0.025503,  0.000000,  0.000000]])
        assert abs(np.abs(nac1.de/(td.e[2] - td.e[1])) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4


if __name__ == "__main__":
    print("Full Tests for TD-RKS nonadiabatic coupling vectors between excited states")
    unittest.main()
