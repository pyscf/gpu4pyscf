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

bas0 = "cc-pvdz"

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
        CIS_DER_NUMSTATE   3
        CALC_NAC           true
        $end

        $derivative_coupling
        0 is the reference state
        0 1 2
        $end
        ---------------------------------------------------
        DC between ground and excited states with ETF:
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.400124       0.000000       0.000000
        2       0.200062      -0.000000      -0.000000
        3       0.200062       0.000000      -0.000000
        ---------------------------------------------------
        ---------------------------------------------------
        CIS Force Matrix Element
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.108311       0.000000       0.000000
        2       0.054155      -0.000000      -0.000000
        3       0.054155       0.000000      -0.000000
        ---------------------------------------------------
        ---------------------------------------------------
        CIS derivative coupling without ETF
        Atom         X              Y              Z     
        ---------------------------------------------------
        1       0.038717       0.000000       0.000000
        2       0.101769      -0.000000      -0.000000
        3       0.101769       0.000000      -0.000000
        ---------------------------------------------------
        """
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        nac1.states=(1,0)
        nac1.kernel()
        ref = np.array([[ 0.038717,  0.000000,  0.000000],
                        [ 0.101769, -0.000000, -0.000000],
                        [ 0.101769,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[-0.400124,  0.000000,  0.000000],
                                   [ 0.200062, -0.000000, -0.000000],
                                   [ 0.200062,  0.000000, -0.000000]])
        ref_etf = np.array([[-0.108311,  0.000000,  0.000000],
                            [ 0.054155, -0.000000, -0.000000],
                            [ 0.054155,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[0]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

        nac1.states=(2,0)
        nac1.kernel()
        ref = np.array([[ 0.000000,  0.000000,  0.000000],
                        [-0.097345, -0.000000, -0.000000],
                        [ 0.097345,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[ 0.000000,  0.000000,  0.000000],
                                   [-0.253906, -0.000000, -0.000000],
                                   [ 0.253906,  0.000000, -0.000000]])
        ref_etf = np.array([[ 0.000000,  0.000000,  0.000000],
                            [-0.086175, -0.000000, -0.000000],
                            [ 0.086175,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[1]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    @pytest.mark.slow
    def test_nac_pbe_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        nstate = 1
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    @pytest.mark.slow
    def test_nac_pbe0_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        nstate = 1
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

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
        nac1.states=(1,0)
        nac1.kernel()
        ref = np.array([[ 0.021798,  0.000000,  0.000000],
                        [ 0.101468, -0.000000, -0.000000],
                        [ 0.101468,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[-0.410555,  0.000000,  0.000000],
                                   [ 0.205278, -0.000000, -0.000000],
                                   [ 0.205278,  0.000000, -0.000000]])
        ref_etf = np.array([[-0.114800,  0.000000,  0.000000],
                            [ 0.057400, -0.000000, -0.000000],
                            [ 0.057400,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[0]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

        nac1.states=(2,0)
        nac1.kernel()
        ref = np.array([[ 0.000000,  0.000000,  0.000000],
                        [-0.098640, -0.000000, -0.000000],
                        [ 0.098640,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[ 0.000000,  0.000000,  0.000000],
                                   [-0.259228, -0.000000, -0.000000],
                                   [ 0.259228,  0.000000, -0.000000]])
        ref_etf = np.array([[ 0.000000,  0.000000,  0.000000],
                            [-0.090242, -0.000000, -0.000000],
                            [ 0.090242,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[1]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

    def test_nac_camb3lyp_tda_singlet_qchem(self):
        """
        benchmark from qchem
        """
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrks.NAC(td)
        nac1.states=(1,0)
        nac1.kernel()
        ref = np.array([[-0.054062,  0.000000,  0.000000],
                        [-0.098522, -0.000000, -0.000000],
                        [-0.098522,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[ 0.395409,  0.000000,  0.000000],
                                   [-0.197704, -0.000000, -0.000000],
                                   [-0.197704,  0.000000, -0.000000]])
        ref_etf = np.array([[ 0.112110,  0.000000,  0.000000],
                            [-0.056055, -0.000000, -0.000000],
                            [-0.056055,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[0]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

        nac1.states=(2,0)
        nac1.kernel()
        ref = np.array([[ 0.000000,  0.000000,  0.000000],
                        [ 0.098640, -0.000000, -0.000000],
                        [-0.098640,  0.000000, -0.000000]])
        ref_etf_scaled = np.array([[-0.000000,  0.000000,  0.000000],
                                   [ 0.256405, -0.000000, -0.000000],
                                   [-0.256405,  0.000000, -0.000000]])
        ref_etf = np.array([[-0.000000,  0.000000,  0.000000],
                            [ 0.090564, -0.000000, -0.000000],
                            [-0.090564,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/td.e[1]) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled)).max() < 1e-4

if __name__ == "__main__":
    print("Full Tests for TD-RKS nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
