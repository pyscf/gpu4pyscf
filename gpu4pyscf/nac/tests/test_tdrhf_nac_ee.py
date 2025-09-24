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
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf
import pytest
from gpu4pyscf.lib.multi_gpu import num_devices

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
    def test_nac_tdhf_singlet_bdf_qchem(self):
        """
        benchmark from both Qchem and BDF
        $rem
        JOBTYPE              sp          
        METHOD               hf       
        BASIS                cc-pvdz     
        CIS_N_ROOTS          5       
        CIS_SINGLETS         TRUE        
        CIS_TRIPLETS         FALSE       
        SYMMETRY             FALSE       
        SYM_IGNORE           TRUE   
        SCF_CONVERGENCE      14
        XC_GRID 000099000590
        RPA True
        BASIS_LIN_DEP_THRESH 12
        CIS_DER_NUMSTATE   3
        CALC_NAC           true
        $end

        $derivative_coupling
        0 is the reference state
        0 1 2
        $end
        ======== next is bdf input =======
        $COMPASS
        Title
        NAC-test
        Basis
        cc-pvdz
        Geometry
        O 0. 0. 0.
        H  0.  -0.757 0.587
        H  0.   0.757 0.587
        End geometry
        unit
        angstrom
        Nosymm
        false
        $END

        $XUANYUAN
        $END

        $SCF
        RHF
        $END

        $tddft
        iroot
        2 # One root for each irrep
        istore
        1 # File number, to be used later in $resp
        crit_vec
        1.d-6
        crit_e
        1.d-8
        gridtol
        1.d-7 # tighten the tolerance value of XC grid generation. This helps to
            # reduce numerical error, and is recommended for open-shell molecules
        $end

        $resp
        iprt
        1
        QUAD # quadratic response
        FNAC # first-order NACME
        double # calculation of properties from single residues (ground state-excited
            # state fo-NACMEs belong to this kind of properties)
        norder
        1
        method
        2
        nfiles
        1 # must be the same as the istore value in the $TDDFT block
        pairs
        1 
        1 1 1 1 1 2
        noresp
        $end

        ==== next is qchem output ====
        ---------------------------------------------------
        CIS derivative coupling without ETF
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       2.322814      -0.000000
        2       0.000000      -1.261329       0.874838
        3       0.000000      -1.261329      -0.874838
        ---------------------------------------------------
        ---------------------------------------------------
        CIS Force Matrix Element
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       0.155023      -0.000000
        2       0.000000      -0.077512       0.058107
        3       0.000000      -0.077512      -0.058107
        ---------------------------------------------------
        ---------------------------------------------------
        CIS derivative coupling with ETF
        Atom         X              Y              Z     
        ---------------------------------------------------
        1      -0.000000       2.391220      -0.000000
        2       0.000000      -1.195610       0.896297
        3       0.000000      -1.195610      -0.896297
        ---------------------------------------------------
        ==== next is BDF output ====
        Gradient contribution from Final-NAC(R)-Escaled
        1       -0.0000000000        2.3228388812       -0.0000000000
        2        0.0000000001       -1.2613416810        0.8748886382
        3       -0.0000000001       -1.2613416810       -0.8748886382
        ......
        Gradient contribution from Final-NAC(S)
        1       -0.0000000000        0.1550246190       -0.0000000000
        2        0.0000000000       -0.0775123095        0.0581102860
        3       -0.0000000000       -0.0775123095       -0.0581102860
        ......
        Gradient contribution from Final-NAC(S)-Escaled
        1       -0.0000000000        2.3912443625       -0.0000000000
        2        0.0000000001       -1.1956221812        0.8963472690
        3       -0.0000000001       -1.1956221812       -0.8963472690
        """
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDHF().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
        nac1.states=(1,2)
        nac1.kernel()
        ref_etf_scaled_qchem = np.array([[-0.000000,  2.391220, -0.000000],
                                         [ 0.000000, -1.195610,  0.896297],
                                         [ 0.000000, -1.195610, -0.896297]])
        ref_qchem = np.array([[-0.000000,  2.322814, -0.000000],
                              [ 0.000000, -1.261329,  0.874838],
                              [ 0.000000, -1.261329, -0.874838]])
        ref_etf_qchem = np.array([[-0.000000,  0.155023, -0.000000],
                                  [ 0.000000, -0.077512,  0.058107],
                                  [ 0.000000, -0.077512, -0.058107]])
        ref_bdf = np.array([[-0.0000000000,  0.1505898846, -0.0000000000],
                            [ 0.0000000000, -0.0817729115,  0.0567191208],
                            [-0.0000000000, -0.0817729115, -0.0567191208],])
        ref_scaled_bdf = np.array([[-0.0000000000,  2.3228388812, -0.0000000000],
                                   [ 0.0000000001, -1.2613416810,  0.8748886382],
                                   [-0.0000000001, -1.2613416810, -0.8748886382],])
        ref_etf_bdf = np.array([[-0.0000000000,  0.1550246190, -0.0000000000],
                                [ 0.0000000000, -0.0775123095,  0.0581102860],
                                [-0.0000000000, -0.0775123095, -0.0581102860],])
        ref_etf_scaled_bdf = np.array([[-0.0000000000,  2.3912443625, -0.0000000000],
                                       [ 0.0000000001, -1.1956221812,  0.8963472690],
                                       [-0.0000000001, -1.1956221812, -0.8963472690],])
        assert abs(np.abs(nac1.de/(td.e[1] - td.e[0])) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de) - np.abs(ref_bdf)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_scaled_bdf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf_bdf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled_bdf)).max() < 1e-4

    @unittest.skipIf(num_devices > 1, '')
    def test_nac_tda_singlet_qchem(self):
        """
        Comapre with qchem
        """
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
        nac1.states=(1,2)
        nac1.kernel()
        ref_etf_scaled_qchem = np.array([[-0.000000,  2.324939,  0.000000],
                                         [ 0.000000, -1.162470,  0.870959],
                                         [ 0.000000, -1.162470, -0.870959]])
        ref_qchem = np.array([[-0.000000,  2.256714,  0.000000],
                              [ 0.000000, -1.228419,  0.849342],
                              [ 0.000000, -1.228419, -0.849343]])
        ref_etf_qchem = np.array([[-0.000000,  0.151652,  0.000000],
                                  [ 0.000000, -0.075826,  0.056811],
                                  [ 0.000000, -0.075826, -0.056811]])
        assert abs(np.abs(nac1.de/(td.e[1] - td.e[0])) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled_qchem)).max() < 1e-4

        nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
        nac1.states=(1,3)
        nac1.kernel()
        ref_etf_scaled_qchem = np.array([[-0.785926,  0.000000,  0.000000],
                                         [ 0.392963, -0.000000, -0.000000],
                                         [ 0.392963,  0.000000, -0.000000]])
        ref_qchem = np.array([[-0.897866,  0.000000,  0.000000],
                              [ 0.436555, -0.000000, -0.000000],
                              [ 0.436555,  0.000000, -0.000000]])
        ref_etf_qchem = np.array([[-0.075626,  0.000000,  0.000000],
                                  [ 0.037813, -0.000000, -0.000000],
                                  [ 0.037813,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/(td.e[2] - td.e[0])) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled_qchem)).max() < 1e-4

        nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
        nac1.states=(2,3)
        nac1.kernel()
        ref_etf_scaled_qchem = np.array([[-0.000000,  0.000000,  0.000000],
                                         [ 0.021695, -0.000000, -0.000000],
                                         [-0.021695,  0.000000, -0.000000]])
        ref_qchem = np.array([[-0.000000,  0.000000,  0.000000],
                              [ 0.022423, -0.000000, -0.000000],
                              [-0.022423,  0.000000, -0.000000]])
        ref_etf_qchem = np.array([[-0.000000,  0.000000,  0.000000],
                                  [ 0.000672, -0.000000, -0.000000],
                                  [-0.000672,  0.000000, -0.000000]])
        assert abs(np.abs(nac1.de/(td.e[2] - td.e[1])) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(ref_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(ref_etf_qchem)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(ref_etf_scaled_qchem)).max() < 1e-4

    @pytest.mark.slow
    def test_nac_tda_singlet_fdiff(self):
        """
        Compare with finite difference
        """
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = gpu4pyscf.nac.tdrhf.NAC(td)
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrhf.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 5.0E-5

        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrhf.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 5.0E-5


if __name__ == "__main__":
    print("Full Tests for TD-RHF nonadiabatic coupling vectors between excited states.")
    unittest.main()
