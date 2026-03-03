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
import pytest
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf.properties.eda import eval_ALMO_EDA_2_energies
from gpu4pyscf.lib.multi_gpu import num_devices

def setUpModule():
    global system_svp, system_tzvpp, system_charged, system_two_Li, system_three_Li

    if True:
        basis = 'def2-svp'

        frag_1_mol = pyscf.M(
        atom = """
            O      0.554494    0.121391    0.181900
            O      0.928732    0.108734    1.555021
            H      1.393687   -0.115878   -0.260764
            H      0.080501    0.248683    1.992760
        """,
        # unit = "B",
        basis = basis,
        charge = 0,
        spin = 0,
        output = '/dev/null',
        )

        frag_2_mol = pyscf.M(
        atom = """
            H      3.538911   -0.228621    1.021708
            C      4.008307   -0.010800    0.064384
            H      4.220975    1.060539    0.006821
            H      4.944427   -0.570295   -0.015462
            O      3.088372   -0.407041   -0.933335
            H      3.447994   -0.197411   -1.801233
        """,
        # unit = "B",
        basis = basis,
        charge = 0,
        spin = 0,
        output = '/dev/null',
        )

        frag_3_mol = pyscf.M(
        atom = """
            Ne 1.0 0.0 -2.5
        """,
        # unit = "B",
        basis = basis,
        charge = 0,
        spin = 0,
        output = '/dev/null',
        )

        system_svp = [frag_1_mol, frag_2_mol, frag_3_mol]

    if True:
        basis = 'def2-tzvpp'

        frag_1_mol = pyscf.M(
        atom = """
            C     -0.072852   -0.328834    0.654799
            H      0.403601   -0.466284    1.618541
            H      0.553011   -0.581815   -0.192581
            H     -1.081982   -0.721916    0.596137
            H     -0.213445    0.912668    0.560567
        """,
        # unit = "B",
        basis = basis,
        charge = 0,
        spin = 0,
        output = '/dev/null',
        )

        frag_2_mol = pyscf.M(
        atom = """
            O     -0.414478    2.335424    0.468285
            H     -1.320507    2.431870    0.784608
        """,
        # unit = "B",
        basis = basis,
        charge = -1,
        spin = 0,
        output = '/dev/null',
        )

        frag_3_mol = pyscf.M(
        atom = """
            F      0.250406   -4.009722    0.855334
            H      0.435881   -3.016345    0.778319
        """,
        # unit = "B",
        basis = basis,
        charge = 0,
        spin = 0,
        output = '/dev/null',
        )

        system_tzvpp = [frag_1_mol, frag_2_mol, frag_3_mol]

    if True:
        basis = 'def2-tzvpp'

        frag_1_mol = pyscf.M(
        atom = """
            O      0.199968    0.000000    0.000006
            H      1.174548   -0.000000   -0.000001
            H     -0.287258    0.844506   -0.000003
            H     -0.287258   -0.844506   -0.000003
        """,
        # unit = "B",
        basis = basis,
        charge = 1,
        spin = 0,
        output = '/dev/null',
        )

        frag_2_mol = pyscf.M(
        atom = """
            O     -0.414478    3.335424    0.468285
            H     -1.320507    3.431870    0.784608
        """,
        # unit = "B",
        basis = basis,
        charge = -1,
        spin = 0,
        output = '/dev/null',
        )

        system_charged = [frag_1_mol, frag_2_mol]

    if True:
        basis = 'cc-pvdz'

        frag_1_mol = pyscf.M(
        atom = """
            Li 0 0 0
        """,
        # unit = "B",
        basis = basis,
        charge = 1,
        spin = 0,
        output = '/dev/null',
        )

        frag_2_mol = pyscf.M(
        atom = """
            Li 2.5 0.1 0
        """,
        # unit = "B",
        basis = basis,
        charge = 1,
        spin = 0,
        output = '/dev/null',
        )

        frag_3_mol = pyscf.M(
        atom = """
            Li 8.0 -0.1 0
        """,
        # unit = "B",
        basis = basis,
        charge = 1,
        spin = 0,
        output = '/dev/null',
        )

        system_two_Li = [frag_1_mol, frag_2_mol]
        system_three_Li = [frag_1_mol, frag_2_mol, frag_3_mol]

def tearDownModule():
    global system_svp, system_tzvpp, system_charged, system_two_Li, system_three_Li
    to_clean = [system_svp, system_tzvpp, system_charged, system_two_Li, system_three_Li]
    for system in to_clean:
        for mol in system:
            mol.stdout.close()
    del to_clean

class KnownValues(unittest.TestCase):
    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_hf_svp(self):
        ### Q-Chem input
        # $molecule
        # 0 1
        # --
        # 0 1
        #             O      0.554494    0.121391    0.181900
        #             O      0.928732    0.108734    1.555021
        #             H      1.393687   -0.115878   -0.260764
        #             H      0.080501    0.248683    1.992760
        # --
        # 0 1
        #             H      3.538911   -0.228621    1.021708
        #             C      4.008307   -0.010800    0.064384
        #             H      4.220975    1.060539    0.006821
        #             H      4.944427   -0.570295   -0.015462
        #             O      3.088372   -0.407041   -0.933335
        #             H      3.447994   -0.197411   -1.801233
        # --
        # 0 1
        #             Ne 1.0 0.0 -2.5
        # $end

        # $rem
        # JOBTYPE                     eda
        # EDA2                        1
        # METHOD                      HF
        # BASIS                       def2-svp
        # XC_GRID                     000099000590
        # NL_GRID                     000050000194
        # MAX_SCF_CYCLES              100
        # SCF_CONVERGENCE             10
        # THRESH                      14
        # MEM_STATIC                  8000
        # MEM_TOTAL                   80000
        # SYMMETRY                    FALSE
        # SYM_IGNORE                  TRUE
        # $end

        reference_eda_result = {
            "total"                   : -29.5993,
            "frozen"                  :  -5.0756,
            "electrostatic"           : -79.8476,
            "classical electrostatic" : -45.5569,
            "dispersion"              :   0.0000,
            "pauli"                   :  74.7720,
            "polarization"            :  -6.6736,
            "charge transfer"         : -17.8500,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -150.6495465870, -114.9539939337, -128.3764068109, -393.9912209745 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_svp, xc = "HF")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_hf_svp_df(self):
        ### This is a consistent test, if you put these additional keywords into Q-Chem 6.1,
        ### it will provide results that are clearly garbage.
        ### And we're not aware of any other packages capable of ALMO EDA 2.
        # $rem
        # RI_J TRUE
        # RI_K TRUE
        # AUX_BASIS RIJK-def2-qzvpp
        # $end

        reference_eda_result = {
            "total"                   : -29.495283227856994,
            "frozen"                  :  -4.97442204643555,
            "electrostatic"           : -79.85901245951626,
            "classical electrostatic" : -45.56597554325682,
            "dispersion"              :   0.0,
            "pauli"                   :  74.88459041308072,
            "polarization"            :  -6.673561913726887,
            "charge transfer"         : -17.847299267694556,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -150.64941120207857, -114.95392336587221, -128.3763244401833, -393.99089316822756 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_svp, xc = "HF", auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")


        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-8

    @pytest.mark.slow
    def test_almo_eda_2_wb97xv_svp(self):
        ### Q-Chem input difference
        # $rem
        # METHOD                      wB97X-V
        # $end

        reference_eda_result = {
            "total"                   : -52.5576,
            "frozen"                  : -14.6896,
            "electrostatic"           : -78.1825,
            "classical electrostatic" : -41.5678,
            "dispersion"              : -16.5685,
            "pauli"                   :  80.0614,
            "polarization"            :  -5.6167,
            "charge transfer"         : -32.2513,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -151.3676763251, -115.5893121943, -128.7613734911, -395.7383799036 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_svp)

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        print(test_dft_energies)
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-4

    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_wb97xv_svp_df(self):
        ### All density fitting tests are consistent tests, see comment above
        reference_eda_result = {
            "total"                   : -52.53194727429772,
            "frozen"                  : -14.669063810976995,
            "electrostatic"           : -78.18469159102223,
            "classical electrostatic" : -41.57005079093759,
            "dispersion"              : -16.678265350380972,
            "pauli"                   :  80.1938931304262,
            "polarization"            :  -5.6144614956538925,
            "charge transfer"         : -32.24842196766683,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -151.36771262102798, -115.5893510822469, -128.76138318143654, -395.7384552467653 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_svp, auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-7

    @pytest.mark.slow
    def test_almo_eda_2_hf_tzvpp(self):
        ### Q-Chem input difference
        # $molecule
        # -1 1
        # --
        # 0 1
        #             C     -0.072852   -0.328834    0.654799
        #             H      0.403601   -0.466284    1.618541
        #             H      0.553011   -0.581815   -0.192581
        #             H     -1.081982   -0.721916    0.596137
        #             H     -0.213445    0.912668    0.560567
        # --
        # -1 1
        #             O     -0.414478    2.335424    0.468285
        #             H     -1.320507    2.431870    0.784608
        # --
        # 0 1
        #             F      0.250406   -4.009722    0.855334
        #             H      0.435881   -3.016345    0.778319
        # $end
        #
        # $rem
        # BASIS                       def2-tzvpp
        # $end

        reference_eda_result = {
            "total"                   :  -62.8865,
            "frozen"                  :  132.6279,
            "electrostatic"           : -228.7827,
            "classical electrostatic" : -171.9995,
            "dispersion"              :    0.0000,
            "pauli"                   :  361.4106,
            "polarization"            : -116.6104,
            "charge transfer"         :  -78.9040,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -40.1985737780, -75.3997650604, -100.0524766223, -215.6747673743 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_tzvpp, xc = "HF")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-8

    @pytest.mark.slow
    def test_almo_eda_2_hf_tzvpp_df(self):
        ### All density fitting tests are consistent tests, see comment above
        reference_eda_result = {
            "total"                   :  -62.86023730675086,
            "frozen"                  :  132.66000202363256,
            "electrostatic"           : -228.7990731720863,
            "classical electrostatic" : -172.01034976945365,
            "dispersion"              :    0.0,
            "pauli"                   :  361.45907519571887,
            "polarization"            : -116.62119992580904,
            "charge transfer"         :  -78.89903940457438,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -40.1985786985109, -75.39975636142482, -100.05247315769718, -215.67475041761023 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_tzvpp, xc = "HF", auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-8

    @pytest.mark.skip("Too slow, functionality roughly covered by corresponding density fitting test")
    def test_almo_eda_2_wb97xv_tzvpp(self):
        ### Q-Chem input difference
        # $rem
        # METHOD                      wB97X-V
        # BASIS                       def2-tzvpp
        # $end

        reference_eda_result = {
            "total"                   : -106.4940,
            "frozen"                  :   99.5674,
            "electrostatic"           : -236.6728,
            "classical electrostatic" : -175.7119,
            "dispersion"              :  -25.6342,
            "pauli"                   :  361.8743,
            "polarization"            : -103.7696,
            "charge transfer"         : -102.2918,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -40.5026792948, -75.7784565806, -100.4463700782, -216.7680668747 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_tzvpp)

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-4

    @pytest.mark.slow
    def test_almo_eda_2_wb97xv_tzvpp_df(self):
        ### All density fitting tests are consistent tests, see comment above
        reference_eda_result = {
            "total"                   : -106.4673627872119,
            "frozen"                  :   99.60213546985209,
            "electrostatic"           : -236.6698345647513,
            "classical electrostatic" : -175.71013136670737,
            "dispersion"              :  -25.672944475878136,
            "pauli"                   :  361.9449145104815,
            "polarization"            : -103.763200520556,
            "charge transfer"         : -102.306297736508,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -40.502726568106056, -75.77847787861819, -100.44639685986992, -216.76815258352408 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_tzvpp, auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])

        # Why do we need this special treatment, and why is this threshold so loose?
        # The reason lies in the current density fitting implementation:
        # We compute the full range and long range (omega = 0.3) 2-center integrals,
        # and perform Chelosky decomposition on each of them.
        # The long range 2-center matrix is singular. The CD supposes to fail,
        # and we will switch to eigenvalue decomposition.
        # Unfortunately for a small molecule like CH4, the CD happens not to fail,
        # but the result is noisy.
        # This results in a relatively big noise in the DFT energy.
        # To avoid this problem, look for a try-except block with cholesky function call
        # in gpu4pyscf/df/df.py, and turn off the try block, i.e. force the code to do
        # eigenvalue decomposition.
        # The correct solution to this problem is: instead of decomposing the long-range
        # 2-center integral, decompose the combined (c1 SR + c2 LR) instead.
        # TODO: Once that's done, we should rerun the reference and remove the following
        # hack.
        assert np.max(np.abs(test_dft_energies[0] - reference_dft_energies[0])) < 3e-6
        assert np.max(np.abs(test_dft_energies[1:] - reference_dft_energies[1:])) < 1e-7

    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_pbe0_charged(self):
        ### Q-Chem input difference
        # $molecule
        # 0 1
        # --
        # 1 1
        # O      0.199968    0.000000    0.000006
        # H      1.174548   -0.000000   -0.000001
        # H     -0.287258    0.844506   -0.000003
        # H     -0.287258   -0.844506   -0.000003
        # --
        # -1 1
        # O     -0.414478    3.335424    0.468285
        # H     -1.320507    3.431870    0.784608
        # $end
        # $rem
        # METHOD                      PBE0
        # BASIS                       def2-tzvpp
        # $end

        reference_eda_result = {
            "total"                   : -536.3337,
            "frozen"                  : -438.8586,
            "electrostatic"           : -445.0982,
            "classical electrostatic" : -441.1186,
            "dispersion"              :   -1.5747,
            "pauli"                   :    7.8144,
            "polarization"            :  -15.5046,
            "charge transfer"         :  -81.9705,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -76.6561807725, -75.7235013652, -152.5839584027 ],
            "gradient" : [
                np.array([
                    [ 0.0002753,   0.0041637,  -0.0022195,  -0.0022195],
                    [ 0.0000000,  -0.0000000,   0.0039551,  -0.0039551],
                    [-0.0000005,   0.0000002,   0.0000002,   0.0000002],
                ]).T,
                np.array([
                    [ 0.0004203,  -0.0004203],
                    [-0.0000452,   0.0000452],
                    [-0.0001475,   0.0001475],
                ]).T,
                np.array([
                    [-0.0104900,   0.0037940,   0.0139932,  -0.0037347,  -0.0004318,  -0.0031309],
                    [ 0.0280038,  -0.0032517,  -0.0533465,  -0.0025257,   0.0287522,   0.0023679],
                    [ 0.0014947,  -0.0009571,  -0.0065862,  -0.0001392,   0.0048620,   0.0013258],
                ]).T,
            ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_charged, xc = "PBE0", if_compute_gradient = True)

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 1e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-6

        reference_dft_gradients = reference_dft_result["gradient"]
        test_dft_gradients      = test_dft_result["gradient"]
        assert len(reference_dft_gradients) == len(test_dft_gradients)
        for reference_dft_gradient, test_dft_gradient in zip(reference_dft_gradients, test_dft_gradients):
            assert np.max(np.abs(test_dft_gradient - reference_dft_gradient)) < 1e-6

    def test_almo_eda_2_pbe0_charged_df(self):
        ### All density fitting tests are consistent tests, see comment above
        reference_eda_result = {
            "total"                   : -536.3265135788014,
            "frozen"                  : -438.8503695025353,
            "electrostatic"           : -445.09564511278717,
            "classical electrostatic" : -441.11672553473835,
            "dispersion"              :   -1.5794393738524033,
            "pauli"                   :    7.824714984104298,
            "polarization"            :  -15.504103411878171,
            "charge transfer"         :  -81.97204066438793,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -76.65622798813817, -75.72351533555, -152.5840193046801 ],
            "gradient" : [
                np.array([
                    [ 2.74872178e-04, -8.17004852e-16, -4.69974993e-07],
                    [ 4.18497282e-03, -2.21735663e-17,  1.64042142e-07],
                    [-2.22999622e-03,  3.97325469e-03,  1.52931104e-07],
                    [-2.22999622e-03, -3.97325469e-03,  1.52931104e-07],
                ]),
                np.array([
                    [ 4.44095590e-04, -4.73412818e-05, -1.55244231e-04],
                    [-4.44139825e-04,  4.72821205e-05,  1.55058695e-04],
                ]),
                np.array([
                    [-0.01050011,  0.02803036,  0.00149239],
                    [ 0.00381769, -0.00325474, -0.00095648],
                    [ 0.01399252, -0.0533515 , -0.00658428],
                    [-0.003747  , -0.00254602, -0.00013928],
                    [-0.00040712,  0.02875082,  0.00485327],
                    [-0.00315618,  0.00237103,  0.0013342 ],
                ]),
            ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_charged, xc = "PBE0", auxbasis = "def2-universal-jkfit",
                                                                    if_compute_gradient = True)

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                assert abs(test_value - reference_value) < 1e-2, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 1e-7

        reference_dft_gradients = reference_dft_result["gradient"]
        test_dft_gradients      = test_dft_result["gradient"]
        assert len(reference_dft_gradients) == len(test_dft_gradients)
        for reference_dft_gradient, test_dft_gradient in zip(reference_dft_gradients, test_dft_gradients):
            assert np.max(np.abs(test_dft_gradient - reference_dft_gradient)) < 1e-6

    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_two_Li_edgecase(self):
        ### Q-Chem input
        # $molecule
        # 2 1
        # --
        # 1 1
        #             Li 0 0 0
        # --
        # 1 1
        #             Li 2.5 0.1 0
        # $end

        # $rem
        # JOBTYPE                     eda
        # EDA2                        1
        # METHOD                      PBE
        # BASIS                       cc-pvdz
        # XC_GRID                     000099000590
        # NL_GRID                     000050000194
        # MAX_SCF_CYCLES              100
        # SCF_CONVERGENCE             10
        # THRESH                      14
        # MEM_STATIC                  8000
        # MEM_TOTAL                   80000
        # SYMMETRY                    FALSE
        # SYM_IGNORE                  TRUE
        # $end
        reference_eda_result = {
            "total"                   :  553.9834,
            "frozen"                  :  555.2922,
            "electrostatic"           :  555.3023,
            "classical electrostatic" :  555.3150,
            "dispersion"              :   -0.0273,
            "pauli"                   :    0.0171,
            "polarization"            :   -0.3832,
            "charge transfer"         :   -0.9256,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -7.2555003103, -7.2555003103, -14.3000020085 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_two_Li, xc = "PBE")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                # The Hartree to kJ/mol conversion factor of Q-Chem (2625.531) is derived from Q-Chem total energy term.
                # It is important since several terms are big in value.
                # Henry has no idea why they adopt such a conversion factor.
                reference_value *= 2625.500 / 2625.531
                assert abs(test_value - reference_value) < 1e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 2e-7

    @unittest.skipIf(num_devices > 1, '')
    def test_almo_eda_2_three_Li_edgecase(self):
        reference_eda_result = {
            "total"                   :  980.0584,
            "frozen"                  :  981.3959,
            "electrostatic"           :  981.4061,
            "classical electrostatic" :  981.4190,
            "dispersion"              :   -0.0275,
            "pauli"                   :    0.0173,
            "polarization"            :   -0.3721,
            "charge transfer"         :   -0.9654,
            "unit"                    : "kJ/mol",
        }
        reference_dft_result = {
            "energy" : [ -7.2555003103, -7.2555003103, -7.2555003103, -21.3932208818 ],
        }

        test_eda_result, test_dft_result = eval_ALMO_EDA_2_energies(system_three_Li, xc = "PBE")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            elif type(reference_value) is float:
                # The Hartree to kJ/mol conversion factor of Q-Chem (2625.531) is derived from Q-Chem total energy term.
                # It is important since several terms are big in value.
                # Henry has no idea why they adopt such a conversion factor.
                reference_value *= 2625.500 / 2625.531
                assert abs(test_value - reference_value) < 1e-3, \
                    f"term = {key}, ref = {reference_value}, test = {test_value}"
            else:
                raise ValueError(f"Incorrect type of {key} = {reference_value}")

        reference_dft_energies = np.array(reference_dft_result["energy"])
        test_dft_energies      = np.array(test_dft_result["energy"])
        assert np.max(np.abs(test_dft_energies - reference_dft_energies)) < 2e-7

if __name__ == "__main__":
    print("Full Tests for ALMO EDA 2 energies")
    unittest.main()
