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

def setUpModule():
    global system_svp, system_tzvpp, system_charged

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

def tearDownModule():
    global system_svp, system_tzvpp, system_charged
    for mol in system_svp:
        mol.stdout.close()
    del system_svp
    for mol in system_tzvpp:
        mol.stdout.close()
    del system_tzvpp
    for mol in system_charged:
        mol.stdout.close()
    del system_charged

class KnownValues(unittest.TestCase):
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
            "total"           : -29.5993,
            "frozen"          :  -5.0756,
            "electrostatic"   : -79.8476,
            "dispersion"      :   0.0000,
            "pauli"           :  74.7720,
            "polarization"    :  -6.6736,
            "charge transfer" : -17.8500,
            "unit"            : "kJ/mol",
        }

        test_eda_result = eval_ALMO_EDA_2_energies(system_svp, xc = "HF")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value
            if type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3

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
            "total"           : -29.495283227856994,
            "frozen"          :  -4.97442204643555,
            "electrostatic"   : -79.8561011797297,
            "dispersion"      :   0.0,
            "pauli"           :  74.88167913,
            "polarization"    :  -6.673561913726887,
            "charge transfer" : -17.847299267694556,
            "unit"            : "kJ/mol",
        }

        test_eda_result = eval_ALMO_EDA_2_energies(system_svp, xc = "HF", auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value
            if type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3

    def test_almo_eda_2_wb97xv_svp(self):
        ### Q-Chem input difference
        # $rem
        # METHOD                      wB97X-V
        # $end

        reference_eda_result = {
            "total"           : -52.5576,
            "frozen"          : -14.6896,
            "electrostatic"   : -78.1825,
            "dispersion"      : -16.5685,
            "pauli"           :  80.0614,
            "polarization"    :  -5.6167,
            "charge transfer" : -32.2513,
            "unit"            : "kJ/mol",
        }

        test_eda_result = eval_ALMO_EDA_2_energies(system_svp)

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value
            if type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3

    def test_almo_eda_2_wb97xv_svp_df(self):
        ### All density fitting tests are consistent tests, see comment above
        reference_eda_result = {
            "total"           : -52.53194727429772,
            "frozen"          : -14.669063810976995,
            "electrostatic"   : -78.18252068636505,
            "dispersion"      : -16.678265350380972,
            "pauli"           :  80.19172222576903,
            "polarization"    :  -5.6144614956538925,
            "charge transfer" : -32.24842196766683,
            "unit"            : "kJ/mol",
        }

        test_eda_result = eval_ALMO_EDA_2_energies(system_svp, auxbasis = "def2-universal-jkfit")

        for key in reference_eda_result.keys():
            assert key in test_eda_result
            reference_value = reference_eda_result[key]
            test_value = test_eda_result[key]

            if type(reference_value) is str:
                assert reference_value == test_value
            if type(reference_value) is float:
                assert abs(test_value - reference_value) < 2e-3

    # def test_almo_eda_2_hf_tzvpp(self):
    #     ### Q-Chem input difference
    #     # $molecule
    #     # -1 1
    #     # --
    #     # 0 1
    #     #             C     -0.072852   -0.328834    0.654799
    #     #             H      0.403601   -0.466284    1.618541
    #     #             H      0.553011   -0.581815   -0.192581
    #     #             H     -1.081982   -0.721916    0.596137
    #     #             H     -0.213445    0.912668    0.560567
    #     # --
    #     # -1 1
    #     #             O     -0.414478    2.335424    0.468285
    #     #             H     -1.320507    2.431870    0.784608
    #     # --
    #     # 0 1
    #     #             F      0.250406   -4.009722    0.855334
    #     #             H      0.435881   -3.016345    0.778319
    #     # $end
    #     #
    #     # $rem
    #     # BASIS                       def2-tzvpp
    #     # $end

    #     reference_eda_result = {
    #         "total"           :  -62.8865,
    #         "frozen"          :  132.6279,
    #         "electrostatic"   : -228.7827,
    #         "dispersion"      :    0.0000,
    #         "pauli"           :  361.4106,
    #         "polarization"    : -116.6104,
    #         "charge transfer" :  -78.9040,
    #         "unit"            : "kJ/mol",
    #     }

    #     test_eda_result = eval_ALMO_EDA_2_energies(system_tzvpp, xc = "HF")

    #     for key in reference_eda_result.keys():
    #         assert key in test_eda_result
    #         reference_value = reference_eda_result[key]
    #         test_value = test_eda_result[key]

    #         if type(reference_value) is str:
    #             assert reference_value == test_value
    #         if type(reference_value) is float:
    #             assert abs(test_value - reference_value) < 2e-3

    # TODO
    # def test_almo_eda_2_hf_tzvpp_df(self):
    # def test_almo_eda_2_wb97xv_tzvpp(self):
    # def test_almo_eda_2_wb97xv_tzvpp_df(self):

    # def test_almo_eda_2_pbe0_charged(self):
    # def test_almo_eda_2_pbe0_charged(self):


if __name__ == "__main__":
    print("Full Tests for ALMO EDA 2 energies")
    unittest.main()
