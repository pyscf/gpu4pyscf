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
from gpu4pyscf.scf.hf  import RHF
from gpu4pyscf.scf.uhf import UHF
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.qmmm.mbis import mbis

class KnownValues(unittest.TestCase):
    def test_mbis_rhf(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                Ar 3.0 0.0 0.0
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
                Kr 10.0 0.0 0.1
            """,
            basis = "def2-svp",
            verbose = 0,
        )
        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = Grids(mol)
        grids.atom_grid = (99,590)

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm)

        ### Reference ORCA input
        # ! HF DEF2-SVP TightSCF MBIS

        # *XYZ 0 1
        # O  0.0000  0.7375 -0.0528
        # Ar 3.0 0.0 0.0
        # O  0.0000 -0.7375 -0.1528
        # H  0.8190  0.8170  0.4220
        # H -0.8190 -0.8170  1.4220
        # Kr 10.0 0.0 0.1
        # *

        # %method
        # MBIS_LARGEPRINT TRUE
        # end
        ref_energy = -3428.74836336563203
        ref_charges = np.array([ -0.434717, -0.007150, -0.269490,  0.446555,  0.264804,  0.000032, ])
        ref_dipoles = np.array([
            [-0.041973, -0.136388, -0.062346],
            [ 0.089139, -0.037956, -0.033764],
            [ 0.043744,  0.145279, -0.118807],
            [ 0.029044, -0.001060,  0.014901],
            [-0.040549,  0.029240,  0.075222],
            [ 0.001148, -0.000346, -0.001124],
        ])
        ref_quadrupoles_orca_shape = np.array([
            [ -4.232479,  -3.953297,  -4.297775,  0.038177,  0.061874,  0.086958],
            [ -8.640563,  -8.468726,  -8.437907,  0.028931,  0.029546, -0.010502],
            [ -4.374082,  -3.956366,  -4.252054, -0.031096, -0.079295,  0.010022],
            [ -0.241443,  -0.233252,  -0.244380,  0.000400,  0.006490, -0.000383],
            [ -0.718140,  -0.692559,  -0.722051,  0.003976,  0.003360, -0.008766],
            [-12.973837, -12.973442, -12.973462,  0.000041,  0.000086, -0.000001],
        ])
        ref_quadrupoles = np.zeros((mol.natm, 3, 3))
        ref_quadrupoles[:, 0, 0] = ref_quadrupoles_orca_shape[:, 0]
        ref_quadrupoles[:, 1, 1] = ref_quadrupoles_orca_shape[:, 1]
        ref_quadrupoles[:, 2, 2] = ref_quadrupoles_orca_shape[:, 2]
        ref_quadrupoles[:, 0, 1] = ref_quadrupoles[:, 1, 0] = ref_quadrupoles_orca_shape[:, 3]
        ref_quadrupoles[:, 0, 2] = ref_quadrupoles[:, 2, 0] = ref_quadrupoles_orca_shape[:, 4]
        ref_quadrupoles[:, 1, 2] = ref_quadrupoles[:, 2, 1] = ref_quadrupoles_orca_shape[:, 5]
        ref_octupoles_orca_shape = np.array([
            [-0.226183,  0.299418, -0.201317, -0.082593, -0.168408,  0.031377, -0.007359, -0.059904, -0.050612, -0.107788],
            [ 1.061837, -0.166123, -0.146957, -0.092879, -0.103728,  0.284886,  0.022325,  0.192540, -0.058588, -0.055291],
            [ 0.214378, -0.255909, -0.750374,  0.134926, -0.249658,  0.035722, -0.008278,  0.172565, -0.201107,  0.170053],
            [ 0.019067,  0.006566,  0.014910,  0.003795,  0.004336,  0.005429, -0.000371,  0.010907,  0.001581, -0.002522],
            [-0.063466,  0.082347,  0.104127,  0.031548,  0.040824, -0.024044,  0.010057, -0.010704,  0.039112,  0.022798],
            [ 0.005829, -0.001493, -0.005046, -0.000510, -0.001581,  0.001364,  0.000001,  0.001393, -0.001568, -0.000498],
        ])
        ref_octupoles = np.zeros((mol.natm, 3, 3, 3))
        ref_octupoles[:, 0, 0, 0] = ref_octupoles_orca_shape[:, 0]
        ref_octupoles[:, 1, 1, 1] = ref_octupoles_orca_shape[:, 1]
        ref_octupoles[:, 2, 2, 2] = ref_octupoles_orca_shape[:, 2]
        ref_octupoles[:, 0, 0, 1] = ref_octupoles[:, 0, 1, 0] = ref_octupoles[:, 1, 0, 0] = ref_octupoles_orca_shape[:, 3]
        ref_octupoles[:, 0, 0, 2] = ref_octupoles[:, 0, 2, 0] = ref_octupoles[:, 2, 0, 0] = ref_octupoles_orca_shape[:, 4]
        ref_octupoles[:, 0, 1, 1] = ref_octupoles[:, 1, 0, 1] = ref_octupoles[:, 1, 1, 0] = ref_octupoles_orca_shape[:, 5]
        ref_octupoles[:, 0, 1, 2] = ref_octupoles[:, 0, 2, 1] = ref_octupoles[:, 1, 0, 2] = ref_octupoles[:, 1, 2, 0] = ref_octupoles[:, 2, 0, 1] = ref_octupoles[:, 2, 1, 0] = ref_octupoles_orca_shape[:, 6]
        ref_octupoles[:, 0, 2, 2] = ref_octupoles[:, 2, 0, 2] = ref_octupoles[:, 2, 2, 0] = ref_octupoles_orca_shape[:, 7]
        ref_octupoles[:, 1, 1, 2] = ref_octupoles[:, 1, 2, 1] = ref_octupoles[:, 2, 1, 1] = ref_octupoles_orca_shape[:, 8]
        ref_octupoles[:, 1, 2, 2] = ref_octupoles[:, 2, 1, 2] = ref_octupoles[:, 2, 2, 1] = ref_octupoles_orca_shape[:, 9]
        ref_r3_moment = np.array([ 23.528681, 53.205098, 24.264910,  1.159630,  5.106663, 84.125575, ])

        assert abs(test_energy - ref_energy) < 3e-3
        assert np.max(np.abs(test_charges - ref_charges)) < 1e-4
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 1e-4
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 3e-4
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-4
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 2e-3

    def test_mbis_uhf(self):
        mol = pyscf.M(
            atom = """
                C      0.751600   -0.022500   -0.020900
                H      1.114611    1.036945   -0.070252
                H      1.425195   -0.765480   -0.086513
                C     -0.751600    0.022500    0.020900
                H     -1.166900   -0.833400    0.568700
                H     -1.115700    0.932600    0.515100
                H     -1.185000    0.004400   -0.987500
            """,
            basis = "def2-svp",
            charge = 0,
            spin = 1,
            verbose = 0,
        )
        mf = UHF(mol).density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = Grids(mol)
        grids.atom_grid = (99,590)

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm, initial_guess = "repo_guess")

        ### Reference ORCA input
        # ! UHF DEF2-SVP TightSCF MBIS

        # *XYZ 0 2
        # C      0.751600   -0.022500   -0.020900
        # H      1.114611    1.036945   -0.070252
        # H      1.425195   -0.765480   -0.086513
        # C     -0.751600    0.022500    0.020900
        # H     -1.166900   -0.833400    0.568700
        # H     -1.115700    0.932600    0.515100
        # H     -1.185000    0.004400   -0.987500
        # *

        # %method
        # MBIS_LARGEPRINT TRUE
        # end
        ref_energy = -78.53051246228665
        ref_charges = np.array([ -0.205005,  0.072520,  0.117649, -0.268727,  0.088103,  0.101233,  0.094227, ])
        ref_dipoles = np.array([
            [-0.078226, -0.039803, -0.004656],
            [-0.001643,  0.074782, -0.002074],
            [ 0.049522, -0.047391, -0.004075],
            [ 0.065578, -0.002901, -0.005529],
            [-0.011259, -0.048658,  0.030992],
            [-0.012812,  0.050277,  0.026834],
            [-0.014467, -0.001207, -0.056023],
        ])
        ref_quadrupoles_orca_shape = np.array([
            [-4.824590, -4.679704, -4.600241, -0.002897,  0.006717,  0.000808],
            [-0.564636, -0.569426, -0.590391, -0.008137,  0.000058,  0.001029],
            [-0.459656, -0.471293, -0.492972, -0.001660, -0.001425, -0.000196],
            [-4.888772, -4.741776, -4.770745, -0.011820, -0.001720, -0.000645],
            [-0.532825, -0.543627, -0.542792, -0.006815,  0.001946,  0.002455],
            [-0.514160, -0.527752, -0.526260,  0.004737,  0.001844, -0.001993],
            [-0.530109, -0.536188, -0.541702, -0.001059, -0.005993,  0.000292],
        ])
        ref_quadrupoles = np.zeros((mol.natm, 3, 3))
        ref_quadrupoles[:, 0, 0] = ref_quadrupoles_orca_shape[:, 0]
        ref_quadrupoles[:, 1, 1] = ref_quadrupoles_orca_shape[:, 1]
        ref_quadrupoles[:, 2, 2] = ref_quadrupoles_orca_shape[:, 2]
        ref_quadrupoles[:, 0, 1] = ref_quadrupoles[:, 1, 0] = ref_quadrupoles_orca_shape[:, 3]
        ref_quadrupoles[:, 0, 2] = ref_quadrupoles[:, 2, 0] = ref_quadrupoles_orca_shape[:, 4]
        ref_quadrupoles[:, 1, 2] = ref_quadrupoles[:, 2, 1] = ref_quadrupoles_orca_shape[:, 5]
        ref_octupoles_orca_shape = np.array([
            [ 0.355595, -0.305559, -0.013791, -0.077817, -0.032610, -0.305003, -0.003629, -0.216232, -0.025838, -0.070222],
            [-0.018827,  0.079790, -0.002129,  0.031058, -0.004883, -0.016412,  0.001083, -0.020223,  0.000245,  0.032523],
            [ 0.052193, -0.062686, -0.006471, -0.014600, -0.001437,  0.007842, -0.000686,  0.017698, -0.000361, -0.022211],
            [-0.399013, -0.096826,  0.161903,  0.013851,  0.019154,  0.296749, -0.004111,  0.271843, -0.219002,  0.002636],
            [-0.000016, -0.049306,  0.034932, -0.015585,  0.008427,  0.011259,  0.002490,  0.013881,  0.001783, -0.016186],
            [-0.008240,  0.047928,  0.029822,  0.015545,  0.006893,  0.008478, -0.001930,  0.011187,  0.000798,  0.017043],
            [-0.008763, -0.002825, -0.047771,  0.001349, -0.014213,  0.013746, -0.000768,  0.007682, -0.025727,  0.000224],
        ])
        ref_octupoles = np.zeros((mol.natm, 3, 3, 3))
        ref_octupoles[:, 0, 0, 0] = ref_octupoles_orca_shape[:, 0]
        ref_octupoles[:, 1, 1, 1] = ref_octupoles_orca_shape[:, 1]
        ref_octupoles[:, 2, 2, 2] = ref_octupoles_orca_shape[:, 2]
        ref_octupoles[:, 0, 0, 1] = ref_octupoles[:, 0, 1, 0] = ref_octupoles[:, 1, 0, 0] = ref_octupoles_orca_shape[:, 3]
        ref_octupoles[:, 0, 0, 2] = ref_octupoles[:, 0, 2, 0] = ref_octupoles[:, 2, 0, 0] = ref_octupoles_orca_shape[:, 4]
        ref_octupoles[:, 0, 1, 1] = ref_octupoles[:, 1, 0, 1] = ref_octupoles[:, 1, 1, 0] = ref_octupoles_orca_shape[:, 5]
        ref_octupoles[:, 0, 1, 2] = ref_octupoles[:, 0, 2, 1] = ref_octupoles[:, 1, 0, 2] = ref_octupoles[:, 1, 2, 0] = ref_octupoles[:, 2, 0, 1] = ref_octupoles[:, 2, 1, 0] = ref_octupoles_orca_shape[:, 6]
        ref_octupoles[:, 0, 2, 2] = ref_octupoles[:, 2, 0, 2] = ref_octupoles[:, 2, 2, 0] = ref_octupoles_orca_shape[:, 7]
        ref_octupoles[:, 1, 1, 2] = ref_octupoles[:, 1, 2, 1] = ref_octupoles[:, 2, 1, 1] = ref_octupoles_orca_shape[:, 8]
        ref_octupoles[:, 1, 2, 2] = ref_octupoles[:, 2, 1, 2] = ref_octupoles[:, 2, 2, 1] = ref_octupoles_orca_shape[:, 9]
        ref_r3_moment = np.array([ 34.333544,  3.336940,  2.556058, 35.225552,  3.058440,  2.936329,  3.037714, ])

        assert abs(test_energy - ref_energy) < 2e-5
        assert np.max(np.abs(test_charges - ref_charges)) < 5e-5
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 5e-5
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 2e-4
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-4
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 1e-3

    def test_mbis_rks(self):
        mol = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                H      0.629312    0.629312    0.629312
                F     -0.768949   -0.768949    0.768949
                Cl    -1.022775    1.022775   -1.022775
                Br     1.115470   -1.115470   -1.115470
            """,
            basis = "def2-svp",
            ecp = "def2-svp",
            verbose = 0,
        )
        mf = RKS(mol, xc = "PBE").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = mf.grids

        test_shell_populations, test_shell_widths, test_shell_atom_indices = mbis(mol, grids, dm, compute_properties = False)

        ### Reference ORCA input
        # ! PBE def2-svp TightSCF DEFGRID3 MBIS

        # *XYZ 0 1
        # C      0.000000    0.000000    0.000000
        # H      0.629312    0.629312    0.629312
        # F     -0.768949   -0.768949    0.768949
        # Cl    -1.022775    1.022775   -1.022775
        # Br     1.115470   -1.115470   -1.115470
        # *

        # %method
        # MBIS_LARGEPRINT TRUE
        # end
        ref_energy = -3171.61135289084723
        ref_shell_atom_indices = np.array([0,0, 1, 2,2, 3,3,3, 4,4,4,4], dtype = np.int32)
        ref_valence_shell_populations = np.array([ 4.091516, 0.870251, 7.544253, 8.586013, 9.661445, ])
        ref_valence_shell_widths = np.array([ 0.501597, 0.371141, 0.337167, 0.518189, 0.551814, ])
        valence_shell_indices = len(ref_shell_atom_indices) - 1 - np.unique(ref_shell_atom_indices[::-1], return_index = True)[1]

        assert abs(test_energy - ref_energy) < 5e-4
        assert np.all(test_shell_atom_indices == ref_shell_atom_indices)
        assert np.max(np.abs(test_shell_populations[valence_shell_indices] - ref_valence_shell_populations)) < 5e-4
        assert np.max(np.abs(test_shell_widths[valence_shell_indices] - ref_valence_shell_widths)) < 3e-5

    def test_mbis_uks(self):
        mol = pyscf.M(
            atom = """
                H      1.8853     -0.0401      1.0854
                C      1.2699     -0.0477      0.1772
                H      1.5840      0.8007     -0.4449
                H      1.5089     -0.9636     -0.3791
                C     -0.2033      0.0282      0.5345
                H     -0.4993     -0.8287      1.1714
                H     -0.4235      0.9513      1.1064
                O     -0.9394      0.0157     -0.6674
                H     -1.8540      0.0626     -0.4252
            """,
            basis = "6-31g",
            verbose = 0,
        )
        mf = UKS(mol, xc = "wB97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99, 590)
        mf.nlcgrids.atom_grid = (50, 194)
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = mf.grids

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm)

        ### Reference ORCA input
        # ! wB97M-V 6-31G TightSCF DEFGRID3 MBIS

        # *XYZ 0 1
        #   H      1.8853     -0.0401      1.0854
        #   C      1.2699     -0.0477      0.1772
        #   H      1.5840      0.8007     -0.4449
        #   H      1.5089     -0.9636     -0.3791
        #   C     -0.2033      0.0282      0.5345
        #   H     -0.4993     -0.8287      1.1714
        #   H     -0.4235      0.9513      1.1064
        #   O     -0.9394      0.0157     -0.6674
        #   H     -1.8540      0.0626     -0.4252
        # *

        # %method
        # MBIS_LARGEPRINT TRUE
        # end
        ref_energy = -154.90319593093577
        ref_charges = np.array([  0.117186, -0.420738,  0.134820,  0.134753,  0.180223,  0.032960,  0.032951, -0.633758,  0.421603, ])
        ref_dipoles = np.array([
            [ 0.017113,  0.000541,  0.034242],
            [-0.037792,  0.000936, -0.018536],
            [ 0.004548,  0.030361, -0.024600],
            [ 0.001878, -0.032385, -0.022251],
            [-0.020487, -0.004667, -0.152211],
            [-0.003377, -0.047545,  0.025437],
            [ 0.000736,  0.049394,  0.021883],
            [ 0.035161,  0.003982,  0.152033],
            [-0.003109,  0.000205,  0.002150],
        ])
        ref_quadrupoles_orca_shape = np.array([
            [-0.515032, -0.520371, -0.525178, -0.000623, -0.010598,  0.000272],
            [-5.014804, -4.962680, -4.929532,  0.004351,  0.058871, -0.001303],
            [-0.487768, -0.502440, -0.501109, -0.005069,  0.002106,  0.004693],
            [-0.487080, -0.503970, -0.500481,  0.003854,  0.001376, -0.004663],
            [-4.101315, -3.995645, -3.950766,  0.007312,  0.076506, -0.001696],
            [-0.584720, -0.605972, -0.584801, -0.004211,  0.011924,  0.010115],
            [-0.584134, -0.608067, -0.583413,  0.003111,  0.010815, -0.009413],
            [-4.235539, -4.493263, -4.370167, -0.008869,  0.053753,  0.002204],
            [-0.266421, -0.261132, -0.250649,  0.000356,  0.003732,  0.000220],
        ])
        ref_quadrupoles = np.zeros((mol.natm, 3, 3))
        ref_quadrupoles[:, 0, 0] = ref_quadrupoles_orca_shape[:, 0]
        ref_quadrupoles[:, 1, 1] = ref_quadrupoles_orca_shape[:, 1]
        ref_quadrupoles[:, 2, 2] = ref_quadrupoles_orca_shape[:, 2]
        ref_quadrupoles[:, 0, 1] = ref_quadrupoles[:, 1, 0] = ref_quadrupoles_orca_shape[:, 3]
        ref_quadrupoles[:, 0, 2] = ref_quadrupoles[:, 2, 0] = ref_quadrupoles_orca_shape[:, 4]
        ref_quadrupoles[:, 1, 2] = ref_quadrupoles[:, 2, 1] = ref_quadrupoles_orca_shape[:, 5]
        ref_octupoles_orca_shape = np.array([
            [ 0.015708,  0.002527,  0.031941, -0.000547,  0.008271, -0.002028,  0.000458, -0.003699,  0.020285, -0.000173],
            [ 0.343862,  0.037233, -0.097685, -0.038176, -0.254288, -0.163899,  0.016613, -0.174204,  0.140883, -0.006867],
            [-0.003382,  0.031765, -0.028056,  0.008328, -0.013457, -0.007328, -0.000552, -0.007512, -0.001288,  0.008999],
            [-0.005737, -0.029971, -0.025889, -0.010245, -0.012586, -0.008216,  0.001569, -0.008486, -0.000595, -0.010197],
            [-0.213450, -0.031980, -0.154812,  0.015932,  0.023653,  0.068174, -0.005042,  0.208991, -0.212431,  0.001006],
            [ 0.012995, -0.049963,  0.007492, -0.019385,  0.008915,  0.014179,  0.001461,  0.008619, -0.007251, -0.010913],
            [ 0.018137,  0.044551,  0.004974,  0.021105,  0.007010,  0.014401, -0.003205,  0.009920, -0.008653,  0.011763],
            [ 0.294780,  0.006843, -0.073839, -0.012596, -0.100903,  0.040943,  0.003176, -0.117808,  0.110438, -0.005769],
            [ 0.006033,  0.000384, -0.002924, -0.000194,  0.003007, -0.000577, -0.000038, -0.001549,  0.002913, -0.000251],
        ])
        ref_octupoles = np.zeros((mol.natm, 3, 3, 3))
        ref_octupoles[:, 0, 0, 0] = ref_octupoles_orca_shape[:, 0]
        ref_octupoles[:, 1, 1, 1] = ref_octupoles_orca_shape[:, 1]
        ref_octupoles[:, 2, 2, 2] = ref_octupoles_orca_shape[:, 2]
        ref_octupoles[:, 0, 0, 1] = ref_octupoles[:, 0, 1, 0] = ref_octupoles[:, 1, 0, 0] = ref_octupoles_orca_shape[:, 3]
        ref_octupoles[:, 0, 0, 2] = ref_octupoles[:, 0, 2, 0] = ref_octupoles[:, 2, 0, 0] = ref_octupoles_orca_shape[:, 4]
        ref_octupoles[:, 0, 1, 1] = ref_octupoles[:, 1, 0, 1] = ref_octupoles[:, 1, 1, 0] = ref_octupoles_orca_shape[:, 5]
        ref_octupoles[:, 0, 1, 2] = ref_octupoles[:, 0, 2, 1] = ref_octupoles[:, 1, 0, 2] = ref_octupoles[:, 1, 2, 0] = ref_octupoles[:, 2, 0, 1] = ref_octupoles[:, 2, 1, 0] = ref_octupoles_orca_shape[:, 6]
        ref_octupoles[:, 0, 2, 2] = ref_octupoles[:, 2, 0, 2] = ref_octupoles[:, 2, 2, 0] = ref_octupoles_orca_shape[:, 7]
        ref_octupoles[:, 1, 1, 2] = ref_octupoles[:, 1, 2, 1] = ref_octupoles[:, 2, 1, 1] = ref_octupoles_orca_shape[:, 8]
        ref_octupoles[:, 1, 2, 2] = ref_octupoles[:, 2, 1, 2] = ref_octupoles[:, 2, 2, 1] = ref_octupoles_orca_shape[:, 9]
        ref_r3_moment = np.array([   2.949209, 36.547898,  2.777550,  2.778054, 28.564182,  3.424751,  3.425067, 25.124448,  1.288743, ])

        assert abs(test_energy - ref_energy) < 1e-4
        assert np.max(np.abs(test_charges - ref_charges)) < 3e-3
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 3e-4
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 1e-2
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-2
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 1e-1

    def test_mbis_ecp_I(self):
        mol = pyscf.M(
            atom = """
                I -2.0 0.0 0.0
                H 0 0 0
                I  2.0 0.1 0.0
            """,
            basis = "def2-svp",
            ecp = "def2-svp",
            charge = -1,
            verbose = 0,
        )
        mf = RKS(mol, xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = mf.grids

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm)

        ### Reference ORCA input
        # ! PBE0 def2-svp TightSCF DEFGRID3 MBIS

        # *XYZ -1 1
        # I -2.0 0.0 0.0
        # H 0 0 0
        # I  2.0 0.1 0.0
        # *

        # %method
        # MBIS_LARGEPRINT TRUE
        # end
        ref_energy = -596.13570631596986
        ref_charges = np.array([ -0.538668,  0.078969, -0.540302 ])
        ref_dipoles = np.array([
            [ 0.046821,  0.004997,  -0.000000],
            [ 0.000171, -0.003007,   0.000000],
            [-0.046702,  0.002658,  -0.000000],
        ])
        ref_quadrupoles_orca_shape = np.array([
            [-21.009513,-21.964324, -21.964249, -0.009349, -0.000000,  0.000000],
            [ -0.912617, -0.835838,  -0.835839, -0.001925,  0.000000, -0.000000],
            [-21.021372,-21.966649, -21.969880,  0.056766, -0.000000, -0.000000],
        ])
        ref_quadrupoles = np.zeros((mol.natm, 3, 3))
        ref_quadrupoles[:, 0, 0] = ref_quadrupoles_orca_shape[:, 0]
        ref_quadrupoles[:, 1, 1] = ref_quadrupoles_orca_shape[:, 1]
        ref_quadrupoles[:, 2, 2] = ref_quadrupoles_orca_shape[:, 2]
        ref_quadrupoles[:, 0, 1] = ref_quadrupoles[:, 1, 0] = ref_quadrupoles_orca_shape[:, 3]
        ref_quadrupoles[:, 0, 2] = ref_quadrupoles[:, 2, 0] = ref_quadrupoles_orca_shape[:, 4]
        ref_quadrupoles[:, 1, 2] = ref_quadrupoles[:, 2, 1] = ref_quadrupoles_orca_shape[:, 5]
        ref_octupoles_orca_shape = np.array([
            [-1.988769,  0.023054,  -0.000000, -0.012919,  0.000000,  0.240698,  0.000000,  0.242010, -0.000000,  0.007699],
            [-0.000337, -0.014498,   0.000000,  0.001570, -0.000000,  0.000545, -0.000000,  0.000224,  0.000000, -0.004838],
            [ 1.980318, -0.012595,  -0.000000,  0.110068, -0.000000, -0.233815,  0.000000, -0.239820, -0.000000, -0.004308],
        ])
        ref_octupoles = np.zeros((mol.natm, 3, 3, 3))
        ref_octupoles[:, 0, 0, 0] = ref_octupoles_orca_shape[:, 0]
        ref_octupoles[:, 1, 1, 1] = ref_octupoles_orca_shape[:, 1]
        ref_octupoles[:, 2, 2, 2] = ref_octupoles_orca_shape[:, 2]
        ref_octupoles[:, 0, 0, 1] = ref_octupoles[:, 0, 1, 0] = ref_octupoles[:, 1, 0, 0] = ref_octupoles_orca_shape[:, 3]
        ref_octupoles[:, 0, 0, 2] = ref_octupoles[:, 0, 2, 0] = ref_octupoles[:, 2, 0, 0] = ref_octupoles_orca_shape[:, 4]
        ref_octupoles[:, 0, 1, 1] = ref_octupoles[:, 1, 0, 1] = ref_octupoles[:, 1, 1, 0] = ref_octupoles_orca_shape[:, 5]
        ref_octupoles[:, 0, 1, 2] = ref_octupoles[:, 0, 2, 1] = ref_octupoles[:, 1, 0, 2] = ref_octupoles[:, 1, 2, 0] = ref_octupoles[:, 2, 0, 1] = ref_octupoles[:, 2, 1, 0] = ref_octupoles_orca_shape[:, 6]
        ref_octupoles[:, 0, 2, 2] = ref_octupoles[:, 2, 0, 2] = ref_octupoles[:, 2, 2, 0] = ref_octupoles_orca_shape[:, 7]
        ref_octupoles[:, 1, 1, 2] = ref_octupoles[:, 1, 2, 1] = ref_octupoles[:, 2, 1, 1] = ref_octupoles_orca_shape[:, 8]
        ref_octupoles[:, 1, 2, 2] = ref_octupoles[:, 2, 1, 2] = ref_octupoles[:, 2, 2, 1] = ref_octupoles_orca_shape[:, 9]
        ref_r3_moment = np.array([ 172.882242,  6.163151, 172.958552 ])

        assert abs(test_energy - ref_energy) < 1e-4
        assert np.max(np.abs(test_charges - ref_charges)) < 3e-5
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 3e-4
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 1e-3
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-3
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 1e-2

    def test_mbis_negative_charged(self):
        # For a highly negatively charged system, different MBIS initial guess will produce different results
        mol = pyscf.M(
            atom = """
                K    0.00000000    0.00000000    0.00000000
                B    1.60000000    1.55000000    1.58000000
                H    2.06760000    2.42960000    2.28440000
                H    2.47960000    1.08240000    0.87560000
                H    0.72040000    2.01760000    0.87560000
                H    1.13240000    0.67040000    2.28440000
                B    1.62000000   -1.52000000   -1.63000000
                H    2.58640000   -0.81560000   -1.38900000
                H    1.86090000   -2.22440000   -2.59640000
                H    0.65360000   -0.81560000   -1.87100000
                H    1.37910000   -2.22440000   -0.66360000
                B   -1.55000000    1.66000000   -1.58000000
                H   -0.84560000    2.03320000   -0.65630000
                H   -0.84560000    1.28680000   -2.50370000
                H   -2.25440000    2.58370000   -1.95320000
                H   -2.25440000    0.73630000   -1.20680000
                # B   -1.68000000   -1.57000000    1.53000000
                # H   -0.69350000   -1.43140000    2.23440000
                # H   -1.54140000   -2.55650000    0.82560000
                # H   -1.81860000   -0.58350000    0.82560000
                # H   -2.66650000   -1.70860000    2.23440000
            """,
            basis = "6-31g",
            charge = -2,
            verbose = 0,
        )
        mf = RKS(mol, xc = "wB97X").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-10

        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = mf.grids

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm, initial_guess = "repo_guess")

        # This is a consistency test
        ref_charges = np.array([ 0.8909499921551003,
                                -0.0921156654947133, -0.2258429337283929, -0.1893384029700311, -0.1877233061026962, -0.2674486150173099,
                                -0.0910678184407452, -0.1899315547577647, -0.2263299478842682, -0.1973729769621559, -0.2597918760817628,
                                -0.0902560198569784, -0.1892012698479042, -0.1904121602586875, -0.2276572877939158, -0.2664854783057549])
        ref_dipoles = np.array([
            [ 0.023419321455609 ,  0.0234477504819632, -0.0272592498631867],
            [ 0.0197167992410572,  0.0126988156726076,  0.0232527532535311],
            [ 0.0186525105252655,  0.0346483959924703,  0.0241409073251306],
            [ 0.0480186139365017, -0.0171255916748207, -0.0327992773767285],
            [-0.0417754530674809,  0.0129557142860823, -0.0438255433899712],
            [-0.0077598749486747, -0.0227376671754191,  0.0299747248995671],
            [ 0.0234663952467644, -0.0197228615217348, -0.012310775784914 ],
            [ 0.0522828192427483,  0.0309195218505533,  0.0051087476880132],
            [ 0.0111662638549762, -0.0243160830427069, -0.0385955154692152],
            [-0.0430428859209466,  0.0445042819884445,  0.0068790048631962],
            [ 0.0018884891463419, -0.0307615162631208,  0.0266321120421124],
            [-0.0225187657963353,  0.0137013834417173, -0.0204206594318816],
            [ 0.043899856573229 ,  0.0074899598111398,  0.0430884582780572],
            [ 0.0328872695492637, -0.0118476630144499, -0.0505509410003933],
            [-0.0241897930271487,  0.0363502308749354, -0.0154884487342735],
            [-0.0296005797639728, -0.0247895655534034,  0.0043222828408922],
        ])
        ref_r3_moment = np.array([56.82302819572204  ,
                                  46.51840101796637  ,  6.708120816726169 ,  6.279104759005428 ,  6.042560107135095 ,  7.638430994843543 ,
                                  46.720074282005626 ,  6.281968995387113 ,  6.70264536240944  ,  6.09119674312146  ,  7.4889066162149565,
                                  46.48891139636986  ,  6.057328075412309 ,  6.2841335390764845,  6.72476165030775  ,  7.602442196333006 ])

        assert np.max(np.abs(test_charges - ref_charges)) < 1e-6
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 1e-6
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 1e-6

        test_charges, test_dipoles, test_quadrupoles, test_octupoles, test_r2_moment, test_r3_moment, test_r4_moment \
            = mbis(mol, grids, dm, initial_guess = "paper_guess")

        # This is a consistency test
        ref_charges = np.array([ 0.89673853675923  ,
                                -0.1322783947707027, -0.2167353814008159, -0.1797350780947937, -0.175459395741578 , -0.2602669259542929,
                                -0.1366619875509816, -0.1802953794510391, -0.2163503230591712, -0.1806793028929277, -0.251395566521637 ,
                                -0.1319057103676924, -0.1762132257326412, -0.181586548890543 , -0.2181520791537017, -0.25892635193223  ])
        ref_dipoles = np.array([
            [ 0.0096540615063273,  0.0039841279836268, -0.0097173157861106],
            [ 0.0250033464550202,  0.0138327079189506,  0.0151821437099846],
            [ 0.0180572329893498,  0.0331310461877771,  0.0233399754108119],
            [ 0.0452140684019164, -0.0171112923737458, -0.0317603418023786],
            [-0.0407799283121582,  0.0091667770426139, -0.0431897930354902],
            [-0.0064354622281213, -0.0202897906439849,  0.026158125813849 ],
            [ 0.0322363937183838, -0.0120010788154995, -0.0133415160496679],
            [ 0.0499631953121447,  0.0293870194765333,  0.0051351621136582],
            [ 0.0111142319109201, -0.0234036406944146, -0.0367309085464654],
            [-0.0420364680997373,  0.0441860298772078,  0.0109435298737584],
            [ 0.0026736699328282, -0.0271330534069336,  0.0239609701508522],
            [-0.0150514614657867,  0.0158816557519964, -0.026427635727183 ],
            [ 0.0432100445679517,  0.0037142704490381,  0.0418701794946069],
            [ 0.0314891582506749, -0.0114303209855326, -0.0482954717255673],
            [-0.0232969263045234,  0.0346755137540988, -0.0150926655983875],
            [-0.0259015051466177, -0.0222276633185535,  0.0032883717977225],
        ])
        ref_r3_moment = np.array([41.78304210654531  ,
                                  50.561849810729996 ,  6.581838424148762 ,  6.146577743748766 ,  5.874468470008853 ,  7.729219900307689 ,
                                  51.00257932076755  ,  6.157194336907767 ,  6.567136836715232 ,  5.854806201497654 ,  7.52702746953835  ,
                                  50.61052402050666  ,  5.8783740903796815,  6.173248070105693 ,  6.591809218745358 ,  7.677542000806872 ])

        assert np.max(np.abs(test_charges - ref_charges)) < 1e-6
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 1e-6
        assert np.max(np.abs(test_r3_moment - ref_r3_moment)) < 1e-6

if __name__ == "__main__":
    print("Full Tests for MBIS multipole")
    unittest.main()
