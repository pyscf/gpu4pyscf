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

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = mbis(mol, grids, dm)

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

        assert abs(test_energy - ref_energy) < 3e-3
        assert np.max(np.abs(test_charges - ref_charges)) < 1e-4
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 1e-4
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 3e-4
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-4

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

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = mbis(mol, grids, dm)

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

        assert abs(test_energy - ref_energy) < 2e-5
        assert np.max(np.abs(test_charges - ref_charges)) < 5e-5
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 5e-5
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 2e-4
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-4

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

        test_shell_populations, test_shell_widths, test_shell_atom_indices = mbis(mol, grids, dm, compute_multipoles = False)

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

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = mbis(mol, grids, dm)

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

        assert abs(test_energy - ref_energy) < 1e-4
        assert np.max(np.abs(test_charges - ref_charges)) < 3e-3
        assert np.max(np.abs(test_dipoles - ref_dipoles)) < 3e-4
        assert np.max(np.abs(test_quadrupoles - ref_quadrupoles)) < 1e-2
        assert np.max(np.abs(test_octupoles - ref_octupoles)) < 5e-2

if __name__ == "__main__":
    print("Full Tests for MBIS multipole")
    unittest.main()
