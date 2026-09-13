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
import gpu4pyscf
from gpu4pyscf.scf.hf  import RHF
from gpu4pyscf.scf.uhf import UHF
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.qmmm.hirshfeld import hirshfeld, hirshfeld_kernel

class KnownValues(unittest.TestCase):
    def test_hirshfeld_neutral_rhf(self):
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

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = hirshfeld(mol, grids, dm, xc = "HF", auxbasis = "def2-universal-jkfit")

        ### Reference Q-Chem input
        # $rem
        # JOBTYPE force
        # BASIS def2-svp
        # METHOD HF
        # XC_GRID       000099000590
        # NL_GRID       000050000194
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # ri_j        True
        # ri_k        True
        # aux_basis RIJK-def2-TZVP
        # HIRSHFELD True
        # HIRSHITER False
        # $end
        ref_energy = -3428.7481489134
        ref_charges = np.array([ -0.197481,  0.046375, -0.172539,  0.128914,  0.194641,  0.000029, ])

        assert abs(test_energy - ref_energy) < 3e-9
        assert np.max(np.abs(test_charges - ref_charges)) < 2e-4

    def test_hirshfeld_charged_uhf(self):
        mol = pyscf.M(
            atom = """
                F -1.2 0.0 0.0
                Li 0 0 0
                F  1.2 0.0 0.0
            """,
            basis = {"Li" : "sto-6g", "F" : "def2-qzvp"},
            charge = -1,
            verbose = 0,
        )
        mf = UHF(mol).density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = Grids(mol)
        grids.atom_grid = (99,590)
        grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        grids.prune = None
        grids.radii_adjust = None

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = hirshfeld(mol, grids, dm, xc = "HF", auxbasis = "def2-universal-jkfit")

        ### Reference Q-Chem input
        # $rem
        # JOBTYPE force
        # BASIS gen
        # METHOD HF
        # XC_GRID       000099000590
        # NL_GRID       000050000194
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # ri_j        True
        # ri_k        True
        # aux_basis RIJK-def2-TZVP
        # HIRSHFELD True
        # HIRSHITER False
        # $end

        # $basis
        # Li     0
        # S   6   1.00
        #       0.1671758462D+03       0.9163596281D-02
        #       0.3065150840D+02       0.4936149294D-01
        #       0.8575187477D+01       0.1685383049D+00
        #       0.2945808337D+01       0.3705627997D+00
        #       0.1143943581D+01       0.4164915298D+00
        #       0.4711391391D+00       0.1303340841D+00
        # SP   6   1.00
        #       0.6597563981D+01      -0.1325278809D-01       0.3759696623D-02
        #       0.1305830092D+01      -0.4699171014D-01       0.3767936984D-01
        #       0.4058510193D+00      -0.3378537151D-01       0.1738967435D+00
        #       0.1561455158D+00       0.2502417861D+00       0.4180364347D+00
        #       0.6781410394D-01       0.5951172526D+00       0.4258595477D+00
        #       0.3108416550D-01       0.2407061763D+00       0.1017082955D+00
        # ****
        # F     0
        # S   8   1.00
        #  132535.9734500              0.47387482743D-04
        #   19758.1125880              0.37070120897D-03
        #    4485.1996947              0.19450784713D-02
        #    1273.8151020              0.80573291994D-02
        #     418.93831236             0.27992880781D-01
        #     152.55721985             0.82735120175D-01
        #      59.821524823            0.19854169012
        #      24.819076932            0.34860632233
        # S   2   1.00
        #     100.74446673             0.10505068816
        #      30.103728290            0.94068472434
        # S   1   1.00
        #      10.814283272            1.0000000
        # S   1   1.00
        #       4.8172886770           1.0000000
        # S   1   1.00
        #       1.6559334213           1.0000000
        # S   1   1.00
        #       0.64893519582          1.0000000
        # S   1   1.00
        #       0.24778104545          1.0000000
        # P   5   1.00
        #     240.96654114             0.30389933451D-02
        #      57.020699781            0.24357738582D-01
        #      18.126952120            0.11442925768
        #       6.6457404621           0.37064659853
        #       2.6375722892           0.79791551766
        # P   1   1.00
        #       1.0638217200           1.0000000
        # P   1   1.00
        #       0.41932562750          1.0000000
        # P   1   1.00
        #       0.15747588299          1.0000000
        # D   1   1.00
        #       5.01400000             1.0000000
        # D   1   1.00
        #       1.72500000             1.0000000
        # D   1   1.00
        #       0.58600000             1.0000000
        # F   1   1.00
        #       3.56200000             1.0000000
        # F   1   1.00
        #       1.14800000             1.0000000
        # G   1   1.00
        #       2.37600000             1.0000000
        # ****
        # $end
        ref_energy = -206.3071084879
        ref_charges = np.array([ -0.598679,  0.197359, -0.598679, ])

        assert abs(test_energy - ref_energy) < 3e-9
        assert np.max(np.abs(test_charges - ref_charges)) < 1e-5

    def test_hirshfeld_neutral_rks(self):
        mol = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                H      0.629312    0.629312    0.629312
                F     -0.768949   -0.768949    0.768949
                Cl    -1.022775    1.022775   -1.022775
                Br     1.115470   -1.115470   -1.115470
            """,
            basis = "def2-svp",
            verbose = 0,
        )
        mf = RKS(mol, xc = "wB97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99, 590)
        mf.nlcgrids.atom_grid = (50, 194)
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = mf.grids

        def _make_mf(mol):
            mf = UKS(mol, xc = "wB97MV").density_fit(auxbasis = "def2-universal-jkfit")
            mf.grids.atom_grid = (99, 590)
            mf.nlcgrids.atom_grid = (50, 194)
            mf.conv_tol = 1e-10
            return mf

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = hirshfeld_kernel(mol, grids, dm, make_mf = _make_mf)

        ref_energy = -3172.1209095366
        ref_charges = np.array([  0.145544,  0.054114, -0.084534, -0.054412, -0.060709, ])

        assert abs(test_energy - ref_energy) < 1e-5
        assert np.max(np.abs(test_charges - ref_charges)) < 1e-4


    def test_hirshfeld_charged_ecp(self):
        mol = pyscf.M(
            atom = """
                I -2.0 0.0 0.0
                H 0 0 0
                I  2.0 0.1 0.0
            """,
            basis = "def2-svp",
            ecp = "def2-svp",
            charge = -1,
            verbose = 4,
        )
        mf = UKS(mol, xc = "wB97X").density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10

        test_energy = mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()

        grids = Grids(mol)
        grids.atom_grid = (99,590)

        test_charges, test_dipoles, test_quadrupoles, test_octupoles = hirshfeld(mol, grids, dm, xc = "wB97X", auxbasis = "def2-universal-jkfit")

        ### Reference ORCA input
        # !wB97X def2-SVP DEFGRID3 Hirshfeld

        # * xyz -1 1
        #                 I -2.0 0.0 0.0
        #                 H 0 0 0
        #                 I  2.0 0.1 0.0
        # *
        ref_energy = -596.19707049266685
        ### This is ORCA result. From ORCA document:
        ### In ORCA, the pro-atomic density within the Hirshfeld method is calculated via density fitting with a set of Gaussian s-functions per element.
        ### So it is a bit different from our result.
        ### Q-Chem ECP + Hirshfeld doesn't work.
        # ref_charges = np.array([ -0.529947,  0.061624, -0.531676 ])
        ### So we just check against an old result, i.e. This is a consistency test.
        ref_charges = np.array([ -0.51646553,  0.03472508, -0.51825951 ])

        assert abs(test_energy - ref_energy) < 1e-4
        assert np.max(np.abs(test_charges - ref_charges)) < 1e-6

if __name__ == "__main__":
    print("Full Tests for Hirshfeld multipole")
    unittest.main()
