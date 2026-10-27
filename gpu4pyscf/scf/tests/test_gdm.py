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
import pyscf, gpu4pyscf
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.scf.gdm import gdm

class KnownValues(unittest.TestCase):
    def test_gdm_rhf(self):
        mol = pyscf.M(
            atom = """
                H      1.2001      0.0363      0.8431
                C      0.7031      0.0083     -0.1305
                H      0.9877      0.8943     -0.7114
                H      1.0155     -0.8918     -0.6742
                O     -0.6582     -0.0067      0.1730
                H     -1.1326     -0.0311     -0.6482
            """,
            basis = "cc-pvdz",
            verbose = 0,
        )

        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-12
        ref_energy = mf.kernel()
        assert mf.converged
        ref_gradient = mf.Gradients().kernel()
        ref_n_cycle = mf.cycles
        del mf

        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-12
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_gradient = mf.Gradients().kernel()
        test_n_cycle = mf.cycles

        assert np.abs(test_energy - ref_energy) < 1e-11
        assert np.max(np.abs(test_gradient - ref_gradient)) < 5e-8
        assert test_n_cycle <= ref_n_cycle + 5

    def test_gdm_rks(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-tzvp",
            verbose = 0,
        )

        mf = RKS(mol, xc = "r2scan").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None

        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ### Q-Chem reference input
        # $rem
        # JOBTYPE              force
        # METHOD               R2SCAN
        # BASIS                def2-TZVP
        # ECP                  def2-ecp
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP
        # XC_GRID              000099000590
        # NL_GRID              000050000194
        # BECKE_SHIFT          UNSHIFTED
        # BASIS_LIN_DEP_THRESH 6
        # UNRESTRICTED         False
        # SCF_GUESS_MIX        0
        # SCF_GUESS            AUTOSAD
        # SCF_ALGORITHM        GDM
        # MEM_TOTAL            7000
        # MEM_STATIC           800
        # MAX_SCF_CYCLES       100
        # THRESH               13
        # SCF_CONVERGENCE      6
        # INTEGRAL_SYMMETRY    false
        # POINT_GROUP_SYMMETRY false
        # NO_REORIENT          true
        # $end
        ref_energy = -151.3987667624
        ref_gradient = np.array([
            [ 0.0044537,  0.0445215, -0.0182507, -0.0307246],
            [ 0.0344355, -0.0418900, -0.0080918,  0.0155463],
            [ 0.0273595, -0.0832309, -0.0077350,  0.0636064],
        ]).T
        ref_n_cycle = 15

        assert np.abs(test_energy - ref_energy) < 2e-7
        assert np.max(np.abs(test_gradient - ref_gradient)) < 5e-7
        assert test_n_cycle <= ref_n_cycle + 5

    def test_gdm_uks_spin_eq_0(self):
        mol = pyscf.M(
            atom = """
                C        0.98632600      -1.52268500      -0.15943600
                C       -0.34861600      -0.95090900       0.34204200
                H       -0.61716886      -1.14263981       1.38089649
                H       -0.72441489       1.25404039       0.29970284
                H        2.39684273      -0.65357967      -1.34207790
                N       -0.18040400       0.45285000       0.01291700
                N        0.92764200       0.68910200      -0.67486400
                N        1.54231200      -0.52297300      -0.81975700
                I       -5.43052800       0.09309000      -0.30056500
                H        1.36403000      -2.57919300      -0.31290800
                H       -1.22506800      -1.16805200      -0.21933200
            """, # The iodine atom is necessary to reproduce the behavior
            basis = "def2-SVP",
            ecp = "def2-SVP",
            charge = 1,
            spin = 0,
            verbose = 0,
        )

        mf = UKS(mol, xc = "wB97X").density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.max_cycle = 100
        mf.conv_tol_grad = 1e-6

        dm0 = mf.get_init_guess(key='qchem_sad') # Yes, we start with a spin-averaged guess
        mf.conv_tol = 1e-10

        mf = gdm(mf)
        test_energy = mf.kernel(dm0 = dm0)
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_s2, _ = mf.spin_square()

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ### Q-Chem reference input
        # $rem
        # JOBTYPE              force
        # METHOD               wB97X
        # BASIS                def2-SVP
        # ECP                  def2-ecp
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP
        # XC_GRID              000099000590
        # NL_GRID              000050000194
        # BASIS_LIN_DEP_THRESH 6
        # POP_MULLIKEN         true
        # CHELPG               true
        # MULTIPOLE_ORDER      4
        # UNRESTRICTED         true
        # SCF_GUESS_MIX        0
        # SCF_GUESS            AUTOSAD
        # SCF_ALGORITHM        GDM
        # MEM_TOTAL            7000
        # MEM_STATIC           800
        # CPSCF_NSEG           10
        # MAX_SCF_CYCLES       200
        # THRESH               13
        # SCF_CONVERGENCE      6
        # INTEGRAL_SYMMETRY    false
        # POINT_GROUP_SYMMETRY false
        # NO_REORIENT          true
        # $end
        ref_energy = -540.5917460796
        ref_s2 = 1.033701096
        ref_gradient = np.array([
            [ 0.0280932, -0.0176043, -0.0046859,  0.0063775, -0.0111521, -0.0611342,
             -0.0071530,  0.0339246, -0.0010829,  0.0115247,  0.0228923],
            [ 0.0107968, -0.0041461,  0.0004219, -0.0052152, -0.0142149,  0.0079076,
              0.0271706, -0.0309390,  0.0001212, -0.0229070,  0.0310041],
            [ 0.0124926, -0.0081475, -0.0109490, -0.0019482,  0.0084664,  0.0135321,
              0.0055755, -0.0272142, -0.0000788, -0.0103788,  0.0186499],
        ]).T
        ref_n_cycle = 72

        assert np.abs(test_energy - ref_energy) < 2e-4
        assert np.abs(test_s2 - ref_s2) < 5e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 5e-5
        assert test_n_cycle <= ref_n_cycle + 15

    def test_gdm_uks_spin_neq_0(self):
        mol = pyscf.M(
            atom = """
                C        0.94054455      -0.01152091      -0.00390959
                C       -0.27309667      -0.13851223      -0.00409319
                C       -1.67604997      -0.25458966      -0.00381518
                C       -2.44189679      -1.40807448      -0.01287824
                C       -3.84081477      -1.14512030      -0.00878136
                C       -4.13038620       0.19818729       0.00308203
                S       -2.69869599       1.16992462       0.01007284
                C        2.33820378       0.12319870      -0.00415143
                C        3.19315017       1.19082272      -0.02078984
                C        4.53911846       0.74097922      -0.01114025
                C        4.42898145      -0.62636054       0.01175974
                O        3.12366760      -1.02802943       0.01662357
                H       -1.99946927      -2.39813430      -0.02214605
                H       -4.60251584      -1.91767715      -0.01465898
                H       -5.10413454       0.67146955       0.00782707
                H        5.45048689       1.32281605      -0.01954215
                H        5.16145794      -1.42168823       0.02623033
            """,
            basis = "def2-SVP",
            charge = 0,
            spin = 1,
            verbose = 0,
        )

        mf = UKS(mol, xc = "wB97X").density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.max_cycle = 100
        mf.conv_tol_grad = 1e-6

        dm0 = mf.get_init_guess(key='qchem_sad') # Yes, we start with a spin-averaged guess
        mf.conv_tol = 1e-10

        mf = gdm(mf)
        test_energy = mf.kernel(dm0 = dm0)
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_s2, _ = mf.spin_square()

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ref_energy = -856.7060020335
        ref_s2 = 0.772751683
        ref_gradient = np.array([
            [ 0.0234882, -0.0236192,  0.0226606, -0.0024212, -0.0019048, -0.0153281,
              0.0016676, -0.0261750,  0.0027655,  0.0061326,  0.0201931, -0.0059912,
             -0.0019052,  0.0032071,  0.0049403, -0.0041785, -0.0035318],
            [ 0.0043018, -0.0036710, -0.0002197, -0.0114842, -0.0104154,  0.0040103,
              0.0142593,  0.0062651,  0.0068086,  0.0099299, -0.0072166, -0.0190018,
              0.0042783,  0.0032299, -0.0021699, -0.0025508,  0.0036465],
            [-0.0000067, -0.0000125, -0.0000385, -0.0000942, -0.0000827,  0.0000480,
              0.0001347, -0.0001561, -0.0000994, -0.0001705,  0.0001532,  0.0003176,
              0.0000387,  0.0000253, -0.0000289,  0.0000371, -0.0000653],
        ]).T
        ref_n_cycle = 35

        assert np.abs(test_energy - ref_energy) < 1e-4
        assert np.abs(test_s2 - ref_s2) < 1e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 5e-5
        assert test_n_cycle <= ref_n_cycle + 5

    def test_gdm_rks_pcm(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-tzvp",
            verbose = 0,
        )

        mf = RKS(mol, xc = "wB97M-V").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf = mf.PCM()
        mf.with_solvent.method = "IEFPCM"
        mf.with_solvent.eps = 40.0
        mf.conv_tol = 1e-11

        ref_energy = mf.kernel()
        assert mf.converged
        ref_n_cycle = mf.cycles
        gobj = mf.Gradients()
        gobj.grid_response = True
        ref_gradient = gobj.kernel()
        del mf, gobj

        #

        mf = RKS(mol, xc = "wB97M-V").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf = mf.PCM()
        mf.with_solvent.method = "IEFPCM"
        mf.with_solvent.eps = 40.0
        mf.conv_tol = 1e-11

        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-7
        assert test_n_cycle <= ref_n_cycle + 5

if __name__ == "__main__":
    print("Full Tests for GDM")
    unittest.main()
