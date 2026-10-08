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
from gpu4pyscf.scf.uhf import UHF

def get_qchem_autosad_guess_energy(mf, dm0):
    mol = mf.mol

    mf.max_cycle = 0
    e = mf.kernel(dm0 = dm0)

    # The following code resolves a bug in Q-Chem, when running density fitting calculation,
    # The energy of guess step (step 1 in Q-Chem output) does not include HF exchange energy.
    # This happens for both HF and DFT, except for pure functionals without HF exchange contribution.
    if hasattr(mf, "with_df"):
        hermi = 1
        if hasattr(mf, 'xc'):
            ni = mf._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vk = mf.get_k(mol, dm0, hermi, omega=omega, lr_factor=alpha, sr_factor=hyb)
        else:
            vk = mf.get_k(mol, dm0, hermi)
        if dm0.ndim == 2:
            e += 0.25 * float(cp.einsum('ij,ji->', dm0, vk).real)
        else:
            e += 0.5 * float(cp.einsum('uij,uji->', dm0, vk).real)

    return e

class KnownValues(unittest.TestCase):
    # Attention: Do not use STO or any other minimal basis for testing, they will hide potential bugs.

    def test_sad_guess_rhf(self):
        mol = pyscf.M(
            atom = """
                H 0 0 0
                F 0 0.1 1
            """,
            basis = "cc-pvdz", # nctr != 1
            verbose = 0,
        )

        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        ### Reference Q-Chem input
        # $rem
        # JOBTYPE              sp
        # METHOD               HF
        # SCF_GUESS            AUTOSAD
        # BASIS                cc-pvdz
        # ECP                  def2-ecp
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP
        # XC_GRID              000099000590
        # NL_GRID              000050000194
        # BASIS_LIN_DEP_THRESH 6
        # MEM_TOTAL            7000
        # MEM_STATIC           800
        # THRESH               13
        # SCF_CONVERGENCE      6
        # PURECART             1111
        # INTEGRAL_SYMMETRY    false
        # POINT_GROUP_SYMMETRY false
        # NO_REORIENT          true
        # BECKE_SHIFT          UNSHIFTED
        # !SCF_PRINT            2
        # $end
        ref_guess_energy = -89.6672455721

        assert abs(test_guess_energy - ref_guess_energy) < 5e-6

    def test_sad_guess_rhf_direct(self):
        mol = pyscf.M(
            atom = """
                O   0.00000000   0.00000000   0.00000000
                H   0.94361690   0.00000000   0.26468890
                H  -0.47180845   0.81719736   0.26468890
                H  -0.47180845  -0.81719736   0.26468890
            """,
            basis = "6-31g",
            charge = 1,
            verbose = 0,
        )

        mf = RHF(mol)

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        ### Remove the following
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP
        ref_guess_energy = -76.7151836021

        assert abs(test_guess_energy - ref_guess_energy) < 2e-6

    def test_sad_guess_rks(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-svp",
            verbose = 0,
        )

        mf = RKS(mol, xc = "PBE").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Since we cannot control the grid setup in Q-Chem atomic calculation
        # # (it always uses SG1, can Henry cannot figure out how to reproduce it exactly), the atomic calculation
        # # result is quite off.
        # ref_guess_energy = -151.1607143427
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -151.16046138557607

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

    def test_sad_guess_rks_vv10(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-svp",
            verbose = 0,
        )

        mf = RKS(mol, xc = "wB97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Since we cannot control the grid setup in Q-Chem atomic calculation
        # # (it always uses SG1, can Henry cannot figure out how to reproduce it exactly), the atomic calculation
        # # result is quite off.
        # ref_guess_energy = -146.3329001918
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -146.33290480934588

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

    def test_hcore_guess_rks(self):
        mol = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-svp",
            verbose = 0,
        )

        mf = RKS(mol, xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None

        dm0 = mf.init_guess_by_1e()

        mf.max_cycle = 0
        test_guess_energy = mf.kernel(dm0 = dm0)

        ### Modify the following
        # SCF_GUESS            CORE
        ref_guess_energy = -139.4993766993

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

    def test_sad_guess_uhf(self):
        mol = pyscf.M(
            atom = """
                H      1.0686     -0.1411      1.0408
                C      0.5979      0.0151      0.0688
                H      1.2687      0.2002     -0.7717
                O     -0.5960     -0.0151     -0.0686
            """,
            basis = "6-31g**",
            verbose = 0,
        )

        mf = UHF(mol).density_fit(auxbasis = "def2-universal-jkfit")

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        ### Add the following
        # UNRESTRICTED         true
        # SCF_GUESS_MIX        0
        ref_guess_energy = -100.1703417241

        assert abs(test_guess_energy - ref_guess_energy) < 5e-5

    def test_sad_guess_uks(self):
        mol = pyscf.M(
            atom = """
                O   0.00000000   0.00000000   0.00000000
                H   0.94361690   0.00000000   0.26468890
                H  -0.47180845   0.81719736   0.26468890
                H  -0.47180845  -0.81719736   0.26468890
            """,
            basis = "def2-tzvp",
            spin = 1,
            verbose = 0,
        )

        mf = UKS(mol, xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit")

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Since we cannot control the grid setup in Q-Chem atomic calculation
        # # (it always uses SG1, can Henry cannot figure out how to reproduce it exactly), the atomic calculation
        # # result is quite off.
        # ref_guess_energy = -74.9178805257
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -74.9206587667209

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

    def test_sad_guess_uks_ecp(self):
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

        mf = UKS(mol, xc = "wB97X")

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_autosad_guess_energy(mf, dm0)

        ### Add the following
        # ECP_FIT              False
        # ECP_QUAD             True
        ### Remove the following, because Q-Chem ECP doesn't go well with density fitting
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP

        # # The following value is from Q-Chem. Since we cannot control the grid setup in Q-Chem atomic calculation
        # # (it always uses SG1, can Henry cannot figure out how to reproduce it exactly), the atomic calculation
        # # result is quite off.
        # ref_guess_energy = -596.0625315683
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -596.0618708861276

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

if __name__ == "__main__":
    print("Full Tests for initial guess")
    unittest.main()
