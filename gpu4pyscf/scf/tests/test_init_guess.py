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
from unittest import mock
import numpy as np
import cupy as cp
import pyscf, gpu4pyscf
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.scf.uhf import UHF
from pyscf import gto
from gpu4pyscf.scf import hf, rohf, ghf

def get_qchem_sad_guess_energy(mf, dm0):
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

    # Another likely bug, the guess step energy misses the solvent energy.
    if hasattr(mf, 'with_solvent'):
        e -= mf.with_solvent.e

    return e

class KnownValues(unittest.TestCase):
    # Attention: Do not use STO or any other minimal basis for testing, they will hide potential bugs.

    # Note on Q-Chem reference:
    # 1. Q-Chem doesn't copy all parameters for molecular calculation to atomic calculation, in particular:
    #    The density fitting is not turned on in atomic calculation, which causes a 1e-6 level of error.
    #    The DFT grid is always SG1 for atomic calculation (Henry not able to reproduce SG1 exactly), which causes a 1e-3 level of error.
    # 2. Q-Chem default basis for 6-31g and cc-pvdz are slightly different from BSE basis, which also causes a 1e-6 level of error.
    #    The difference in guess energy is much more significant than the final SCF energy.
    #    So we use BASIS GEN and copy the BSE basis for Q-Chem input.

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Reference Q-Chem input
        # $rem
        # JOBTYPE              sp
        # METHOD               HF
        # SCF_GUESS            AUTOSAD
        # BASIS                GEN
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
        ref_guess_energy = -89.6672456210

        assert abs(test_guess_energy - ref_guess_energy) < 3e-6

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Remove the following
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP
        ref_guess_energy = -76.7151843245

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Off because of the different grid for atomic calculation.
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
        mf.nlcgrids.atom_grid = (50,194)

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Off because of the different grid for atomic calculation.
        # ref_guess_energy = -146.3329001918
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -146.33290506213888

        assert abs(test_guess_energy - ref_guess_energy) < 5e-7

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        # # The following value is from Q-Chem. Off because of the different grid for atomic calculation.
        # ref_guess_energy = -74.9178805257
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -74.9206587667209

        assert abs(test_guess_energy - ref_guess_energy) < 2e-7

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

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Add the following
        # ECP_FIT              False
        # ECP_QUAD             True
        ### Remove the following, because Q-Chem ECP doesn't go well with density fitting
        # RI_J                 true
        # RI_K                 true
        # AUX_BASIS            RIJK-def2-TZVP

        # # The following value is from Q-Chem. Off because of the different grid for atomic calculation.
        # ref_guess_energy = -596.0625315683
        # # As a result, we do not check against Q-Chem value, we made a consistency test.
        ref_guess_energy = -596.0618708861276

        assert abs(test_guess_energy - ref_guess_energy) < 1e-7

    def test_sad_guess_rhf_ghost(self):
        mol = pyscf.M(
            atom = """
                O   0.00000000   0.00000000   0.00000000
                H   0.94361690   0.00000000   0.26468890
                H  -0.47180845   0.81719736   0.26468890
                GHOST:H  -0.47180845  -0.81719736   0.26468890
            """,
            basis = "def2-svp",
            charge = 0,
            verbose = 0,
        )

        mf = RHF(mol)

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ref_guess_energy = -75.7944143470

        assert abs(test_guess_energy - ref_guess_energy) < 2e-7

        # Direct above, DF below

        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ref_guess_energy = -67.1808492499

        assert abs(test_guess_energy - ref_guess_energy) < 2e-5

    def test_sad_guess_rhf_pcm(self):
        mol = pyscf.M(
            atom = """
                H      1.0686     -0.1411      1.0408
                C      0.5979      0.0151      0.0688
                H      1.2687      0.2002     -0.7717
                O     -0.5960     -0.0151     -0.0686
            """,
            basis = "def2-svp",
            verbose = 0,
        )

        mf = RHF(mol).PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = 80.0

        dm0 = mf.init_guess_by_sad()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Add the following
        # $rem
        # SOLVENT_METHOD       PCM
        # $end

        # $pcm
        # Theory IEFPCM
        # HeavyPoints 302
        # HPoints 302
        # $end

        # $solvent
        #    Dielectric 80.0
        # $end
        ref_guess_energy = -114.0059265428

        assert abs(test_guess_energy - ref_guess_energy) < 2e-7

    def test_sad_hf_guess_rhf(self):
        mol = pyscf.M(
            atom = """
                H      1.2001      0.0363      0.8431
                C      0.7031      0.0083     -0.1305
                H      0.9877      0.8943     -0.7114
                H      1.0155     -0.8918     -0.6742
                O     -0.6582     -0.0067      0.1730
                H     -1.1326     -0.0311     -0.6482
            """,
            basis = "cc-pvdz", # nctr != 1
            verbose = 0,
        )

        mf = RHF(mol).density_fit(auxbasis = "def2-universal-jkfit")

        dm0 = mf.init_guess_by_sad_hf()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Modify the following
        # SCF_GUESS            SAD
        # BASIS                cc-pvdz
        ref_guess_energy = -100.6465454190

        assert abs(test_guess_energy - ref_guess_energy) < 3e-5

    def test_sad_hf_guess_rks(self):
        mol = pyscf.M(
            atom = """
                H      1.2001      0.0363      0.8431
                C      0.7031      0.0083     -0.1305
                H      0.9877      0.8943     -0.7114
                H      1.0155     -0.8918     -0.6742
                O     -0.6582     -0.0067      0.1730
                H     -1.1326     -0.0311     -0.6482
            """,
            basis = "cc-pvdz", # nctr != 1
            verbose = 0,
        )

        mf = RKS(mol, xc = "wB97MV")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (50,194)
        mf.nlcgrids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.nlcgrids.prune = None
        mf.nlcgrids.radii_adjust = None

        dm0 = mf.init_guess_by_sad_hf()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ### Modify the following
        # SCF_GUESS            SAD
        # BASIS                cc-pvdz
        ref_guess_energy = -115.8971071820

        assert abs(test_guess_energy - ref_guess_energy) < 2e-7

    def test_sad_hf_guess_uks(self):
        mol = pyscf.M(
            atom = """
                O 0 0 1
                O 0 1.2 1
            """,
            basis = "def2-tzvp",
            spin = 2,
            verbose = 0,
        )

        mf = UKS(mol, xc = "r2scan")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None

        dm0 = mf.init_guess_by_sad_hf()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ref_guess_energy = -150.5207824299

        assert abs(test_guess_energy - ref_guess_energy) < 4e-7

        # Direct above, DF below

        mf = UKS(mol, xc = "r2scan").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None

        dm0 = mf.init_guess_by_sad_hf()

        test_guess_energy = get_qchem_sad_guess_energy(mf, dm0)

        ref_guess_energy = -150.5208015220

        assert abs(test_guess_energy - ref_guess_energy) < 3e-5

    def test_minao_heavy_atom_fallback(self):
        mol = pyscf.M(atom='O 0 0 0; O 0 0 1.2', basis='cc-pvdz', verbose=0)
        ref = hf.init_guess_by_atom(mol)
        with mock.patch.object(hf, 'MINAO_MAX_Z', 1):
            dm = hf.init_guess_by_minao(mol)
        assert abs(ref - dm).max().get() < 1e-14

    def test_minao_high_spin_rhf(self):
        mol = pyscf.M(atom='O 0 0 0; O 0 0 1.2', basis='cc-pvdz', verbose=0)
        mf = mol.RHF().to_gpu()
        dm1 = mf.get_init_guess(key='minao')
        dm2 = mf.get_init_guess(key='minao_high_spin')
        assert abs(dm1 - dm2).max().get() < 1e-14

    def _check_direct_high_spin_guess(self, mf, name):
        method = getattr(mf, 'init_guess_by_' + name)
        dm = method(high_spin=True)
        reference = mf.get_init_guess(key=name+'_high_spin')
        cp.testing.assert_allclose(dm, reference, atol=1e-12)
        cp.testing.assert_allclose(dm.mo_coeff, reference.mo_coeff, atol=1e-12)
        cp.testing.assert_allclose(dm.mo_occ, reference.mo_occ, atol=1e-12)
        ordinary = method(high_spin=False)
        cp.testing.assert_allclose(ordinary, mf.get_init_guess(key=name), atol=1e-12)

    def _check_high_spin_rohf_density_fitting(self, key):
        mol = pyscf.M(atom='O 0 0 0; O 0 0 1', basis='cc-pvdz', spin=2, verbose=0)
        mf = rohf.ROHF(mol).density_fit(auxbasis='weigend')
        self._check_direct_high_spin_guess(mf, key.split('_')[0])
        dm = mf.get_init_guess(key=key)
        self.assertEqual(dm.shape, (2, mol.nao, mol.nao))
        self.assertEqual(dm.mo_coeff.ndim, 2)
        self.assertEqual(dm.mo_occ.ndim, 1)
        # Fractional O occupations must survive conversion of ROHF tags.
        self.assertTrue(bool(cp.any((dm.mo_occ > 1) & (dm.mo_occ < 2))))
        cp.testing.assert_allclose(mf.make_rdm1(dm.mo_coeff, dm.mo_occ), dm, atol=1e-12)
        veff = mf.get_veff(mol, dm)
        plain_dm = cp.array(dm)  # Strip tags to exercise density-based J/K.
        self.assertFalse(hasattr(plain_dm, 'mo_coeff'))
        reference = mf.get_veff(mol, plain_dm)
        cp.testing.assert_allclose(veff, reference, atol=1e-9, rtol=1e-9)


    def test_minao_high_spin_rohf_density_fitting(self):
        self._check_high_spin_rohf_density_fitting('minao_high_spin')

    def test_atom_high_spin_rohf_density_fitting(self):
        self._check_high_spin_rohf_density_fitting('atom_high_spin')

    def test_minao_high_spin_uhf(self):
        mol = pyscf.M(atom='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self._check_direct_high_spin_guess(UHF(mol), 'minao')
        dm = UHF(mol).get_init_guess(key='minao_high_spin')
        reference = hf.init_guess_by_minao(mol, high_spin=True)
        self.assertEqual(dm.shape, (2, mol.nao, mol.nao))
        self.assertEqual(dm.mo_coeff.ndim, 3)
        self.assertEqual(dm.mo_occ.ndim, 2)
        # Aligned O and H atomic spins differ from the molecular doublet.
        cp.testing.assert_allclose(dm.mo_occ.sum(axis=1), cp.array([6, 3]), atol=1e-12)
        cp.testing.assert_allclose(dm, reference, atol=1e-12)
        c, occ = dm.mo_coeff, dm.mo_occ
        cp.testing.assert_allclose(dm, (c*occ[:, None]) @ c.transpose(0, 2, 1), atol=1e-12)

    def test_minao_high_spin_ghf(self):
        mol = pyscf.M(atom='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self._check_direct_high_spin_guess(ghf.GHF(mol), 'minao')
        dm = ghf.GHF(mol).get_init_guess(key='minao_high_spin')
        reference = hf.init_guess_by_minao(mol, high_spin=True)
        nao = mol.nao
        self.assertEqual(dm.shape, (2*nao, 2*nao))
        self.assertEqual(dm.mo_coeff.ndim, 2)
        self.assertEqual(dm.mo_occ.ndim, 1)
        cp.testing.assert_allclose(dm[:nao, :nao], reference[0], atol=1e-12)
        cp.testing.assert_allclose(dm[nao:, nao:], reference[1], atol=1e-12)
        cp.testing.assert_allclose(dm[:nao, nao:], 0, atol=1e-12)
        cp.testing.assert_allclose(dm[nao:, :nao], 0, atol=1e-12)
        cp.testing.assert_allclose(dm, (dm.mo_coeff*dm.mo_occ) @ dm.mo_coeff.T, atol=1e-12)

    def test_atom_high_spin_direct_uhf(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self._check_direct_high_spin_guess(UHF(mol), 'atom')

    def test_atom_high_spin_direct_ghf(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        self._check_direct_high_spin_guess(ghf.GHF(mol), 'atom')

    def test_rohf_spin_averaged_guess_tags(self):
        mol = gto.M(atom='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1, verbose=0)
        mf = rohf.ROHF(mol).density_fit(auxbasis='weigend')
        dm = mf.init_guess_by_minao()
        cp.testing.assert_allclose(dm[0], dm[1], atol=1e-12)
        cp.testing.assert_allclose(dm, mf.make_rdm1(dm.mo_coeff, dm.mo_occ), atol=1e-12)
        cp.testing.assert_allclose(mf.get_veff(mol, dm), mf.get_veff(mol, cp.array(dm)), atol=1e-9)

if __name__ == "__main__":
    print("Full Tests for initial guess")
    unittest.main()
