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
import pyscf
import gpu4pyscf
from gpu4pyscf.tdscf import ris

def numerical_tddft_gradient(mol, get_energy, dx = 1e-4):
    numerical_gradient = np.zeros([mol.natm, 3])

    mol_copy = mol.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords(unit='Bohr')
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            e_p = get_energy(mol_copy)

            xyz_m = mol.atom_coords(unit='Bohr')
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            e_m = get_energy(mol_copy)

            numerical_gradient[i_atom, i_xyz] = (e_p - e_m) / (2 * dx)

    np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    print(repr(numerical_gradient))
    return numerical_gradient


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.remove_overlap_zero_eigenvalue = gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue
        cls.overlap_zero_eigenvalue_threshold = gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = cls.remove_overlap_zero_eigenvalue
        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = cls.overlap_zero_eigenvalue_threshold

    def test_rtddft_diffuse_against_qchem(self):
        mol = pyscf.M(
            atom = """
                F      0.675261    0.254864    1.141395
                C      0.000000    0.000000    0.000000
                H      0.205100    0.824000   -0.678600
                H      0.334500   -0.931400   -0.449600
                H     -1.121420   -0.177825   -0.199081
            """,
            basis = "aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-3

        mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None
        mf.small_rho_cutoff = 1e-30
        mf.conv_tol = 1e-10
        test_scf_energy = mf.kernel()
        assert mf.converged

        td = mf.TDDFT().set(nstates = 3)
        assert td.device == 'gpu'
        td.conv_tol = 1e-6
        td.singlet = True
        test_singlet_energies, _ = td.kernel()
        assert np.all(td.converged)
        test_singlet_energies += test_scf_energy

        gobj = td.nuc_grad_method()
        gobj.state = 1 # CIS_STATE_DERIV 3
        test_singlet_1_gradient = gobj.kernel()

        td = mf.TDDFT().set(nstates = 3)
        td.singlet = False
        test_triplet_energies, _ = td.kernel()
        assert np.all(td.converged)
        test_triplet_energies += test_scf_energy

        gobj = td.nuc_grad_method()
        gobj.state = 2 # CIS_STATE_DERIV 2
        test_triplet_2_gradient = gobj.kernel()

        # def get_singlet_1_energy(mol):
        #     mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        #     mf.grids.atom_grid = (99,590)
        #     mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        #     mf.grids.prune = None
        #     mf.grids.radii_adjust = None
        #     mf.small_rho_cutoff = 1e-30
        #     mf.conv_tol = 1e-10
        #     test_scf_energy = mf.kernel()
        #     assert mf.converged

        #     td = mf.TDDFT().set(nstates = 3)
        #     assert td.device == 'gpu'
        #     td.conv_tol = 1e-6
        #     td.singlet = True
        #     test_singlet_energies, _ = td.kernel()
        #     assert np.all(td.converged)
        #     test_singlet_energies += test_scf_energy
        #     return test_singlet_energies[0]
        # ref_singlet_1_gradient = numerical_tddft_gradient(mol, get_singlet_1_energy)

        # def get_triplet_2_energy(mol):
        #     mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        #     mf.grids.atom_grid = (99,590)
        #     mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        #     mf.grids.prune = None
        #     mf.grids.radii_adjust = None
        #     mf.small_rho_cutoff = 1e-30
        #     mf.conv_tol = 1e-10
        #     test_scf_energy = mf.kernel()
        #     assert mf.converged

        #     td = mf.TDDFT().set(nstates = 3)
        #     assert td.device == 'gpu'
        #     td.conv_tol = 1e-6
        #     td.singlet = False
        #     test_triplet_energies, _ = td.kernel()
        #     assert np.all(td.converged)
        #     test_triplet_energies += test_scf_energy
        #     return test_triplet_energies[1]
        # ref_triplet_2_gradient = numerical_tddft_gradient(mol, get_triplet_2_energy)

        ### Q-Chem input
        ### Q-Chem TDDFT gradient is very different from pyscf result, so we use finite difference as the reference for TDDFT gradient instead.
        # $rem
        # JOBTYPE sp
        # RPA True
        # CIS_N_ROOTS 6
        # CIS_CONVERGENCE 8
        # METHOD PBE0
        # BASIS aug-cc-pvdz
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # XC_GRID       000099000590
        # BECKE_SHIFT UNSHIFTED
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # ri_j        True
        # ri_k        True
        # aux_basis     RIJK-def2-TZVP
        # SCF_CONVERGENCE 10
        # THRESH        14
        # BASIS_LIN_DEP_THRESH 3
        # $end
        ref_scf_energy = -139.5904389422
        ref_singlet_energies = np.array([-139.28735232, -139.27452668, -139.26901821,])
        ref_triplet_energies = np.array([-139.30767598, -139.29226247, -139.27926973,])
        ref_singlet_1_gradient = np.array([
            [ 0.0341825081306979,  0.0156161539166533,  0.0206368730459872],
            [-0.0652173824278179, -0.0379486081669711, -0.0128495858575661],
            [ 0.0125513142279488, -0.0239322515938056,  0.0057381774354326],
            [ 0.0046570089295983,  0.0401830611451715, -0.0031792475851944],
            [ 0.0138265522764414,  0.006081655357093 , -0.0103462176070934],
        ])
        ref_triplet_2_gradient = np.array([
            [ 0.00985549405641  ,  0.0101892580062213,  0.00632112431731  ],
            [-0.0128975244706453, -0.0270148710512785,  0.0051668347111899],
            [ 0.0083609690193498, -0.0330941698223342,  0.0021627970170357],
            [-0.0013813220789416,  0.0462163322367815, -0.0089571662442722],
            [-0.0039375916571771,  0.0037138477182452, -0.0046935949171711],
        ])

        ### Reference with BASIS_LIN_DEP_THRESH 10 (not removing any linear dependency)
        # ref_scf_energy = -139.5910409246
        # ref_singlet_energies = np.array([-139.28772480, -139.27509670, -139.26944616,])
        # ref_triplet_energies = np.array([-139.30814946, -139.29271798, -139.27977349,])
        # ref_singlet_1_gradient = np.array([
        #     [ 0.0341510596513217,  0.0156179385157884,  0.0206547967707138],
        #     [-0.0657904273282384, -0.0378705796322265, -0.0114566792319692],
        #     [ 0.0125930571925892, -0.0232024312651902,  0.0049629416309926],
        #     [ 0.0047988601181714,  0.0393317939995086, -0.003713287952678 ],
        #     [ 0.0142474510766988,  0.0061232900350205, -0.0104477717854934],
        # ])
        # ref_triplet_2_gradient = np.array([
        #     [ 0.0106237588681779,  0.0104388406896305,  0.0071343137619806],
        #     [-0.0142863571284124, -0.0272687016433792,  0.0061527389050298],
        #     [ 0.0084349584028587, -0.032458151508763 ,  0.0013600767090338],
        #     [-0.0012460564846606,  0.0455557507450521, -0.0095190482340968],
        #     [-0.0035257778563391,  0.003731472020263 , -0.0051285741164975],
        # ])

        assert np.abs(test_scf_energy - ref_scf_energy) < 1e-9
        assert np.max(np.abs(test_singlet_energies - ref_singlet_energies)) < 2e-5
        assert np.max(np.abs(test_triplet_energies - ref_triplet_energies)) < 2e-5
        # The finite difference can match down to 1e-5 without removing linear dependent basis functions
        assert np.max(np.abs(test_singlet_1_gradient - ref_singlet_1_gradient)) < 5e-4
        assert np.max(np.abs(test_triplet_2_gradient - ref_triplet_2_gradient)) < 5e-4

    @pytest.mark.slow
    def test_utda_direct_diffuse_against_qchem(self):
        mol = pyscf.M(
            atom = """
                F      0.675261    0.254864    1.141395
                C      0.000000    0.000000    0.000000
                H      0.205100    0.824000   -0.678600
                H      0.334500   -0.931400   -0.449600
            """,
            basis = "aug-cc-pvdz",
            charge = 0,
            spin = 1,
            verbose = 0,
        )

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-3

        mf = mol.UKS(xc = "m06l").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        mf.grids.prune = None
        mf.grids.radii_adjust = None
        mf.small_rho_cutoff = 1e-30
        mf.conv_tol = 1e-10
        test_scf_energy = mf.kernel()
        assert mf.converged

        td = mf.TDA().set(nstates = 3)
        assert td.device == 'gpu'
        td.conv_tol = 1e-6
        test_state_energies, _ = td.kernel()
        assert np.all(td.converged)
        test_state_energies += test_scf_energy

        gobj = td.nuc_grad_method()
        gobj.state = 2
        test_state_2_gradient = gobj.kernel()

        # def get_state_2_energy(mol):
        #     mf = mol.UKS(xc = "m06l").to_gpu()
        #     mf.grids.atom_grid = (99,590)
        #     mf.grids.radi_method = gpu4pyscf.dft.radi.euler_macLaurin
        #     mf.grids.prune = None
        #     mf.grids.radii_adjust = None
        #     mf.small_rho_cutoff = 1e-30
        #     mf.conv_tol = 1e-10
        #     test_scf_energy = mf.kernel()
        #     assert mf.converged

        #     td = mf.TDA().set(nstates = 3)
        #     assert td.device == 'gpu'
        #     td.conv_tol = 1e-6
        #     test_state_energies, _ = td.kernel()
        #     assert np.all(td.converged)
        #     test_state_energies += test_scf_energy
        #     return test_state_energies[1]
        # ref_state_2_gradient = numerical_tddft_gradient(mol, get_state_2_energy)
        # print(repr(ref_state_2_gradient))

        ### Q-Chem input
        ### Q-Chem TDDFT gradient is very different from pyscf result, so we use finite difference as the reference for TDDFT gradient instead.
        # $rem
        # JOBTYPE sp
        # UNRESTRICTED TRUE
        # RPA FALSE
        # CIS_N_ROOTS 6
        # CIS_CONVERGENCE 8
        # METHOD r2scan
        # BASIS aug-cc-pvdz
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # XC_GRID       000099000590
        # BECKE_SHIFT UNSHIFTED
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # BASIS_LIN_DEP_THRESH 3
        # $end
        ref_scf_energy = -139.0571419388
        ref_state_energies = np.array([-138.86083206, -138.84488951, -138.82007526,])
        ref_state_2_gradient = np.array([
            [ 0.0627993213697664,  0.0192707379653712,  0.0498325158559965],
            [-0.1376427276511549, -0.0245450483760123, -0.0343138971459211],
            [ 0.0408006751229095, -0.0235423952688052, -0.0011358186213783],
            [ 0.0340427284584166,  0.0288167176165643, -0.0143828094678611],
        ])

        # ### Reference with BASIS_LIN_DEP_THRESH 10 (not removing any linear dependency)
        # ref_scf_energy = -139.0576173854
        # ref_state_energies = np.array([-138.86133008, -138.84846150, -138.82054757,])
        # ref_state_2_gradient = np.array([
        #     [ 0.0648373651301881,  0.0198196728717903,  0.0514799437212332],
        #     [-0.1406236054890542, -0.0260033536392257, -0.0386600272861415],
        #     [ 0.0414164493633962, -0.024805980842757 ,  0.0002403007215435],
        #     [ 0.034369789716493 ,  0.0309896624628436, -0.0130602170145266],
        # ])

        assert np.abs(test_scf_energy - ref_scf_energy) < 1e-7
        assert np.max(np.abs(test_state_energies - ref_state_energies)) < 1e-6
        # The finite difference can match down to 1e-4 without removing linear dependent basis functions
        assert np.max(np.abs(test_state_2_gradient - ref_state_2_gradient)) < 1e-3

    def test_rtddft_ris_diffuse(self):
        mol = pyscf.M(
            atom = """
                F      0.675261    0.254864    1.141395
                C      0.000000    0.000000    0.000000
                H      0.205100    0.824000   -0.678600
                H      0.334500   -0.931400   -0.449600
                H     -1.121420   -0.177825   -0.199081
            """,
            basis = "aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-3

        mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        test_scf_energy = mf.kernel()
        assert mf.converged

        td = ris.TDDFT(mf, gram_schmidt = True, Ktrunc = 0.0).set(nstates = 3)
        assert td.device == 'gpu'
        td.conv_tol = 1e-6
        td.singlet = True
        test_state_energies, X, Y, test_oscillator_strength, test_rotatory_strength = td.kernel()
        assert np.all(td.converged)

        test_state_energies = test_state_energies.get()
        test_oscillator_strength = test_oscillator_strength.get()
        test_rotatory_strength = test_rotatory_strength.get()

        gobj = td.nuc_grad_method()
        gobj.state = 1
        gobj.ris_zvector_solver = True
        test_singlet_1_gradient = gobj.kernel()

        # Reference value are obtained without basis function linear dependency removal.
        # Since no better reference is avaible, this test only gaurantees
        # the result after removal is not crazily different from the result before removal.
        ref_scf_energy = -139.59107759417518
        ref_state_energies = np.array([8.112218, 8.388234, 8.605603])
        ref_oscillator_strength = np.array([0.05335856 , 0.004130934, 0.0328691  ])
        ref_rotatory_strength = np.array([1.112451, 2.362968, 1.480116])
        ref_singlet_1_gradient = np.array([
            [ 0.0416032816945595,  0.0186141257232186,  0.0345957891397397],
            [-0.0735629965701481, -0.0433333304490073, -0.0310370001051767],
            [ 0.0127040043417047, -0.0232895707287626,  0.0059528018976145],
            [ 0.0055393196959162,  0.0408127208371667, -0.0019930247219671],
            [ 0.0138523998196531,  0.0073391344779227, -0.0070677916831486],
        ])

        assert np.abs(test_scf_energy - ref_scf_energy) < 1e-3
        assert np.max(np.abs(test_state_energies - ref_state_energies)) < 1e-2
        assert np.max(np.abs(test_oscillator_strength - ref_oscillator_strength)) < 1e-3
        assert np.max(np.abs(test_rotatory_strength - ref_rotatory_strength)) < 1e-1
        assert np.max(np.abs(test_singlet_1_gradient - ref_singlet_1_gradient)) < 5e-3

    def test_rtda_ris_diffuse(self):
        mol = pyscf.M(
            atom = """
                F      0.675261    0.254864    1.141395
                C      0.000000    0.000000    0.000000
                H      0.205100    0.824000   -0.678600
                H      0.334500   -0.931400   -0.449600
                H     -1.121420   -0.177825   -0.199081
            """,
            basis = "d-aug-cc-pvdz",
            charge = 0,
            verbose = 0,
        )

        gpu4pyscf.scf.hf.overlap_zero_eigenvalue_threshold = 1e-4

        mf = mol.RKS(xc = "PBE0").density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        test_scf_energy = mf.kernel()
        assert mf.converged

        td = ris.TDA(mf, gram_schmidt = True, Ktrunc = 0.0).set(nstates = 3)
        assert td.device == 'gpu'
        td.conv_tol = 1e-5
        td.singlet = True
        test_state_energies, X, test_oscillator_strength, test_rotatory_strength = td.kernel()
        assert np.all(td.converged)

        test_state_energies = test_state_energies.get()
        test_oscillator_strength = test_oscillator_strength.get()
        test_rotatory_strength = test_rotatory_strength.get()

        nacobj = td.nac_method()
        nacobj.states = (1,2)
        test_nac_12, _, _, _ = nacobj.kernel()

        # Reference value are obtained without basis function linear dependency removal.
        # Since no better reference is avaible, this test only gaurantees
        # the result after removal is not crazily different from the result before removal.
        ref_scf_energy = -139.5915368842144
        ref_state_energies = np.array([8.079671, 8.31746 , 8.568331])
        ref_oscillator_strength = np.array([0.0494643 , 0.00382414, 0.02974616])
        ref_rotatory_strength = np.array([1.07563043, 2.71594615, 0.76651065])
        ref_nac_12 = np.array([
            [-0.01251436, -0.00414851, -0.01766956],
            [ 0.00089228, -0.00346726,  0.00349558],
            [ 0.00221547,  0.00391658,  0.0020338 ],
            [ 0.00293909,  0.00069779,  0.00458678],
            [ 0.0065278 ,  0.00303701,  0.00768246],
        ])

        assert np.abs(test_scf_energy - ref_scf_energy) < 1e-4
        assert np.max(np.abs(test_state_energies - ref_state_energies)) < 5e-3
        assert np.max(np.abs(test_oscillator_strength - ref_oscillator_strength)) < 2e-4
        assert np.max(np.abs(test_rotatory_strength - ref_rotatory_strength)) < 5e-2
        assert np.max(np.abs(test_nac_12 - ref_nac_12)) < 5e-4


if __name__ == "__main__":
    print("TDDFT Tests for System with Diffuse Orbitals (Ill-conditioned Overlap Matrices)")
    unittest.main()
