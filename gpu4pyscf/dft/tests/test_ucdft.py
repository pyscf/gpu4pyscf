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

import numpy as np
import cupy as cp
import unittest
import pyscf
from gpu4pyscf.dft import uks, ucdft
from gpu4pyscf.dft.cdft_soscf import newton_cdft as newton_penalty
from gpu4pyscf.dft.cdft_soscf_full import newton_cdft

atom = '''
    O    0    0    0
    H    0.   -0.757   0.587
    H    0.   0.757    0.587
'''
bas='def2-tzvp'
auxbas='def2-tzvp-jkfit'
charge_constraints_2 = [ [0, 1], [8.1, 0.95] ]
charge_constraints_3 = [ [0, 1, 2], [8.1, 0.95, 0.95] ]
charge_constraints_2_ao = [ 
    [['0 O 1s', '0 O 2s', '0 O 2px', '0 O 2py', '0 O 2pz'], 1], 
    [8.1, 0.95] ]

def run_dft(mol, xc, method, charge_constraints, 
        soscf=False, penalty=None, projection_method='becke'):
    if method == 'lagrange':
        assert penalty is None
    mf = ucdft.CDFT_UKS(mol,
                       charge_constraints=charge_constraints, 
                       method=method, 
                       penalty_weight=penalty,
                       projection_method=projection_method
                        )
    mf = mf.density_fit(auxbasis=auxbas)
    mf.xc = xc
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1.0E-8
    mf.max_cycle = 100
    if method == 'penalty':
        mf = newton_penalty(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
    elif soscf:
        mf = newton_cdft(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
    else:
        mf.kernel()
        v_lagrange = mf.v_lagrange

    e_tot = mf.e_tot
    homo = mf.mo_energy[0][4]
    lumo = mf.mo_energy[0][5]

    dm = mf.make_rdm1()
    projs = mf.build_projectors()
    O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
    H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])

    return {
        'e_tot': float(e_tot),
        'homo': float(homo),
        'lumo': float(lumo),
        'O_charge': float(O_charge),
        'H1_charge': float(H1_charge),
        'v_lagrange': v_lagrange,
        'mf': mf
    }

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000,
                  verbose=6, spin=0, charge=0, output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_penalty_cons2 = run_dft(mol, 'b3lyp', 'penalty',
            charge_constraints_2, penalty=10, projection_method='becke')
        cls.output_lagrange_soscf_cons2 = run_dft(mol, 'b3lyp', 'lagrange',
            charge_constraints_2, soscf=True, projection_method='becke')
        cls.output_lagrange_nested_cons2 = run_dft(mol, 'b3lyp', 'lagrange',
            charge_constraints_2, projection_method='becke')
        cls.output_lagrange_nested_cons2_minao = run_dft(mol, 'b3lyp', 'lagrange',
            charge_constraints_2, projection_method='minao')

    @classmethod
    def tearDownClass(cls):
        del cls.output_penalty_cons2, cls.output_lagrange_soscf_cons2
        del cls.output_lagrange_nested_cons2, cls.output_lagrange_nested_cons2_minao

    def test_energy(self):
        ref = -76.376907212893 # DF Becke Lagrange reference
        self.assertAlmostEqual(self.output_penalty_cons2['e_tot'], ref, delta=0.05)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['e_tot'], ref, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['e_tot'], ref, 6)

    def test_homo_lumo(self):
        homo_ref = -0.594634337972
        lumo_ref = -0.087117515816
        self.assertAlmostEqual(self.output_penalty_cons2['homo'], homo_ref, delta=0.05)
        self.assertAlmostEqual(self.output_penalty_cons2['lumo'], lumo_ref, delta=0.05)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['homo'], homo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['lumo'], lumo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['homo'], homo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['lumo'], lumo_ref, 6)

    def test_charge(self):
        charge_ref_O = 8.1
        charge_ref_H1 = 0.95

        # A finite quadratic penalty leaves a residual: near the constrained
        # solution, delta_N ~ V / (2 * weight). For O, V ~ -0.45 and weight=10
        # predict about -0.0225 electrons; this is not a rounding tolerance.
        self.assertAlmostEqual(self.output_penalty_cons2['O_charge'], charge_ref_O, delta=0.03)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['O_charge'], charge_ref_O, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['O_charge'], charge_ref_O, 6)
        self.assertAlmostEqual(self.output_penalty_cons2['H1_charge'], charge_ref_H1, delta=0.005)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['H1_charge'], charge_ref_H1, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['H1_charge'], charge_ref_H1, 6)

    def test_multiplier(self):
        # Multipliers are nonvariational and more sensitive than total energies.
        ref_O = -4.50487015E-01
        ref_H = -2.37985800E-08

        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['v_lagrange'][0], ref_O, delta=1e-4)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['v_lagrange'][1], ref_H, delta=1e-4)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['v_lagrange'][0], ref_O, delta=1e-4)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['v_lagrange'][1], ref_H, delta=1e-4)

    def test_minao_projection(self):
        ref_energy = -76.364007406852
        ref_homo = -0.123407786265
        ref_lumo = 0.058274316958
        ref_O_charge = 8.1
        ref_H1_charge = 0.95
        ref_O_multiplier = 0.31278893
        ref_H_multiplier = -0.05760844
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['e_tot'], ref_energy, 7)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['homo'], ref_homo, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['lumo'], ref_lumo, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['O_charge'], ref_O_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['H1_charge'], ref_H1_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['v_lagrange'][0], ref_O_multiplier, delta=1e-4)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2_minao['v_lagrange'][1], ref_H_multiplier, delta=1e-4)

    def test_minao_projection_1ao(self):
        mf = ucdft.CDFT_UKS(mol,
                       charge_constraints=[ ['0 O 2pz'], [1.2] ], 
                       method='lagrange', 
                       projection_method='minao'
                        )
        mf = mf.density_fit(auxbasis=auxbas)
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1.0E-8
        mf.max_cycle = 100
        mf = newton_cdft(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
        e_tot = mf.e_tot
        homo = mf.mo_energy[0][4]
        lumo = mf.mo_energy[0][5]
        dm = mf.make_rdm1()
        projs = mf.build_projectors()
        O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])

        self.assertAlmostEqual(float(O_charge), 1.2, 6)
        self.assertAlmostEqual(float(e_tot), -76.384052803986, 7)
        self.assertAlmostEqual(float(homo), -0.236737660768, 6)
        self.assertAlmostEqual(float(lumo), 0.049445292477, 6)
        self.assertAlmostEqual(float(v_lagrange[0]), 0.288396, delta=1e-4)

    def test_canonical_mo_energy(self):
        mf = self.output_lagrange_nested_cons2_minao['mf']
        mo_energy, mo_coeff = mf.get_canonical_mo()
        dm = mf.make_rdm1()
        fock = (mf.get_hcore() + mf.get_veff(mol, dm)).get()
        overlap = mf.get_ovlp().get()
        np.testing.assert_allclose(
            fock[0] @ mo_coeff[0], (overlap @ mo_coeff[0]) * mo_energy[0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            fock[1] @ mo_coeff[1], (overlap @ mo_coeff[1]) * mo_energy[1], atol=1e-9, rtol=0)
        mf.analyze()

    def test_spin_constraint(self):
        mol = pyscf.M(
            atom='''O    0    0    0
              O    0    0    8
              ''',
            basis=bas,
            max_memory=32000,
            verbose = 1,
            spin = 0,
            charge = 0,
            output = '/dev/null'
        )
        mol_atom = pyscf.M(
            atom='''O    0    0    0
              ''',
            basis=bas,
            max_memory=32000,
            verbose = 1,
            spin = 2,
            charge = 0,
            output = '/dev/null'
        )
        self.addCleanup(mol.stdout.close)
        self.addCleanup(mol_atom.stdout.close)
        mf_atom = uks.UKS(mol_atom).density_fit(auxbasis=auxbas)
        mf_atom.xc = 'b3lyp'
        mf_atom.grids.atom_grid = (99, 590)
        mf_atom.kernel()

        mf = ucdft.CDFT_UKS(mol,
                       spin_constraints= [[0], [2.001657]],
                        )
        mf = mf.density_fit(auxbasis=auxbas)
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-9
        mf.kernel()
        mo_energy = mf.get_canonical_mo()[0]
        # Dissociation and near-degenerate orbital splittings are sensitive
        # to the finite grid; use separate energy and orbital tolerances.
        assert abs(float(mf_atom.e_tot)*2 - float(mf.e_tot)) < 1e-4
        assert abs(float(mo_energy[0][7]) - float(mf_atom.mo_energy[1][2])) < 5e-3
        assert abs(float(mo_energy[0][8]) - float(mf_atom.mo_energy[1][3])) < 5e-3
        assert abs(float(mo_energy[1][7]) - float(mf_atom.mo_energy[1][2])) < 5e-3
        assert abs(float(mo_energy[1][8]) - float(mf_atom.mo_energy[1][3])) < 5e-3

    def test_mixed_charge_and_spin_constraints(self):
        charge_cons = [[0], [8.1]]
        spin_cons = [[0], [0.2]]
        
        mf = ucdft.CDFT_UKS(mol,
                            charge_constraints=charge_cons,
                            spin_constraints=spin_cons,
                            method='lagrange',
                            projection_method='minao')
        mf = mf.density_fit(auxbasis=auxbas)
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.kernel()
        
        dm = mf.make_rdm1()
        projs = mf.build_projectors()
        n_O = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
        m_O = cp.trace(dm[0] @ projs[1]) - cp.trace(dm[1] @ projs[1])
        
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(float(n_O), 8.1, 4)
        self.assertAlmostEqual(float(m_O), 0.2, 4)
        self.assertEqual(len(mf.v_lagrange), 2)
        self.assertAlmostEqual(float(mf.e_tot), -76.361220462326, 6)

    def test_group_constraints(self):
        group_cons = [[[1, 2]], [1.5]]
        
        mf = ucdft.CDFT_UKS(mol,
                            charge_constraints=group_cons,
                            method='lagrange',
                            projection_method='minao')
        mf = mf.density_fit(auxbasis=auxbas)
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (75, 302)
        mf.kernel()
        
        dm = mf.make_rdm1()
        projs = mf.build_projectors()
        n_H_sum = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
        self.assertAlmostEqual(float(n_H_sum), 1.5, 5)
        self.assertAlmostEqual(float(mf.e_tot), -76.445332942880, 6)


class KnownValues(unittest.TestCase):
    def test_charge_potential_signs(self):
        mf = ucdft.CDFT_UKS(mol, charge_constraints=[[0], [8.1]])
        w_charge = mf.build_projectors()[0]
        potential = mf.get_constraint_potential([0.3])
        expected = (0.3 * w_charge).get()
        # Charge constraints shift both spin channels in the same direction.
        np.testing.assert_allclose(potential[0].get(), expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(potential[1].get(), expected, atol=1e-14, rtol=0)

    def test_spin_potential_signs(self):
        mf = ucdft.CDFT_UKS(mol, spin_constraints=[[1], [0.2]])
        w_spin = mf.build_projectors()[0]
        potential = mf.get_constraint_potential([-0.2])
        expected_alpha = (-0.2 * w_spin).get()
        # Spin constraints shift alpha and beta in opposite directions.
        np.testing.assert_allclose(potential[0].get(), expected_alpha, atol=1e-14, rtol=0)
        np.testing.assert_allclose(potential[1].get(), -expected_alpha, atol=1e-14, rtol=0)

    def test_mixed_charge_and_spin_potential_signs(self):
        mf = ucdft.CDFT_UKS(mol,
                            charge_constraints=[[0], [8.1]],
                            spin_constraints=[[1], [0.2]])
        w_charge, w_spin = mf.build_projectors()
        potential = mf.get_constraint_potential([0.3, -0.2])
        expected_alpha = (0.3 * w_charge - 0.2 * w_spin).get()
        expected_beta = (0.3 * w_charge + 0.2 * w_spin).get()
        np.testing.assert_allclose(potential[0].get(), expected_alpha, atol=1e-14, rtol=0)
        np.testing.assert_allclose(potential[1].get(), expected_beta, atol=1e-14, rtol=0)

    def test_error_handling_mismatched_inputs(self):
        groups = [[0], [1]]
        targets = [8.1]
        
        with self.assertRaises(ValueError) as cm:
            ucdft.CDFT_UKS(mol,
                                charge_constraints=[groups, targets])
        self.assertIn("must match the number of targets", str(cm.exception))

    def test_error_handling_becke_orbital_specific(self):
        cons = [[['0 O 2pz']], [1.0]]
        
        mf = ucdft.CDFT_UKS(mol,
                            charge_constraints=cons,
                            projection_method='becke')
        
        with self.assertRaises(ValueError) as cm:
            mf.build_projectors()
        self.assertIn("Becke projection does not support orbital-specific", str(cm.exception))

    def test_precompute_projectors_caching(self):
        mf = ucdft.CDFT_UKS(mol,
                            charge_constraints=[[0], [8.1]],
                            method='lagrange',
                            projection_method='minao')
        mf.xc = 'b3lyp'
        
        projs = mf.build_projectors()
        self.assertIsNotNone(mf.constraint_projectors)
        self.assertEqual(len(projs), 1)
        projs_2 = mf.build_projectors()
        self.assertIs(projs, projs_2)


class PenaltyDensityContracts(unittest.TestCase):
    def setUp(self):
        self.mol = pyscf.M(atom='H 0 0 0; H 0.2 0.1 1.4', unit='Bohr',
                           basis='sto-3g', verbose=0)
        self.mf = ucdft.CDFT_UKS(
            self.mol, charge_constraints=[[0], [1.2]],
            spin_constraints=[[0], [0.25]], method='penalty', penalty_weight=2.)
        self.mf.xc = 'LDA,VWN'
        self.mf.grids.atom_grid = (10, 38)
        overlap = self.mol.intor_symmetric('int1e_ovlp')
        eigenvalues, vectors = np.linalg.eigh(overlap)
        self.orbitals = vectors / np.sqrt(eigenvalues)
        self.dm = np.asarray([np.outer(self.orbitals[:, 0], self.orbitals[:, 0]),
                              np.outer(self.orbitals[:, 1], self.orbitals[:, 1])])

    def test_charge_spin_penalty_orbital_derivative(self):
        potential = self.mf.get_penalty_potential(cp.asarray(self.dm), 2.).get()
        charge_w, spin_w = [w.get() for w in self.mf.constraint_projectors]

        def penalty(density):
            charge = np.einsum('sij,ji->', density, charge_w)
            spin = np.einsum('ij,ji->', density[0] - density[1], spin_w)
            return 2 * ((charge - 1.2)**2 + (spin - 0.25)**2)

        # Rotate occupied into virtual orbitals instead of introducing
        # non-positive densities; each spin retains one electron exactly.
        for spin in (0, 1):
            occupied = self.orbitals[:, spin]
            virtual = self.orbitals[:, 1 - spin]
            tangent = np.outer(occupied, virtual) + np.outer(virtual, occupied)
            expected = np.einsum('ij,ji->', potential[spin], tangent)
            step = 1e-4
            energies = []
            for angle in (step, -step):
                orbital = np.cos(angle) * occupied + np.sin(angle) * virtual
                density = self.dm.copy()
                density[spin] = np.outer(orbital, orbital)
                energies.append(penalty(density))
            self.assertAlmostEqual((energies[0] - energies[1]) / (2 * step),
                                   expected, delta=3e-7)
        # Isolate spin response by removing the common charge contribution.
        charge_error = np.einsum('sij,ji->', self.dm, charge_w) - 1.2
        np.testing.assert_allclose(potential.sum(axis=0),
                                   8 * charge_error * charge_w, atol=1e-13)

    def test_penalty_fock_damping_and_level_shift(self):
        damping, shift = 0.2, 0.3
        dm = cp.asarray(self.dm)
        raw = cp.asarray(self.mf.get_fock(dm=dm)).get()
        swapped = dm[::-1].copy()
        previous = cp.asarray(self.mf.get_fock(dm=swapped))
        overlap = self.mol.intor_symmetric('int1e_ovlp')
        actual = cp.asarray(self.mf.get_fock(
            dm=dm, cycle=0, diis=None, diis_start_cycle=3,
            damp_factor=damping, level_shift_factor=shift,
            fock_last=previous)).get()
        expected = []
        for spin in (0, 1):
            fock = (1 - damping) * raw[spin]
            fock += damping * previous[spin].get()
            fock += shift * (
                overlap - overlap @ self.dm[spin] @ overlap)
            expected.append(fock)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(dm.get(), self.dm)


if __name__ == "__main__":
    print("Full Tests for cdft")
    unittest.main()
