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
from pyscf.data.nist import BOHR
from gpu4pyscf.sem.gto.mole import Mole


class TestPM6Mole(unittest.TestCase):    
    def setUp(self):
        self.atom_str = '''
        O 0.0000 0.0000 0.0000
        H 0.7570 0.5860 0.0000
        H -0.7570 0.5860 0.0000
        '''
        
    def test_build_geometry_and_topology(self):
        mol = Mole(self.atom_str)
        mol.build()
        
        self.assertEqual(mol.natm, 3, "Should have 3 atoms (O, H, H)")
        
        expected_nao = 4 + 1 + 1
        self.assertEqual(mol.nao, expected_nao, f"NAO should be {expected_nao}")
        
        slices = mol._aoslice
        np.testing.assert_array_equal(slices[0], [0, 4])
        np.testing.assert_array_equal(slices[1], [4, 5])
        np.testing.assert_array_equal(slices[2], [5, 6])
        
        natorb = mol.natorb_per_atom
        np.testing.assert_array_equal(natorb, [4, 1, 1])

    def test_electron_counting(self):
        mol = Mole(self.atom_str, charge=0, spin=0)
        mol.build()
        
        self.assertEqual(mol.nelectron, 8)
        self.assertEqual(mol.nelec, (4, 4))
        
        mol_cation = Mole(self.atom_str, charge=1, spin=1)
        mol_cation.build()
        self.assertEqual(mol_cation.nelectron, 7)
        self.assertEqual(mol_cation.nelec, (4, 3))

    def test_unit_conversion(self):
        mol = Mole("H 0 0 1", unit='Angstrom', spin=1)
        mol.build()
        
        coords_bohr = mol.atom_coords(unit='Bohr')
        self.assertAlmostEqual(coords_bohr[0, 2], 1.0 / BOHR, places=4)
        
        coords_ang = mol.atom_coords(unit='Angstrom')
        self.assertAlmostEqual(coords_ang[0, 2], 1.0, places=5)

    def test_model_arrays_initialization(self):
        mol = Mole(self.atom_str)
        mol.build()
        
        self.assertEqual(mol.uspd[0], -91.678761)
        self.assertEqual(mol.uspd[1], -70.460949)
        self.assertEqual(mol.uspd[4], -11.246958)
        
        self.assertAlmostEqual(mol.eta_1e[0, 0], 5.421751)
        self.assertAlmostEqual(mol.eta_1e[0, 1], 2.27096)
        self.assertAlmostEqual(mol.eta_1e[1, 0], 1.268641)
        self.assertAlmostEqual(mol.eta_1e[2, 0], 1.268641)

        self.assertEqual(mol.principal_quantum_number_s[0], 2)
        self.assertEqual(mol.principal_quantum_number_d[0], 3)
        self.assertTrue(not mol.has_d_orbitals[0])

    def test_interface_compatibility(self):
        mol = Mole(self.atom_str)
        mol.build()
        
        slices = mol.aoslice_by_atom()
        
        self.assertEqual(slices[0, 2], 0)
        self.assertEqual(slices[0, 3], 4)
        self.assertEqual(slices[0, 0], 0)
        self.assertEqual(slices[0, 1], 2)

        self.assertEqual(slices[1, 0], 2)
        self.assertEqual(slices[1, 1], 3)

        ao_labels = mol.ao_labels()
        self.assertEqual(ao_labels[0], '0 O 2s')
        self.assertEqual(ao_labels[1], '0 O 2py')
        self.assertEqual(ao_labels[2], '0 O 2pz')
        self.assertEqual(ao_labels[3], '0 O 2px')
        self.assertEqual(ao_labels[4], '1 H 1s')
        self.assertEqual(ao_labels[5], '2 H 1s')

        # Ne has 3s, 2py, 2pz, 2px !
        mol = Mole("Ne 0.0 0.0 0.0")
        mol.build()
        ao_labels = mol.ao_labels()
        self.assertEqual(ao_labels[0], '0 Ne 3s')
        self.assertEqual(ao_labels[1], '0 Ne 2py')

    def test_pari_params(self):
        mol = Mole(self.atom_str)
        mol.build()
        
        ref_npairs = 3
        ref_pair_i = np.array([1, 2, 2], dtype=np.int32)
        ref_core_charges = np.array([6., 1., 1.])
        ref_norbitals_per_atom = np.array([4, 1, 1], dtype=np.int32)
        ref_has_d_orbitals = np.array([False, False, False])
        ref_am = np.array([0.415415881345135, 0.530979416828876, 0.530979416828876])
        ref_core_rho = np.array([1.203613107859472, 0.941656087134429, 0.941656087134429])
        ref_guess1 = np.array([[-0.017771,  0.      ,  0.      ,  0.      ],
            [ 0.024184,  0.      ,  0.      ,  0.      ],
            [ 0.024184,  0.      ,  0.      ,  0.      ]])
        ref_xfac = np.array([0.192295, 0.192295, 2.243587])
        ref_alpb = np.array([1.260942, 1.260942, 3.540942])
        ref_v_par6_4 = np.array([9.278465,  5.983752,  0.      ,  0.      ])

        assert np.abs(mol.npairs - ref_npairs).max() < 1.0E-13
        assert np.abs(mol.pair_i - ref_pair_i).max() < 1.0E-13
        assert np.abs(mol.core_charges.get() - ref_core_charges).max() < 1.0E-13
        assert np.abs(mol.norbitals_per_atom.get() - ref_norbitals_per_atom).max() < 1.0E-13
        assert np.logical_xor(mol.has_d_orbitals.get(), ref_has_d_orbitals).sum() == 0
        assert np.abs(mol.am.get() - ref_am).max() < 1.0E-13
        assert np.abs(mol.core_rho.get() - ref_core_rho).max() < 1.0E-13
        assert np.abs(mol.guess1.get() - ref_guess1).max() < 1.0E-13
        assert np.abs(mol.xfac.get() - ref_xfac).max() < 1.0E-13
        assert np.abs(mol.alpb.get() - ref_alpb).max() < 1.0E-13
        assert np.abs(mol.v_par6[:4].get() - ref_v_par6_4).max() < 1.0E-13

if __name__ == '__main__':
    print("Full tests for PM6Mole...")
    unittest.main()