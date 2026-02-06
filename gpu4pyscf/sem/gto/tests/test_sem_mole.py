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
        mol = Mole(self.atom_str, output='/dev/null')
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
        mol = Mole(self.atom_str, charge=0, spin=0, output='/dev/null')
        mol.build()
        
        self.assertEqual(mol.nelectron, 8)
        self.assertEqual(mol.nelec, (4, 4))
        
        mol_cation = Mole(self.atom_str, charge=1, spin=1, output='/dev/null')
        mol_cation.build()
        self.assertEqual(mol_cation.nelectron, 7)
        self.assertEqual(mol_cation.nelec, (4, 3))

    def test_unit_conversion(self):
        mol = Mole("H 0 0 1", unit='Angstrom', spin=1, output='/dev/null')
        mol.build()
        
        coords_bohr = mol.atom_coords(unit='Bohr')
        self.assertAlmostEqual(coords_bohr[0, 2], 1.0 / BOHR, places=4)
        
        coords_ang = mol.atom_coords(unit='Angstrom')
        self.assertAlmostEqual(coords_ang[0, 2], 1.0, places=5)

    def test_model_arrays_initialization(self):
        mol = Mole(self.atom_str, output='/dev/null')
        mol.build()
        
        self.assertEqual(mol.uspd[0], -91.678761)
        self.assertEqual(mol.uspd[1], -70.460949)
        self.assertEqual(mol.uspd[4], -11.246958)
        
        self.assertAlmostEqual(mol.eta_1e[0, 0], 5.421751)
        self.assertAlmostEqual(mol.eta_1e[0, 1], 2.27096)
        self.assertAlmostEqual(mol.eta_1e[1, 0], 1.268641)
        self.assertAlmostEqual(mol.eta_1e[2, 0], 1.268641)

    def test_interface_compatibility(self):
        mol = Mole(self.atom_str, output='/dev/null')
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
        mol = Mole("Ne 0.0 0.0 0.0", output='/dev/null')
        mol.build()
        ao_labels = mol.ao_labels()
        self.assertEqual(ao_labels[0], '0 Ne 3s')
        self.assertEqual(ao_labels[1], '0 Ne 2py')

if __name__ == '__main__':
    print("Full tests for PM6Mole...")
    unittest.main()