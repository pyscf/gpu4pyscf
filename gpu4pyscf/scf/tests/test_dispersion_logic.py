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
from pyscf import gto, scf
from gpu4pyscf.scf import dispersion

class KnownKS(scf.hf.KohnShamDFT):
    def __init__(self, xc='b3lyp'):
        self.xc = xc
        self.disp = None

class KnownHF(scf.hf.SCF):
    def __init__(self):
        self.disp = None

class TestDispersionLogic(unittest.TestCase):
    def test_parse_dft_with_d3_suffix(self):
        self.assertEqual(
            dispersion.parse_dft('b3lyp-d3bj'),
            ('b3lyp', '', 'd3bj')
        )

    def test_parse_dft_with_hyphenated_d4_method(self):
        self.assertEqual(
            dispersion.parse_dft('wb97x-d4:wb97x-2008'),
            ('wb97x', '', 'd4:wb97x-2008')
        )

    def test_dispersion_mol(self):
        from pyscf.pbc import gto as pbcgto
        mol = gto.M(atom='Na 0 0 0; H 0 0 1.9; ghost-H 0 1 0',
                    basis='lanl2dz', ecp={'Na': 'lanl2dz'}, verbose=0)
        real_mol, atom_idx = dispersion._get_dispersion_mol(mol)
        np.testing.assert_array_equal(atom_idx, [0,1])
        np.testing.assert_array_equal(real_mol.atom_charges(), mol.atom_charges()[:2])
        self.assertTrue(real_mol.has_ecp())
        self.assertEqual(mol.natm, 3)
        self.assertIs(dispersion._get_dispersion_mol(real_mol)[0], real_mol)

        for fractional in [False, True]:
            cell = pbcgto.M(atom='H 0 0 0; ghost-H 0 2 0; H 0 0 1',
                           a=np.eye(3)*8, dimension=1, low_dim_ft_type='inf_vacuum', fractional=fractional, verbose=0)
            real_cell, atom_idx = dispersion._get_dispersion_mol(cell)
            np.testing.assert_array_equal(atom_idx, [0,2])
            np.testing.assert_allclose(real_cell.lattice_vectors(), cell.lattice_vectors())
            np.testing.assert_allclose(real_cell.atom_coords(), cell.atom_coords()[atom_idx])
            self.assertEqual(real_cell.dimension, cell.dimension)
            self.assertEqual(cell.natm, 3)

    def test_parse_disp_none(self):
        # Case 1: All None
        self.assertEqual(dispersion.parse_disp(None, None), (None, None, False))

    def test_parse_disp_explicit(self):
        # Case 2: Explicit disp takes precedence
        # b3lyp normally has no disp.
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj'), ('b3lyp', 'd3bj', False))

        # disp with colon override method
        self.assertEqual(dispersion.parse_disp(None, 'd4:wb97x-3c'), ('wb97x-3c', 'd4', True))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj:pbe'), ('pbe', 'd3bj', False))

        # disp with suffix
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj2b'), ('b3lyp', 'd3bj', False))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bjatm'), ('b3lyp', 'd3bj', True))

        # d4 always implies 3body
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd4'), ('b3lyp', 'd4', True))

    def test_parse_disp_from_method(self):
        # Case 3: Infer from method
        # b3lyp -> no disp
        self.assertEqual(dispersion.parse_disp('b3lyp'), (None, None, False))

        # wb97x-d3bj -> d3bj
        self.assertEqual(dispersion.parse_disp('wb97x-d3bj'), ('wb97x', 'd3bj', False))
        self.assertEqual(dispersion.parse_disp('pbe-d3cso'), ('pbe', 'd3cso', False))

        # wb97x-d4s -> d4s
        self.assertEqual(dispersion.parse_disp('wb97x-d4s'), ('wb97x', 'd4s', True))

        # wb97x-3c -> d4, 3body=True (from whitelist)
        self.assertEqual(dispersion.parse_disp('wb97x-3c'), ('wb97x-3c', 'd4', True))

        # b97-3c -> d3bj with ATM 3-body (from whitelist; per the B97-3c paper)
        self.assertEqual(dispersion.parse_disp('b97-3c'), ('b97-3c', 'd3bj', True))

        # r2scan-3c -> d4, 3body=True (from whitelist)
        self.assertEqual(dispersion.parse_disp('r2scan-3c'), ('r2scan-3c', 'd4', True))

        # underscore spellings are equivalent to the dash spellings
        self.assertEqual(dispersion.parse_disp('b97_3c'), ('b97-3c', 'd3bj', True))
        self.assertEqual(dispersion.parse_disp('r2scan_3c'), ('r2scan-3c', 'd4', True))

        # wb97x-d4 with explicit d4 method should preserve the hyphenated payload
        self.assertEqual(
            dispersion.parse_disp('wb97x-d4:wb97x-2008'),
            ('wb97x-2008', 'd4', True)
        )

    def test_parse_dft_3c_spellings(self):
        # The same (xc, nlc, disp) triples as pyscf.scf.dispersion.parse_dft
        self.assertEqual(dispersion.parse_dft('b97-3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('b97_3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('gga_xc_b97_3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('r2scan-3c'), ('r2scan', False, 'd4:r2scan-3c'))
        self.assertEqual(dispersion.parse_dft('r2scan_3c'), ('r2scan', False, 'd4:r2scan-3c'))
        self.assertEqual(dispersion.parse_dft('wb97x-3c'), ('wb97x-v', False, 'd4:wb97x-3c'))
        self.assertEqual(dispersion.parse_dft('wb97x_3c'), ('wb97x-v', False, 'd4:wb97x-3c'))

        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('pbeh-3c')
        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('hf-3c')

    def test_parse_disp_errors(self):
        # Unknown disp version
        with self.assertRaises(ValueError):
            dispersion.parse_disp('b3lyp', 'unknown_ver')

        # Disp specified but method unknown/missing (if disp string doesn't contain colon)
        # Actually parse_disp(None, 'd3bj') -> raises ValueError "the method used in dispersion d3bj is not specified."
        with self.assertRaises(ValueError):
            dispersion.parse_disp(None, 'd3bj')

    def test_check_disp(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')

        # 1. RHF object (no .xc)
        mf_hf = scf.RHF(mol)
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If mf.disp = None
        mf_hf.disp = None
        # parse_disp('hf', None) -> ('hf', None, False) -> check_disp returns False
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If we set mf.disp = 'd3bj'
        mf_hf.disp = 'd3bj'
        self.assertTrue(dispersion.check_disp(mf_hf))

        # 2. RKS object (has .xc)
        mf_dft = scf.RKS(mol)
        mf_dft.xc = 'b3lyp'
        mf_dft.disp = None

        # b3lyp -> no disp -> False
        self.assertFalse(dispersion.check_disp(mf_dft))

        # Explicit disp
        self.assertTrue(dispersion.check_disp(mf_dft, disp='d3bj'))

        # Explicit disp=False
        self.assertFalse(dispersion.check_disp(mf_dft, disp=False))

        # Implicit disp from method
        mf_dft.xc = 'wb97x-d3bj'
        self.assertTrue(dispersion.check_disp(mf_dft))

        # Unsupported disp version
        with self.assertRaises(ValueError):
            dispersion.check_disp(mf_dft, disp='unsupported')

        mf_dft = scf.RKS(mol).to_gpu()
        mf_dft.xc = 'b3lyp'
        mf_dft.disp = None
        # b3lyp -> no disp -> False
        self.assertFalse(dispersion.check_disp(mf_dft))

        # Explicit disp
        self.assertTrue(dispersion.check_disp(mf_dft, disp='d3bj'))

        # Explicit disp=False
        self.assertFalse(dispersion.check_disp(mf_dft, disp=False))

        # Implicit disp from method
        mf_dft.xc = 'wb97x-d3bj'
        self.assertTrue(dispersion.check_disp(mf_dft))

        # Unsupported disp version
        with self.assertRaises(ValueError):
            dispersion.check_disp(mf_dft, disp='unsupported')

    def test_parse_coach(self):
        xc = 'COACH'
        self.assertEqual(dispersion.parse_dft(xc), ('coach', '', 'd4:coach'))
        self.assertEqual(dispersion.parse_disp(xc), ('coach', 'd4', True))
        self.assertTrue(dispersion.check_disp(KnownKS(xc)))
        self.assertEqual(dispersion.parse_disp(xc, 'd4:pbe'), ('pbe', 'd4', True))

if __name__ == "__main__":
    unittest.main()
