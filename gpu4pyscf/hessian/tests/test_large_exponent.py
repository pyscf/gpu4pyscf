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
import cupy as cp
import pyscf
from gpu4pyscf import scf, dft

def setUpModule():
    global mol_minimal_one_atom, mol_minimal_two_atom, auxbasis_minimal_good, auxbasis_minimal_bad
    mol_minimal_one_atom = pyscf.M(
        atom = """
        He 100 200 300
        """,
        basis = """
        H    S
            100000.0 1.0
        """,
        verbose = 0,
        output='/dev/null',
    )
    mol_minimal_two_atom = pyscf.M(
        atom = """
        H 0   0.2 0
        H 1.0 0.1 0
        """,
        basis = """
        H    S
            100000.0 1.0
        """,
        verbose = 0,
        output='/dev/null',
    )
    auxbasis_minimal_good = """
        H    S
            200000.0 1.0
    """
    auxbasis_minimal_bad = """
        H    S
            2.0 1.0
    """

def tearDownModule():
    global mol_minimal_one_atom, mol_minimal_two_atom
    mol_minimal_one_atom.stdout.close()
    del mol_minimal_one_atom
    mol_minimal_two_atom.stdout.close()
    del mol_minimal_two_atom

class KnownValues(unittest.TestCase):
    def test_hessian_large_exp_one_atom_rhf(self):
        mf = scf.HF(mol_minimal_one_atom)
        mf.conv_tol = 1e-10

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        # Zero hessian for only one atom
        assert np.max(np.abs(hessian)) < 1e-4
        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_two_atom_rhf(self):
        mf = scf.HF(mol_minimal_two_atom)
        mf.conv_tol = 1e-10

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_one_atom_rhf_density_fit(self):
        mf = scf.HF(mol_minimal_one_atom)
        mf.conv_tol = 1e-10
        mf = mf.density_fit(auxbasis = auxbasis_minimal_good)

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        # Zero hessian for only one atom
        assert np.max(np.abs(hessian)) < 1e-4
        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_two_atom_rhf_density_fit(self):
        mf = scf.HF(mol_minimal_two_atom)
        mf.conv_tol = 1e-10
        mf = mf.density_fit(auxbasis = auxbasis_minimal_good)

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_two_atom_rhf_density_fit_bad_auxbasis(self):
        mf = scf.HF(mol_minimal_two_atom)
        mf.conv_tol = 1e-10
        mf = mf.density_fit(auxbasis = auxbasis_minimal_bad)

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_one_atom_rks(self):
        mf = dft.RKS(mol_minimal_one_atom, xc = "r2SCAN")
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.grid_response = True
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        # Zero hessian for only one atom
        assert np.max(np.abs(hessian)) < 1e-4
        assert np.max(np.abs(translation_invariance)) < 1e-4

    def test_hessian_large_exp_two_atom_rks(self):
        mf = dft.RKS(mol_minimal_two_atom, xc = "wB97X")
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        mf.level_shift = 0.001

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.grid_response = True
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        assert np.max(np.abs(translation_invariance)) < 1e-4

    @pytest.mark.skip("Too slow, functionality covered by corresponding tests above")
    def test_hessian_large_exp_methylbromide_rks(self):
        mol_real = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                Br     0.000000    0.000000    1.940000
                H      1.027662    0.000000   -0.363333
                H     -0.513831    0.889981   -0.363333
                H     -0.513831   -0.889981   -0.363333
            """,
            basis = "def2-svp",
            verbose = 4,
            output='/dev/null',
        )

        mf = dft.RKS(mol_real, xc = "wB97X")
        mf.grids.atom_grid = (99,590)
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True
        hessian = hobj.kernel()

        natm = mf.mol.natm
        hessian = hessian.transpose(0,2,1,3)
        hessian = hessian.reshape((natm * 3, natm * 3))
        translation_invariance = np.sum(hessian, axis = 0).reshape(natm, 3)

        assert np.max(np.abs(translation_invariance)) < 1e-7

if __name__ == "__main__":
    print("Edge Case Tests for Hessian Calculation with Large Exponent in Atomic Orbitals")
    unittest.main()
