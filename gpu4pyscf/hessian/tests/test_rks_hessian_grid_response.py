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

import pyscf
import numpy as np
import cupy as cp
import unittest
import pytest
from gpu4pyscf.dft import RKS
from gpu4pyscf.hessian.rks import _get_exc_deriv2, _get_exc_deriv2_numerical, _get_vxc_deriv1, _get_vxc_deriv1_numerical
from gpu4pyscf.hessian.tests.test_vv10_hessian import numerical_d2e_dft

def setUpModule():
    global mol

    mol = pyscf.M(
        atom = '''
            O  0.0000  0.7375 -0.0528
            O  0.0000 -0.7375 -0.1528
            H  0.8190  0.8170  0.4220
            H -0.8190 -0.8170  1.4220
        ''',
        basis = 'def2-svp',
        charge = 0,
        spin = 0,
        output='/dev/null',
        verbose = 0,
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    # All reference results from the same calculation with mf.level_shift = 0

    def test_hessian_grid_response_d2edAdB_lda(self):
        mf = RKS(mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_d2edAdB_gga(self):
        mf = RKS(mol, xc = 'PBE0')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_d2edAdB_mgga(self):
        mf = RKS(mol, xc = 'wB97M-d3bj')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_dFdA_lda(self):
        mf = RKS(mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    def test_hessian_grid_response_dFdA_gga(self):
        mf = RKS(mol, xc = 'wB97X-V')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    def test_hessian_grid_response_dFdA_mgga(self):
        mf = RKS(mol, xc = 'r2SCAN')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        test_energy = mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    # def test_hessian_grid_response_lda(self):
    #     mf = RKS(mol, xc = 'LDA')
    #     mf.grids.atom_grid = (10,14)
    #     mf.conv_tol = 1e-12
    #     mf.conv_tol_cpscf = 1e-10
    #     # mf.cphf_grids.atom_grid = (99,590)
    #     # mf.cphf_grids.atom_grid = (10,14)
    #     mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

    #     test_energy = mf.kernel()
    #     assert mf.converged

    #     hobj = mf.Hessian()
    #     hobj.auxbasis_response = 2
    #     hobj.grid_response = True

    #     test_hessian = hobj.kernel()

    #     reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
    # #     reference_hessian = np.

    #     print(f"test hessian symmetric = {np.max(np.abs(test_hessian - test_hessian.transpose(1,0,3,2)))}")
    #     print(f"ref hessian symmetric = {np.max(np.abs(reference_hessian - reference_hessian.transpose(1,0,3,2)))}")
    #     print(f"test hessian translation invariance = {np.max(np.abs(np.sum(test_hessian, axis = 0)))}")
    #     print(f"ref hessian translation invariance = {np.max(np.abs(np.sum(reference_hessian, axis = 0)))}")
    #     print(f"error = {np.max(np.abs(test_hessian - reference_hessian))}")

    #     assert np.max(np.abs(test_hessian - reference_hessian)) < 1e-10

if __name__ == "__main__":
    print("Tests for KS hessian with grid response")
    unittest.main()
