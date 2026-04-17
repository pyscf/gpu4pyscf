# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

try:
    from gpu4pyscf.lib.cuest_wrapper import apply_cuest_wrapper
except ModuleNotFoundError:
    apply_cuest_wrapper = None

import pyscf
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.scf.uhf import UHF
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.dft.gen_grid import stratmann
from gpu4pyscf.grad.rks import get_exc_full_response, get_nlc_exc_full_response
from gpu4pyscf.grad.uks import get_exc_full_response as get_exc_full_response_uks
from gpu4pyscf.solvent import pcm
from gpu4pyscf.lib.cupy_helper import tag_array

import pytest
import unittest
import numpy as np
import cupy as cp

@pytest.mark.special
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert apply_cuest_wrapper is not None

        mol_sph = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  0.4220
            """,
            basis = {"O" : "def2-tzvp", "H" : "3-21g"},
            charge = 0,
            spin = 0,
            verbose = 0,
            # cart = True,
            # output = '/dev/null',
        )
        cls.mol_sph = mol_sph

        mol_cart = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  0.4220
            """,
            basis = {"O" : "def2-tzvp", "H" : "3-21g"},
            charge = 0,
            spin = 0,
            verbose = 0,
            cart = True,
            # output = '/dev/null',
        )
        cls.mol_cart = mol_cart

        mol_unrestricted = pyscf.M(
            atom = """
                O  0.0000  0.7375 -0.0528
                O  0.0000 -0.7375 -0.1528
                H  0.8190  0.8170  0.4220
                H -0.8190 -0.8170  0.4220
            """,
            basis = {"O" : "def2-tzvp", "H" : "3-21g"},
            charge = 1,
            spin = 1,
            verbose = 0,
            # cart = True,
            # output = '/dev/null',
        )
        cls.mol_unrestricted = mol_unrestricted

        mol_ecp = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                H      0.000000    0.000000    1.090000
                I      2.018000    0.000000   -0.713333
                I     -1.009000    1.747340   -0.713333
                I     -1.009000   -1.747340   -0.713333
            """,
            basis = "def2-tzvp",
            ecp = "def2-tzvp",
            verbose = 0,
            # output = '/dev/null',
        )
        cls.mol_ecp = mol_ecp

        mol_ecp_cart = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                H      0.000000    0.000000    1.090000
                I      2.018000    0.000000   -0.713333
                I     -1.009000    1.747340   -0.713333
                I     -1.009000   -1.747340   -0.713333
            """,
            basis = "def2-tzvp",
            ecp = "def2-tzvp",
            verbose = 0,
            cart = True,
            # output = '/dev/null',
        )
        cls.mol_ecp_cart = mol_ecp_cart

        mol_ecp_unrestricted = pyscf.M(
            atom = """
                C      0.000000    0.000000    0.000000
                I      2.018000    0.000000   -0.713333
                I     -1.009000    1.747340   -0.713333
                I     -1.009000   -1.747340   -0.713333
            """,
            basis = "def2-tzvp",
            ecp = "def2-tzvp",
            charge = 0,
            spin = 1,
            verbose = 0,
            # output = '/dev/null',
        )
        cls.mol_ecp_unrestricted = mol_ecp_unrestricted

        cls.auxbasis = "def2-universal-jkfit"

    @classmethod
    def tearDownClass(cls):
    #     cls.mol_sph.stdout.close()
    #     cls.mol_cart.stdout.close()
    #     cls.mol_unrestricted.stdout.close()
        pass

    def test_overlap_spherical(self):
        mf = RHF(self.mol_sph)
        ref_S = mf.get_ovlp()

        mf = apply_cuest_wrapper(mf)
        test_S = mf.get_ovlp()

        assert cp.max(cp.abs(test_S - ref_S)) < 1e-14

    def test_overlap_cartesian(self):
        mf = RHF(self.mol_cart)
        ref_S = mf.get_ovlp()

        mf = apply_cuest_wrapper(mf)
        test_S = mf.get_ovlp()

        assert cp.max(cp.abs(test_S - ref_S)) < 1e-14

    def test_hcore_spherical(self):
        mf = RHF(self.mol_sph)
        ref_Hcore = mf.get_hcore()

        mf = apply_cuest_wrapper(mf)
        test_Hcore = mf.get_hcore()

        assert cp.max(cp.abs(test_Hcore - ref_Hcore)) < 1e-11

    def test_hcore_cartesian(self):
        mf = RHF(self.mol_cart)
        ref_Hcore = mf.get_hcore()

        mf = apply_cuest_wrapper(mf)
        test_Hcore = mf.get_hcore()

        assert cp.max(cp.abs(test_Hcore - ref_Hcore)) < 1e-11

    def test_J_spherical(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_J = mf.get_j(dm = dm, hermi = False)

        mf = RHF(mol)
        mf = apply_cuest_wrapper(mf)
        mf.auxbasis = auxbasis
        test_J = mf.get_j(dm = dm)

        assert cp.max(cp.abs(test_J - ref_J)) < 1e-9

    def test_J_cartesian(self):
        mol = self.mol_cart
        auxbasis = self.auxbasis

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_J = mf.get_j(dm = dm, hermi = False)

        mf = RHF(mol)
        mf = apply_cuest_wrapper(mf)
        mf.auxbasis = auxbasis
        test_J = mf.get_j(dm = dm)

        assert cp.max(cp.abs(test_J - ref_J)) < 1e-1 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well

    def test_K_spherical_mo_input(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 2e-9

    def test_K_cartesian_mo_input(self):
        mol = self.mol_cart
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e0 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well

    def test_K_spherical_dm_input(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 2e-9

    def test_K_cartesian_dm_input(self):
        mol = self.mol_cart
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e0 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well

    def test_J_spherical_multiple_dms(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        dm = cp.random.rand(3, mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_J = mf.get_j(dm = dm, hermi = False)

        mf = RHF(mol)
        mf = apply_cuest_wrapper(mf)
        mf.auxbasis = auxbasis
        test_J = mf.get_j(dm = dm)

        assert cp.max(cp.abs(test_J - ref_J)) < 1e-9

    def test_K_spherical_multiple_mos(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e-9

    def test_K_spherical_multiple_dms(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e-9

    def test_K_spherical_mo_input_low_precision(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e-8

    def test_K_spherical_multiple_mos_low_precision(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e-9

    def test_K_spherical_mo_input_really_low_precision(self):
        mol = self.mol_sph
        auxbasis = self.auxbasis

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        ref_K = mf.get_k(dm = dm)

        mf = RHF(mol).density_fit(auxbasis = auxbasis)
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        mf.additional_precision_control_parameters["CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT"] = 5
        mf.additional_precision_control_parameters["CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT"] = 5
        test_K = mf.get_k(dm = dm)

        assert cp.max(cp.abs(test_K - ref_K)) < 1e-3
        assert cp.max(cp.abs(test_K - ref_K)) > 1e-5, \
            "It is very likely the low-precision emulation functionality is not turned on, because of an old cuEST version, an old compilation (on CUDA 12), or an old GPU"

    def test_density_fitting_eigenvalue_threshold(self):
        mol = pyscf.M(
            atom = """
                H  1 0 0
                H  0 0 0.1
            """,
            basis = "def2-svp",
            charge = 0,
            spin = 0,
            verbose = 0,
        )

        my_auxbasis = """
            H S
                1.0 1.0
            H S
                1.001 1.0
            H    S
                22.0683430000           0.0530339860
                4.3905712000           0.3946522022
                1.0540787000           0.9172987712
            H    S
                0.2717874000           1.0000000
            H    P
                1.8529979000           1.0000000
            H    P
                0.3881034000           1.0000000
        """

        mf = RHF(mol).density_fit(auxbasis = my_auxbasis)
        mf.with_df.build()

        auxmol = mf.with_df.auxmol
        j2c = auxmol.intor('int2c2e', hermi=1)
        e_j2c, v_j2c = np.linalg.eigh(j2c)

        # The two small eigenvalues are roughly 5e-8
        j2c_threshold = 1e-6

        j2c_1_all_eigenvalue = v_j2c @ np.diag(e_j2c**-1) @ v_j2c.T
        v_j2c = v_j2c[:, e_j2c >= j2c_threshold]
        e_j2c = e_j2c[e_j2c >= j2c_threshold]
        j2c_1_big_eigenvalue = v_j2c @ np.diag(e_j2c**-1) @ v_j2c.T

        from pyscf.df.incore import aux_e2
        j3c = aux_e2(mol, auxmol)

        dm = np.random.rand(mol.nao, mol.nao) * 2 - 1
        dm = dm @ dm.T # symmetric positive definite

        ref_j_all_eigenvalue = np.einsum("pqa,ab,brs,rs->pq", j3c, j2c_1_all_eigenvalue, j3c.transpose(2,0,1), dm)
        ref_j_big_eigenvalue = np.einsum("pqa,ab,brs,rs->pq", j3c, j2c_1_big_eigenvalue, j3c.transpose(2,0,1), dm)
        ref_k_all_eigenvalue = np.einsum("pra,ab,bqs,rs->pq", j3c, j2c_1_all_eigenvalue, j3c.transpose(2,0,1), dm)
        ref_k_big_eigenvalue = np.einsum("pra,ab,bqs,rs->pq", j3c, j2c_1_big_eigenvalue, j3c.transpose(2,0,1), dm)

        mf = RHF(mol).density_fit(auxbasis = my_auxbasis)
        mf = apply_cuest_wrapper(mf)
        assert mf.density_fitting_cutoff == 1e-12

        test_j_all_eigenvalue, test_k_all_eigenvalue = mf.get_jk(dm = dm)
        test_j_all_eigenvalue = test_j_all_eigenvalue.get()
        test_k_all_eigenvalue = test_k_all_eigenvalue.get()

        assert np.max(np.abs(test_j_all_eigenvalue - ref_j_all_eigenvalue)) < 1e-5
        assert np.max(np.abs(test_k_all_eigenvalue - ref_k_all_eigenvalue)) < 1e-5

        mf.density_fitting_cutoff = j2c_threshold

        test_j_big_eigenvalue, test_k_big_eigenvalue = mf.get_jk(dm = dm)
        test_j_big_eigenvalue = test_j_big_eigenvalue.get()
        test_k_big_eigenvalue = test_k_big_eigenvalue.get()

        assert np.max(np.abs(test_j_big_eigenvalue - ref_j_big_eigenvalue)) < 1e-10
        assert np.max(np.abs(test_k_big_eigenvalue - ref_k_big_eigenvalue)) < 1e-10

    def test_rks_vxc_pure_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "r2scan")
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        mf = RKS(mol, xc = "r2scan")
        mf.grids.atom_grid = (99,590)
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_rks_vxc_hybrid_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b3lyp5")
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        mf = RKS(mol, xc = "b3lyp5")
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_rks_vxc_hybrid_cartesian_mo(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b3lyp")
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        mf = RKS(mol, xc = "b3lyp")
        mf.grids.atom_grid = (99,590)
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_rks_vxc_pure_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97")
        mf.grids.level = 1
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        mf = RKS(mol, xc = "b97")
        mf.grids.level = 1
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_uks_vxc_pure_spherical_mos(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "pbe")
        mf.grids.atom_grid = (3,6)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        mf = UKS(mol, xc = "pbe")
        mf.grids.atom_grid = (3,6)
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_uks_vxc_hybrid_spherical_mos(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "pbe0")
        mf.grids.atom_grid = (99,590)
        mf.grids.prune = None
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        mf = UKS(mol, xc = "pbe0")
        mf.grids.atom_grid = (99,590)
        mf.grids.prune = None
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 2e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_uks_vxc_pure_cartesian_mos(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "m06l")
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        mf = UKS(mol, xc = "m06l")
        mf.grids.atom_grid = (99,590)
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_uks_vxc_pure_spherical_dms(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "pbe")
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        mf = UKS(mol, xc = "pbe")
        mf.grids.atom_grid = (99,590)
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_uks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_vv10_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (20,86)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (20,86)
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-8
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-9

    def test_vv10_cartesian_mo(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-8
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-9

    def test_vv10_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-9

    def test_vv10_cartesian_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 3
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 3
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-9

    def test_vv10_spherical_multiple_mos(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-8

    def test_vv10_spherical_multiple_dms(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf = apply_cuest_wrapper(mf)
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-8

    def test_vv10_spherical_mo_low_precision(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (20,86)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (20,86)
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # # The precision is not significantly lower with low precision
        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-8
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-8

    def test_vv10_cartesian_mo_low_precision(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.atom_grid = (50,194)
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-8
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-9

    def test_vv10_spherical_multiple_mos_low_precision(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-8

    def test_vv10_cartesian_multiple_mos_low_precision(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = cp.vstack([mo_occ, mo_occ])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        ref_n, ref_enlc, ref_vnlc = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])

        mf = RKS(mol, xc = "b97mv")
        mf.nlcgrids.level = 0
        mf = apply_cuest_wrapper(mf)
        mf.math_mode = "default"
        test_n, test_enlc, test_vnlc = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_enlc - ref_enlc) < 1e-7
        assert cp.max(cp.abs(test_vnlc - ref_vnlc)) < 1e-8

    def test_customized_grid_setup_default(self):
        mol = pyscf.M(
            atom = """
                C      0.000000     0.000000     0.000000
                H      0.000000     0.000000     1.090000
                F      1.282000     0.000000    -0.453000
                F     -0.641000     1.110000    -0.453000
                Cl    -0.834000    -1.445000    -0.590000
            """,
            basis = "def2-TZVP",
            verbose = 0,
        )

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "pbe")
        mf.grids.atom_grid = {'default': (6, 6), 'F': (66, 266)}
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        ref_n, ref_exc, ref_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        mf = RKS(mol, xc = "pbe")
        mf.grids.atom_grid = {'default': (6, 6), 'F': (66, 266)}
        mf = apply_cuest_wrapper(mf)
        test_n, test_exc, test_vxc = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)

        # assert np.abs(test_n - ref_n) < 1e-16
        assert np.abs(test_exc - ref_exc) < 1e-9
        assert cp.max(cp.abs(test_vxc - ref_vxc)) < 1e-9

    def test_pcm_spherical_cpcm(self):
        mol = self.mol_sph

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "C-PCM"
        solvent.surface_discretization_method = "ISWIG"

        ref_epcm, ref_vpcm = solvent.kernel(dm)
        ref_q = solvent._intermediates['q']

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_epcm, test_vpcm = mf.with_solvent.kernel(dm)
        test_q = mf.with_solvent._intermediates['q']

        assert np.abs(test_epcm - ref_epcm) < 2e-8
        assert cp.max(cp.abs(test_vpcm - ref_vpcm)) < 1e-8

        # The removal of nonzero element is because pyscf and cuest stores different number of pcm grids, cuest keeps all zero-weight grids, while pyscf removes them.
        # The sorting is because pyscf and cuest has different order of pcm grids.
        # The negative sign is because cuest is using a different convention of pcm charge. Only charge is affected, energy and Fock matrix is unaffected.
        nonzero_charge_threshold = 1e-8
        test_q = test_q[cp.abs(test_q) > nonzero_charge_threshold]
        test_q = cp.sort(test_q)
        ref_q = ref_q[cp.abs(ref_q) > nonzero_charge_threshold]
        ref_q = cp.sort(-ref_q)
        assert cp.max(cp.abs(test_q - ref_q)) < 1e-9

    def test_pcm_cartesian_cpcm(self):
        mol = self.mol_cart

        dm = np.random.rand(mol.nao, mol.nao) * 2 - 1

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "C-PCM"
        solvent.surface_discretization_method = "ISWIG"

        ref_epcm, ref_vpcm = solvent.kernel(dm)
        ref_q = solvent._intermediates['q']

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_epcm, test_vpcm = mf.with_solvent.kernel(dm)
        test_q = mf.with_solvent._intermediates['q']

        assert np.abs(test_epcm - ref_epcm) < 1e-7
        assert cp.max(cp.abs(test_vpcm - ref_vpcm)) < 1e-7

        # The removal of nonzero element is because pyscf and cuest stores different number of pcm grids, cuest keeps all zero-weight grids, while pyscf removes them.
        # The sorting is because pyscf and cuest has different order of pcm grids.
        # The negative sign is because cuest is using a different convention of pcm charge. Only charge is affected, energy and Fock matrix is unaffected.
        nonzero_charge_threshold = 1e-8
        test_q = test_q[cp.abs(test_q) > nonzero_charge_threshold]
        test_q = cp.sort(test_q)
        ref_q = ref_q[cp.abs(ref_q) > nonzero_charge_threshold]
        ref_q = cp.sort(-ref_q)
        assert cp.max(cp.abs(test_q - ref_q)) < 1e-8

    def test_pcm_spherical_cosmo(self):
        mol = self.mol_sph

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "COSMO"
        solvent.surface_discretization_method = "ISWIG"

        ref_epcm, ref_vpcm = solvent.kernel(dm)
        ref_q = solvent._intermediates['q']

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "COSMO"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_epcm, test_vpcm = mf.with_solvent.kernel(dm)
        test_q = mf.with_solvent._intermediates['q']

        assert np.abs(test_epcm - ref_epcm) < 2e-8
        assert cp.max(cp.abs(test_vpcm - ref_vpcm)) < 1e-8

        # The removal of nonzero element is because pyscf and cuest stores different number of pcm grids, cuest keeps all zero-weight grids, while pyscf removes them.
        # The sorting is because pyscf and cuest has different order of pcm grids.
        # The negative sign is because cuest is using a different convention of pcm charge. Only charge is affected, energy and Fock matrix is unaffected.
        nonzero_charge_threshold = 1e-8
        test_q = test_q[cp.abs(test_q) > nonzero_charge_threshold]
        test_q = cp.sort(test_q)
        ref_q = ref_q[cp.abs(ref_q) > nonzero_charge_threshold]
        ref_q = cp.sort(-ref_q)
        assert cp.max(cp.abs(test_q - ref_q)) < 1e-9

    def test_pcm_spherical_cpcm_multiple_dms(self):
        mol = self.mol_sph

        dm = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "COSMO"
        solvent.surface_discretization_method = "ISWIG"

        ref_epcm, ref_vpcm = solvent.kernel(dm)
        ref_q = solvent._intermediates['q']

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "COSMO"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_epcm, test_vpcm = mf.with_solvent.kernel(dm)
        test_q = mf.with_solvent._intermediates['q']

        assert np.abs(test_epcm - ref_epcm) < 2e-8
        assert cp.max(cp.abs(test_vpcm - ref_vpcm)) < 1e-8

        # The removal of nonzero element is because pyscf and cuest stores different number of pcm grids, cuest keeps all zero-weight grids, while pyscf removes them.
        # The sorting is because pyscf and cuest has different order of pcm grids.
        # The negative sign is because cuest is using a different convention of pcm charge. Only charge is affected, energy and Fock matrix is unaffected.
        nonzero_charge_threshold = 1e-8
        test_q = test_q[cp.abs(test_q) > nonzero_charge_threshold]
        test_q = cp.sort(test_q)
        ref_q = ref_q[cp.abs(ref_q) > nonzero_charge_threshold]
        ref_q = cp.sort(-ref_q)
        assert cp.max(cp.abs(test_q - ref_q)) < 2e-9

    def test_hcore_ecp_spherical(self):
        mf = RHF(self.mol_ecp)
        ref_Hcore = mf.get_hcore()

        mf = apply_cuest_wrapper(mf)
        test_Hcore = mf.get_hcore()

        assert cp.max(cp.abs(test_Hcore - ref_Hcore)) < 1e-5

    def test_hcore_ecp_cartesian(self):
        mf = RHF(self.mol_ecp_cart)
        ref_Hcore = mf.get_hcore()

        mf = apply_cuest_wrapper(mf)
        test_Hcore = mf.get_hcore()

        assert cp.max(cp.abs(test_Hcore - ref_Hcore)) < 1e-5

    ### Gradient tests from here on

    def test_overlap_derivative_spherical(self):
        mol = self.mol_sph

        dme0 = np.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol)
        gobj = mf.Gradients()

        s1 = gobj.get_ovlp(mol)
        aoslices = mol.aoslice_by_atom()
        ref_ds = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_ds[ia] += np.einsum('xij,ij->x', s1[:,p0:p1], (dme0 + dme0.T)[p0:p1])

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_ds = gobj.get_ovlp_grad(dme0)

        assert np.max(np.abs(test_ds - ref_ds)) < 1e-11

    def test_overlap_derivative_cartesian(self):
        mol = self.mol_cart

        dme0 = np.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol)
        gobj = mf.Gradients()

        s1 = gobj.get_ovlp(mol)
        aoslices = mol.aoslice_by_atom()
        ref_ds = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_ds[ia] += np.einsum('xij,ij->x', s1[:,p0:p1], (dme0 + dme0.T)[p0:p1])

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_ds = gobj.get_ovlp_grad(dme0)

        assert np.max(np.abs(test_ds - ref_ds)) < 1e-11

    def test_hcore_derivative_spherical(self):
        mol = self.mol_sph

        dm0 = np.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol).to_cpu() # Only CPU object has hcore_generator()
        gobj = mf.Gradients()

        hcore_deriv = gobj.hcore_generator(mol)
        aoslices = mol.aoslice_by_atom()
        ref_dhcore = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            h1ao = hcore_deriv(ia)
            ref_dhcore[ia] += np.einsum('xij,ij->x', h1ao, dm0)

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dhcore = gobj.get_hcore_grad(dm0)

        assert np.max(np.abs(test_dhcore - ref_dhcore)) < 1e-10

    def test_hcore_derivative_cartesian(self):
        mol = self.mol_cart

        dm0 = np.random.rand(mol.nao, mol.nao) * 2 - 1

        mf = RHF(mol).to_cpu() # Only CPU object has hcore_generator()
        gobj = mf.Gradients()

        hcore_deriv = gobj.hcore_generator(mol)
        aoslices = mol.aoslice_by_atom()
        ref_dhcore = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            h1ao = hcore_deriv(ia)
            ref_dhcore[ia] += np.einsum('xij,ij->x', h1ao, dm0)

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dhcore = gobj.get_hcore_grad(dm0)

        assert np.max(np.abs(test_dhcore - ref_dhcore)) < 1e-10

    def test_rhf_jk_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = np.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = np.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = self.auxbasis)
        mf = mf.to_cpu() # The GPU gradient is much worse in precision
        gobj = mf.Gradients()

        vhf = gobj.get_veff(mol, dm)
        aoslices = mol.aoslice_by_atom()
        ref_dvhf = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_dvhf[ia] += np.einsum('xij,ij->x', vhf[:,p0:p1], (dm + dm.T)[p0:p1])
        for ia in range(mol.natm):
            ref_dvhf[ia] += gobj.extra_force(ia, {"vhf" : vhf})

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-8

    def test_rhf_jk_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = np.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = np.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = self.auxbasis)
        mf = mf.to_cpu() # The GPU gradient is much worse in precision
        gobj = mf.Gradients()

        vhf = gobj.get_veff(mol, dm)
        aoslices = mol.aoslice_by_atom()
        ref_dvhf = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_dvhf[ia] += np.einsum('xij,ij->x', vhf[:,p0:p1], (dm + dm.T)[p0:p1])
        for ia in range(mol.natm):
            ref_dvhf[ia] += gobj.extra_force(ia, {"vhf" : vhf})

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-8

    def test_rhf_jk_derivative_cartesian_mo(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = np.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = np.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RHF(mol).density_fit(auxbasis = self.auxbasis)
        mf = mf.to_cpu() # The GPU gradient is much worse in precision
        gobj = mf.Gradients()

        vhf = gobj.get_veff(mol, dm)
        aoslices = mol.aoslice_by_atom()
        ref_dvhf = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_dvhf[ia] += np.einsum('xij,ij->x', vhf[:,p0:p1], (dm + dm.T)[p0:p1])
        for ia in range(mol.natm):
            ref_dvhf[ia] += gobj.extra_force(ia, {"vhf" : vhf})

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e1 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well

    def test_uhf_jk_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = np.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occ = np.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occ = np.vstack([mo_occ, mo_occ])
        dm = np.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ np.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UHF(mol).density_fit(auxbasis = self.auxbasis)
        mf = mf.to_cpu() # The GPU gradient is much worse in precision
        gobj = mf.Gradients()

        vhf = gobj.get_veff(mol, dm)
        aoslices = mol.aoslice_by_atom()
        ref_dvhf = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            ref_dvhf[ia] += np.einsum('sxij,sij->x', vhf[:,:,p0:p1], (dm + dm.transpose(0,2,1))[:,p0:p1])
        for ia in range(mol.natm):
            ref_dvhf[ia] += gobj.extra_force(ia, {"vhf" : vhf})

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-8

    def test_rks_pure_xc_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "svwn")
        mf.grids.atom_grid = (20, 86)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()

        exc, vxc = get_exc_full_response(mf._numint, mol, mf.grids, mf.xc, dm)
        ref_dvxc = exc + vxc * 2

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvxc = gobj.get_xc_grad(dm)

        assert np.max(np.abs(test_dvxc - ref_dvxc)) < 1e-10

    def test_rks_pure_xc_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "pbe")
        mf.grids.level = 3
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()

        exc, vxc = get_exc_full_response(mf._numint, mol, mf.grids, mf.xc, dm)
        ref_dvxc = exc + vxc * 2

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvxc = gobj.get_xc_grad(dm)

        assert np.max(np.abs(test_dvxc - ref_dvxc)) < 1e-10

    def test_rks_pure_xc_derivative_cartesian_mo(self):
        mol = self.mol_cart

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "r2scan")
        mf.grids.level = 3
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()

        exc, vxc = get_exc_full_response(mf._numint, mol, mf.grids, mf.xc, dm)
        ref_dvxc = exc + vxc * 2

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvxc = gobj.get_xc_grad(dm)

        assert np.max(np.abs(test_dvxc - ref_dvxc)) < 1e-10

    def test_uks_pure_xc_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "m06l")
        mf.grids.atom_grid = (50, 194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()

        exc, vxc = get_exc_full_response_uks(mf._numint, mol, mf.grids, mf.xc, dm)
        ref_dvxc = exc + vxc * 2

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvxc = gobj.get_xc_grad(dm)

        assert np.max(np.abs(test_dvxc - ref_dvxc)) < 1e-10

    def test_uks_hybrid_xc_derivative_spherical_dm(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "pbe0")
        mf.grids.atom_grid = (6,6)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()

        exc, vxc = get_exc_full_response_uks(mf._numint, mol, mf.grids, mf.xc, dm)
        ref_dvxc = exc + vxc * 2

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvxc = gobj.get_xc_grad(dm)

        assert np.max(np.abs(test_dvxc - ref_dvxc)) < 1e-10

    def test_rks_pure_veff_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "svwn").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.atom_grid = (20, 86)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_rks_pure_veff_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "BLYP").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.atom_grid = (20, 86)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_rks_hybrid_veff_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "B3LYP").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 0
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_rks_hybrid_veff_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "B3LYP5").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_uks_pure_veff_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "PBE").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_uks_pure_veff_derivative_spherical_dm(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "B97").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_uks_hybrid_veff_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "PBE0").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_uks_hybrid_veff_derivative_spherical_dm(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "PBE0").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        from pyscf.dft.gen_grid import sg1_prune
        mf.grids.prune = sg1_prune
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_rks_vv10_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "B97MV")
        mf.nlcgrids.level = 1
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()

        enlc, vnlc = get_nlc_exc_full_response(mf._numint, mol, mf.nlcgrids, mf.xc, dm)
        ref_dvnlc = enlc + 2 * vnlc

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvnlc = gobj.get_nlc_grad(dm)

        assert np.max(np.abs(test_dvnlc - ref_dvnlc)) < 1e-8

    def test_rks_vv10_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "B97MV")
        mf.nlcgrids.atom_grid = (50,194)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()

        enlc, vnlc = get_nlc_exc_full_response(mf._numint, mol, mf.nlcgrids, mf.xc, dm)
        ref_dvnlc = enlc + 2 * vnlc

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvnlc = gobj.get_nlc_grad(dm)

        assert np.max(np.abs(test_dvnlc - ref_dvnlc)) < 1e-8

    def test_uks_vv10_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "B97MV")
        mf.nlcgrids.level = 2
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()

        enlc, vnlc = get_nlc_exc_full_response(mf._numint, mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])
        ref_dvnlc = enlc + 2 * vnlc

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvnlc = gobj.get_nlc_grad(dm)

        assert np.max(np.abs(test_dvnlc - ref_dvnlc)) < 1e-8

    def test_uks_vv10_derivative_spherical_dm(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "B97MV")
        mf.nlcgrids.level = 2
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()

        enlc, vnlc = get_nlc_exc_full_response(mf._numint, mol, mf.nlcgrids, mf.xc, dm[0] + dm[1])
        ref_dvnlc = enlc + 2 * vnlc

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvnlc = gobj.get_nlc_grad(dm)

        assert np.max(np.abs(test_dvnlc - ref_dvnlc)) < 1e-8

    def test_rks_vv10_veff_derivative_spherical_mo(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        mf.nlcgrids.level = 1
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_rks_vv10_veff_derivative_spherical_dm(self):
        mol = self.mol_sph

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        mo_occ = cp.array([2.0] * nocc + [0.0] * (mol.nao - nocc))
        dm = mo_coeff @ cp.diag(mo_occ) @ mo_coeff.T
        # dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = RKS(mol, xc = "b97mv").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        mf.nlcgrids.atom_grid = (50,6)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_uks_vv10_veff_derivative_spherical_mo(self):
        mol = self.mol_unrestricted

        nocc = mol.nelectron // 2
        mo_coeff = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        mo_occa = cp.array([1.0] * nocc + [0.0] * (mol.nao - nocc))
        mo_occb = cp.array([1.0] * (nocc + 1) + [0.0] * (mol.nao - (nocc + 1)))
        mo_occ = cp.vstack([mo_occa, mo_occb])
        dm = cp.empty([2, mol.nao, mol.nao])
        for i_dm in range(2):
            dm[i_dm] = mo_coeff[i_dm] @ cp.diag(mo_occ[i_dm]) @ mo_coeff[i_dm].T
        dm = tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)

        mf = UKS(mol, xc = "b97mv").density_fit(auxbasis = self.auxbasis)
        mf.with_df.build()
        mf.grids.level = 2
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.grids.build()
        mf.nlcgrids.level = 1
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.nlcgrids.build()
        gobj = mf.Gradients()
        gobj.grid_response = True

        ref_dvhf = gobj.energy_ee(mol, dm)

        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dvhf = gobj.energy_ee(mf.mol, dm)

        assert np.max(np.abs(test_dvhf - ref_dvhf)) < 1e-7 * cp.max(dm)

    def test_pcm_derivative_spherical_cpcm(self):
        mol = self.mol_sph

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        dm = 0.5 * (dm + dm.T)

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "C-PCM"
        solvent.surface_discretization_method = "ISWIG"

        ref_dpcm = solvent.grad(dm)

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_dpcm = mf.with_solvent.grad(dm)

        assert np.max(np.abs(test_dpcm - ref_dpcm)) < 1e-6

    def test_pcm_derivative_cartesian_cpcm(self):
        mol = self.mol_cart

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        dm = 0.5 * (dm + dm.T)

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 19
        solvent.method = "C-PCM"
        solvent.surface_discretization_method = "ISWIG"

        ref_dpcm = solvent.grad(dm)

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 19
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_dpcm = mf.with_solvent.grad(dm)

        assert np.max(np.abs(test_dpcm - ref_dpcm)) < 1e-6

    def test_pcm_derivative_spherical_cosmo(self):
        mol = self.mol_sph

        dm = cp.random.rand(mol.nao, mol.nao) * 2 - 1
        dm = 0.5 * (dm + dm.T)

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 3
        solvent.method = "COSMO"
        solvent.surface_discretization_method = "ISWIG"

        ref_dpcm = solvent.grad(dm)

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 3
        mf.with_solvent.method = "COSMO"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_dpcm = mf.with_solvent.grad(dm)

        assert np.max(np.abs(test_dpcm - ref_dpcm)) < 1e-10

    def test_pcm_derivative_spherical_cpcm_multiple_dms(self):
        mol = self.mol_sph

        dm = cp.random.rand(2, mol.nao, mol.nao) * 2 - 1
        dm = 0.5 * (dm + dm.transpose(0,2,1))

        solvent = pcm.PCM(mol)
        solvent.lebedev_order = 7
        solvent.method = "COSMO"
        solvent.surface_discretization_method = "ISWIG"

        ref_dpcm = solvent.grad(dm[0] + dm[1])

        mf = RHF(mol).PCM()
        mf.with_solvent.lebedev_order = 7
        mf.with_solvent.method = "COSMO"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf = apply_cuest_wrapper(mf)

        test_dpcm = mf.with_solvent.grad(dm)

        assert np.max(np.abs(test_dpcm - ref_dpcm)) < 1e-6

    def test_hcore_ecp_derivative_spherical(self):
        mol = self.mol_ecp

        dm0 = np.random.rand(mol.nao, mol.nao) * 2 - 1
        dm0 = 0.5 * (dm0 + dm0.T)

        mf = RHF(mol).to_cpu() # Only CPU object has hcore_generator()
        gobj = mf.Gradients()

        hcore_deriv = gobj.hcore_generator(mol)
        aoslices = mol.aoslice_by_atom()
        ref_dhcore = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            h1ao = hcore_deriv(ia)
            ref_dhcore[ia] += np.einsum('xij,ij->x', h1ao, dm0)

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dhcore = gobj.get_hcore_grad(dm0)

        assert np.max(np.abs(test_dhcore - ref_dhcore)) < 5e-7

    def test_hcore_ecp_derivative_cartesian(self):
        mol = self.mol_ecp_cart

        dm0 = np.random.rand(mol.nao, mol.nao) * 2 - 1
        dm0 = 0.5 * (dm0 + dm0.T)

        mf = RHF(mol).to_cpu() # Only CPU object has hcore_generator()
        gobj = mf.Gradients()

        hcore_deriv = gobj.hcore_generator(mol)
        aoslices = mol.aoslice_by_atom()
        ref_dhcore = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            p0,p1 = aoslices[ia,2:]
            h1ao = hcore_deriv(ia)
            ref_dhcore[ia] += np.einsum('xij,ij->x', h1ao, dm0)

        mf = mf.to_gpu()
        mf = apply_cuest_wrapper(mf)
        gobj = mf.Gradients()

        test_dhcore = gobj.get_hcore_grad(dm0)

        assert np.max(np.abs(test_dhcore - ref_dhcore)) < 5e-7

    ### Integrated test from here on

    def test_rhf_spherical(self):
        mf = RHF(self.mol_sph).density_fit(auxbasis = self.auxbasis)
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -150.82052183502287

        # gobj = mf.Gradients()
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [-0.0149744162106065,  0.0498695235261568,  0.0068685739935875],
            [ 0.0569458014119186, -0.0543752420641752, -0.038062721613191 ],
            [-0.0011214334428944, -0.0103130596278806, -0.0000455414030975],
            [-0.0408499517583927,  0.0148187781659531,  0.0312396890226945],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rhf_cartesian(self):
        mf = RHF(self.mol_cart).density_fit(auxbasis = self.auxbasis)
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -150.82093840108257

        # gobj = mf.Gradients()
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [-0.0158913111441652,  0.049980821891408 ,  0.0063341527430343],
            [ 0.0569981622311735, -0.0545985898663108, -0.0381514513748948],
            [-0.0001516224712388, -0.0101571971696853,  0.0005134402216798],
            [-0.0409552286157271,  0.0147749651445737,  0.0313038584102701],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-5 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-4

    def test_rks_nonhybrid_spherical(self):
        mf = RKS(self.mol_sph, xc = "PBE").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.4475350629541

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0181525597147723,  0.0031046403476438,  0.0201796318349582],
            [ 0.0245197866267168, -0.0068131157745288, -0.0139068632190096],
            [-0.0301507824210185, -0.0097864847052515, -0.0170383515023689],
            [-0.0125215639204717,  0.0134949601321619,  0.0107655828864319],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rks_nonhybrid_cartesian(self):
        mf = RKS(self.mol_sph, xc = "svwn").density_fit(auxbasis = self.auxbasis)
        mf.grids.level = 4
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -150.52006539203825

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0200217108819669,  0.0195136692487825,  0.0213649348477775],
            [ 0.0230039470785544, -0.0227090627828073, -0.0145068873761209],
            [-0.0310376325396065, -0.0092497641190501, -0.017301408588787 ],
            [-0.0119880254209135,  0.0124451576530361,  0.0104433611171331],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-4 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-5

    def test_rks_hybrid_spherical(self):
        mf = RKS(self.mol_sph, xc = "PBE0").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.43683962954555

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0076317038119678,  0.0227477056970482,  0.015674374403195 ],
            [ 0.0348096492653209, -0.0265950576969054, -0.0223779662444672],
            [-0.0200406278952294, -0.0091357394059052, -0.0110765542144287],
            [-0.0224007251820288,  0.0129830914057488,  0.0177801460556934],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rks_hybrid_cartesian(self):
        mf = RKS(self.mol_cart, xc = "B3LYP").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.60147748301569

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0086484503971171,  0.0098745149206163,  0.0158576458875265],
            [ 0.0325555023975974, -0.0139107920070884, -0.0195378279299909],
            [-0.0218743487589048, -0.0099979622287406, -0.012119886509929 ],
            [-0.0193296040358182,  0.0140342393152783,  0.0158000685523969],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-4 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-4

    # def test_rks_range_separated_spherical(self):
    #     ### TODO
    #     raise NotImplementedError()

    # def test_rks_range_separated_cartesian(self):
    #     ### TODO
    #     raise NotImplementedError()

    def test_uhf_spherical(self):
        mf = UHF(self.mol_unrestricted).density_fit(auxbasis = self.auxbasis)
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -150.402311053481

        # gobj = mf.Gradients()
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0273623910671228,  0.1513481718289924,  0.0065653976568676],
            [ 0.0143217647513498, -0.1515325454320564, -0.0428078201729321],
            [-0.0446857379759398, -0.0178217769005201,  0.0051641016743174],
            [ 0.0030015821574838,  0.0180061505035582,  0.0310783208417513],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-6

    def test_uks_nonhybrid_spherical(self):
        mf = UKS(self.mol_unrestricted, xc = "M06L").density_fit(auxbasis = self.auxbasis)
        mf.grids.level = 1
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.154726483159

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0405823359386277,  0.0773931234977194,  0.0089597385457566],
            [ 0.0004014857775136, -0.0770314942224122, -0.0274377098448664],
            [-0.0549361529756331, -0.0162297858802353, -0.0025703508723347],
            [ 0.0139523312595036,  0.0158681566049312,  0.0210483221714484],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-6

    def test_uks_hybrid_spherical(self):
        mf = UKS(self.mol_unrestricted, xc = "PBE0").density_fit(auxbasis = self.auxbasis)
        mf.grids.level = 5
        from pyscf.dft.gen_grid import sg1_prune
        mf.grids.prune = sg1_prune
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.012073447433

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0451111365433339,  0.095882785974192 ,  0.0132005393781485],
            [-0.0024347443112924, -0.0953372817921192, -0.0266137644938298],
            [-0.0589595499613855, -0.0163993328629585, -0.0055380956296445],
            [ 0.0162831577293603,  0.0158538286809381,  0.0189513207453216],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    # def test_uks_range_separated_spherical(self):
    #     ### TODO
    #     raise NotImplementedError()

    # def test_uks_range_separated_cartesian(self):
    #     ### TODO
    #     raise NotImplementedError()

    def test_rks_with_dispersion(self):
        mf = RKS(self.mol_sph, xc = "PBE0").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.disp = "d4:pbe0"
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        # ref_dispersion_energy = mf.get_dispersion()
        ref_energy = -151.43738028079676
        ref_dispersion_energy = -0.000540651238426269

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0076447901312165,  0.022757396670689 ,  0.015685570301247 ],
            [ 0.0347965805843842, -0.0266053675199013, -0.0223668112098732],
            [-0.0200540877344161, -0.0091370679134555, -0.0110869363812426],
            [-0.0223872829823043,  0.0129850387625535,  0.0177681772897176],
        ])

        mf = apply_cuest_wrapper(mf)
        assert mf.do_disp()
        test_energy = mf.kernel()
        assert mf.converged
        test_dispersion_energy = mf.get_dispersion()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert abs(test_dispersion_energy - ref_dispersion_energy) <= 1e-12
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rks_vv10_spherical(self):
        mf = RKS(self.mol_sph, xc = "B97MV").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (20,86)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.548045318808

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0066332529534632,  0.0100430045925215,  0.0142735615897495],
            [ 0.0353064660267146, -0.0143557436962514, -0.0221719297119023],
            [-0.0186836552773175, -0.0084534279489525, -0.0103866636976877],
            [-0.0232560637028483,  0.0127661670527255,  0.0182850318198577],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-7
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rks_vv10_cartesian(self):
        mf = RKS(self.mol_cart, xc = "B97MV").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (99,590)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.level = 3
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.549369443725

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0050152303418058,  0.0110963854126069,  0.0134631167258829],
            [ 0.0360056518369418, -0.0155083176329374, -0.0227022610176562],
            [-0.0171802460115469, -0.008425267024078 , -0.0094714012045511],
            [-0.023840636167197 ,  0.0128371992443888,  0.0187105454963485],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-4 # Cuest uses spherical auxbasis, pyscf uses cartesian auxbasis, they don't match well
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-4

    def test_uks_vv10_spherical(self):
        mf = UKS(self.mol_unrestricted, xc = "B97MV").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (20,86)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.125040895988

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0438253867458369,  0.078535445138959 ,  0.0109191517348799],
            [-0.0017712533563494, -0.0783672569375859, -0.0263241031857531],
            [-0.0571377930856762, -0.0155710655702049, -0.0043885461871895],
            [ 0.0150836596961996,  0.0154028773688586,  0.0197934976380647],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-7
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rhf_pcm_spherical(self):
        mf = RHF(self.mol_sph).density_fit(auxbasis = self.auxbasis)
        mf = mf.PCM()
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -150.832912134331

        # gobj = mf.Gradients()
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [-0.01440432,  0.04994651,  0.0097793 ],
            [ 0.05648449, -0.05459643, -0.03529997],
            [-0.00324735, -0.01180676, -0.00287872],
            [-0.03883283,  0.01645667,  0.02839939],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rks_pcm_spherical(self):
        mf = RKS(self.mol_sph, xc = "B3LYP").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf = mf.PCM()
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.61181099096

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.00912088,  0.01023625,  0.01853435],
            [ 0.03300452, -0.01430458, -0.01760946],
            [-0.02390403, -0.01149941, -0.01471115],
            [-0.01822137,  0.01556774,  0.01378625],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_uks_pcm_spherical(self):
        mf = UKS(self.mol_unrestricted, xc = "M06L").density_fit(auxbasis = self.auxbasis)
        mf.grids.level = 1
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        mf = mf.PCM()
        mf.with_solvent.eps = 10
        mf.with_solvent.lebedev_order = 3
        mf.with_solvent.method = "COSMO"
        mf.with_solvent.surface_discretization_method = "ISWIG"
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.264505317404

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.04366383,  0.08099353,  0.01273346],
            [-0.00284979, -0.08092047, -0.02451411],
            [-0.05656637, -0.01514429, -0.00588853],
            [ 0.01575234,  0.01507123,  0.01766918],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_rhf_ecp_spherical(self):
        mf = RHF(self.mol_ecp).density_fit(auxbasis = self.auxbasis)
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -928.411537297789

        # gobj = mf.Gradients()
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [-4.02181659e-05, -4.31103368e-14,  5.12514757e-03],
            [-6.20080713e-06,  4.55527296e-16,  1.35310249e-02],
            [-2.99121187e-03, -3.53840332e-14, -6.23189602e-03],
            [ 1.51881542e-03, -2.66299573e-03, -6.21213821e-03],
            [ 1.51881542e-03,  2.66299573e-03, -6.21213821e-03],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-5
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-5

    def test_rks_ecp_cartesian(self):
        mf = RKS(self.mol_ecp_cart, xc = "B3LYP").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -932.111319110805

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 7.56308911e-04, -3.73526435e-07,  4.63577676e-03],
            [-3.12493883e-05,  6.45296434e-08,  6.19196277e-03],
            [-1.02988999e-02,  4.33261795e-08, -3.42718436e-03],
            [ 4.78686605e-03, -8.40056055e-03, -3.70030385e-03],
            [ 4.78697326e-03,  8.40082125e-03, -3.70025216e-03],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-4
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-4

    def test_uks_ecp_spherical(self):
        mf = UKS(self.mol_ecp_unrestricted, xc = "B97MV").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (20,86)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (20,50)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -931.0600399723028

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 1.26700485e-03,  9.73479553e-15,  3.88124852e-02],
            [ 7.14969242e-03,  1.06719878e-14, -1.28305758e-02],
            [-4.20834863e-03,  5.75562391e-03, -1.29909547e-02],
            [-4.20834863e-03, -5.75562391e-03, -1.29909547e-02],
        ])

        mf = apply_cuest_wrapper(mf)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-5
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-5

    def test_rks_geometry_optimiation(self):
        mol = pyscf.M(
            atom = '''
                H      1.161000    0.066100    1.023800
                C      0.657900   -0.004500    0.063900
                H      1.335200   -0.083000   -0.781500
                C     -0.657900    0.004500   -0.063900
                H     -1.299706    0.492883    0.733789
                H     -1.160800   -0.066100   -1.023900
            ''', # Distorted ethene
            basis='def2-svp',
            verbose = 4,
        )

        mf = RKS(mol, xc = "B3LYP").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.conv_tol = 1e-12

        mf = apply_cuest_wrapper(mf)

        from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(mf, maxsteps = 20)

        # In order to get this reference result, you need to turn on grid_response in g_scanner in geometric_solver
        ref_geometry = np.array([
            [ 2.18510795,  0.28904983,  1.94854842],
            [ 1.26010701,  0.03112532,  0.11627166],
            [ 2.49294593, -0.39558379, -1.48919977],
            [-1.23754392,  0.22706461, -0.14643388],
            [-2.47038551,  0.65369928,  1.45907265],
            [-2.16277957, -0.03078865, -1.97860878],
        ])

        test_geometry = mol_eq.atom_coords()

        assert np.max(np.abs(test_geometry - ref_geometry)) < 1e-5

    def test_rks_geometry_optimiation_pcm(self):
        mol = pyscf.M(
            atom = '''
                H      1.200100    0.036300    0.843100
                C      0.703100    0.008300   -0.130500
                H      0.987700    0.894300   -0.711400
                H      1.015500   -0.891800   -0.674200
                O     -0.658200   -0.006700    0.173000
                H     -1.156984   -0.740964   -0.955624
            ''', # Distorted ethene
            basis='6-31g',
            verbose = 4,
        )

        mf = RKS(mol, xc = "B97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (30,110)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12

        mf = mf.PCM()
        mf.with_solvent.lebedev_order = 11
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"

        mf = apply_cuest_wrapper(mf)

        from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(mf, maxsteps = 20)

        # In order to get this reference result, you need to turn on grid_response in g_scanner in geometric_solver
        ref_geometry = np.array([
            [ 2.18549005,  0.69076327,  1.37773576],
            [ 1.3253293 ,  0.01624309, -0.36117271],
            [ 1.39605385,  1.52356225, -1.77297937],
            [ 2.39992446, -1.60657166, -1.05506676],
            [-1.25394697, -0.6634382 ,  0.28564292],
            [-2.10102519, -1.28443285, -1.22489057],
        ])

        test_geometry = mol_eq.atom_coords()

        assert np.max(np.abs(test_geometry - ref_geometry)) < 1e-5

    def test_fast_pyscf_cuest_spherical_reorder(self):
        mol = pyscf.M(
            atom = "He 0 0 0; He 0 0 10; Ne 0 0 100",
            basis = {"He" : """
                He S
                1.0 1.0
                He P
                1.0 1.0
                He D
                1.0 1.0
                He F
                1.0 1.0
                He G
                1.0 1.0
            """, "Ne" : """
                Ne S
                1.0 1.0
                Ne S
                10.0 1.0
                Ne D
                1.0 1.0
                Ne G
                1.0 1.0
            """},
            verbose = 0,
        )

        import cupy as cp
        from gpu4pyscf.lib.cuest_wrapper import _cuest_pyscf_reorder_spherical_slow, _cuest_pyscf_reorder_spherical

        ref_M = cp.random.rand(mol.nao, 10, 10)
        test_M = ref_M.copy()
        _cuest_pyscf_reorder_spherical_slow(mol, ref_M, axis = [0], cuest_to_pyscf = True)
        _cuest_pyscf_reorder_spherical(mol, test_M, axis = [0], cuest_to_pyscf = True)
        assert cp.max(cp.abs(test_M - ref_M)) == 0.0

        ref_M = cp.random.rand(10, mol.nao, 10)
        test_M = ref_M.copy()
        _cuest_pyscf_reorder_spherical_slow(mol, ref_M, axis = [1], cuest_to_pyscf = True)
        _cuest_pyscf_reorder_spherical(mol, test_M, axis = [1], cuest_to_pyscf = True)
        assert cp.max(cp.abs(test_M - ref_M)) == 0.0

        ref_M = cp.random.rand(10, 10, mol.nao)
        test_M = ref_M.copy()
        _cuest_pyscf_reorder_spherical_slow(mol, ref_M, axis = [2], cuest_to_pyscf = False)
        _cuest_pyscf_reorder_spherical(mol, test_M, axis = [2], cuest_to_pyscf = False)
        assert cp.max(cp.abs(test_M - ref_M)) == 0.0

        ref_M = cp.random.rand(mol.nao, mol.nao)
        test_M = ref_M.copy()
        _cuest_pyscf_reorder_spherical_slow(mol, ref_M, cuest_to_pyscf = False)
        _cuest_pyscf_reorder_spherical(mol, test_M, cuest_to_pyscf = False)
        assert cp.max(cp.abs(test_M - ref_M)) == 0.0

        ref_M = cp.random.rand(2, mol.nao, mol.nao)
        test_M = ref_M.copy()
        _cuest_pyscf_reorder_spherical_slow(mol, ref_M, axis = [1,2], cuest_to_pyscf = False)
        _cuest_pyscf_reorder_spherical(mol, test_M, axis = [1,2], cuest_to_pyscf = False)
        assert cp.max(cp.abs(test_M - ref_M)) == 0.0

    def test_turn_off_everything_cuest_rks(self):
        mf = RKS(self.mol_sph, xc = "B97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (30,110)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12

        mf = mf.PCM()
        mf.with_solvent.lebedev_order = 11
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"

        # ref_energy = mf.kernel()
        ref_energy = -151.55876540852802

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0061999326655547,  0.0103113207464326,  0.0161820460746163],
            [ 0.0359435221113695, -0.0146739202537985, -0.0205672949252587],
            [-0.0197138150630107, -0.0097682512894919, -0.0122285636555485],
            [-0.0224296397149849,  0.0141308507967007,  0.0166138125060378],
        ])

        mf = apply_cuest_wrapper(mf)
        mf.turn_on_cuest_hcore = False
        mf.turn_on_cuest_jk = False
        mf.turn_on_cuest_xc = False
        mf.turn_on_cuest_nlc = False
        mf.turn_on_cuest_pcm = False

        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_turn_off_everything_cuest_uks(self):
        mf = UKS(self.mol_unrestricted, xc = "B97MV").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (50,194)
        mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = None
        mf.nlcgrids.atom_grid = (30,110)
        mf.nlcgrids.becke_scheme = stratmann
        mf.nlcgrids.radii_adjust = None
        mf.conv_tol = 1e-12

        mf = mf.PCM()
        mf.with_solvent.lebedev_order = 11
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.surface_discretization_method = "ISWIG"

        # ref_energy = mf.kernel()
        ref_energy = -151.2552951119203

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.0436628435897044,  0.0822305284665879,  0.0135398542832707],
            [-0.0014862073884895, -0.0825295878691523, -0.0248987836605871],
            [-0.0562402640818798, -0.014300558402444 , -0.0063823270650011],
            [ 0.0140636278795982,  0.0145996178048596,  0.0177412564423006],
        ])

        mf = apply_cuest_wrapper(mf)
        mf.turn_on_cuest_hcore = False
        mf.turn_on_cuest_jk = False
        mf.turn_on_cuest_xc = False
        mf.turn_on_cuest_nlc = False
        mf.turn_on_cuest_pcm = False

        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

    def test_turn_off_xc_rks(self):
        mf = RKS(self.mol_sph, xc = "PBE").density_fit(auxbasis = self.auxbasis)
        mf.grids.atom_grid = (50,194)
        from gpu4pyscf.dft.gen_grid import original_becke
        from gpu4pyscf.dft.radi import treutler_atomic_radii_adjust
        mf.grids.becke_scheme = original_becke
        mf.grids.radii_adjust = treutler_atomic_radii_adjust
        mf.conv_tol = 1e-12
        # ref_energy = mf.kernel()
        ref_energy = -151.44754580781756

        # gobj = mf.Gradients()
        # gobj.grid_response = True
        # ref_gradient = gobj.kernel()
        ref_gradient = np.array([
            [ 0.018167960483545 ,  0.003148471368462 ,  0.0201228515860936],
            [ 0.0245139382570376, -0.0068424979524799, -0.0138759432810263],
            [-0.0301722785930933, -0.009822446230231 , -0.0169911275329757],
            [-0.0125096201486006,  0.0135164728140924,  0.0107442192276996],
        ])

        mf = apply_cuest_wrapper(mf)
        mf.turn_on_cuest_xc = False
        mf.grids._backup_grids.becke_scheme = original_becke
        mf.grids._backup_grids.radii_adjust = treutler_atomic_radii_adjust

        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-7

if __name__ == '__main__':
    print("Full Tests for CuEST Wrapper")
    unittest.main()
