# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib, gto, df
from gpu4pyscf.gto.int3c1e_ipip import int1e_grids_ipip1, int1e_grids_ipvip1, int1e_grids_ip1ip2, int1e_grids_ipip2

def setUpModule():
    global mol_sph, mol_cart, grid_points, integral_threshold, density_contraction_threshold, charge_contraction_threshold
    atom = '''
O	0.0000	0.7375	-0.0528
O	0.0000	-0.7375	-0.1528
H	0.8190	0.8170	0.4220
H	-0.8190	-0.8170	0.4220
'''
    bas='def2-qzvpp'

    mol_sph = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.verbose = 0
    mol_sph.build()

    mol_cart = pyscf.M(atom=atom, basis=bas, max_memory=32000, cart=True)
    mol_cart.output = '/dev/null'
    mol_cart.verbose = 0
    mol_cart.build()

    xs = np.arange(-2.01, 2.0, 0.5)
    ys = np.arange(-2.02, 2.0, 0.5)
    zs = np.arange(-2.03, 2.0, 0.5)
    grid_points = lib.cartesian_prod([xs, ys, zs])

    grid_points = np.vstack([grid_points, mol_sph.atom_coords()])

    # All of the following thresholds bound the max value of the corresponding matrix / tensor.
    integral_threshold = 1e-12
    density_contraction_threshold = 1e-10
    charge_contraction_threshold = 1e-12

def tearDownModule():
    global mol_sph, mol_cart, grid_points
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart, grid_points

class KnownValues(unittest.TestCase):
    '''
    Values are compared to PySCF CPU intor() function
    '''
    def test_int1e_grids_ipip1_charge_contracted_cart(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdA = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdA = ref_int1e_dAdA.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdA = int1e_grids_ipip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdA, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdA, test_int1e_dAdA, atol = integral_threshold)

    def test_int1e_grids_ipip1_charge_contracted_sph(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdA = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdA = ref_int1e_dAdA.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdA = int1e_grids_ipip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdA, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdA, test_int1e_dAdA, atol = integral_threshold)

    def test_int1e_grids_ipip1_charge_contracted_gaussian_charge(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdA = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdA = ref_int1e_dAdA.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdA = int1e_grids_ipip1(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdA, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdA, test_int1e_dAdA, atol = integral_threshold)

    def test_int1e_grids_ipip1_charge_contracted_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdA = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdA = ref_int1e_dAdA.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdA = int1e_grids_ipip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdA, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdA, test_int1e_dAdA, atol = integral_threshold)

    def test_int1e_grids_ipip1_charge_contracted_gaussian_charge_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdA = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdA = ref_int1e_dAdA.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdA = int1e_grids_ipip1(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdA, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdA, test_int1e_dAdA, atol = integral_threshold)

    # ^ ipip1    v ipvip1

    def test_int1e_grids_ipvip1_charge_contracted_cart(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdB = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdB = ref_int1e_dAdB.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdB = int1e_grids_ipvip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdB, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdB, test_int1e_dAdB, atol = integral_threshold)

    def test_int1e_grids_ipvip1_charge_contracted_sph(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdB = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdB = ref_int1e_dAdB.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdB = int1e_grids_ipvip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdB, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdB, test_int1e_dAdB, atol = integral_threshold)

    def test_int1e_grids_ipvip1_charge_contracted_gaussian_charge(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdB = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdB = ref_int1e_dAdB.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdB = int1e_grids_ipvip1(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdB, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdB, test_int1e_dAdB, atol = integral_threshold)

    def test_int1e_grids_ipvip1_charge_contracted_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdB = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdB = ref_int1e_dAdB.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdB = int1e_grids_ipvip1(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdB, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdB, test_int1e_dAdB, atol = integral_threshold)

    def test_int1e_grids_ipvip1_charge_contracted_gaussian_charge_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdB = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdB = ref_int1e_dAdB.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdB = int1e_grids_ipvip1(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdB, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdB, test_int1e_dAdB, atol = integral_threshold)

    # ^ ipvip1    v ip1ip2

    def test_int1e_grids_ip1ip2_charge_contracted_cart(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdC = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdC = ref_int1e_dAdC.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdC = int1e_grids_ip1ip2(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdC, test_int1e_dAdC, atol = integral_threshold)

    def test_int1e_grids_ip1ip2_charge_contracted_sph(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdC = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdC = ref_int1e_dAdC.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdC = int1e_grids_ip1ip2(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdC, test_int1e_dAdC, atol = integral_threshold)

    def test_int1e_grids_ip1ip2_charge_contracted_gaussian_charge(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdC = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdC = ref_int1e_dAdC.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdC = int1e_grids_ip1ip2(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdC, test_int1e_dAdC, atol = integral_threshold)

    def test_int1e_grids_ip1ip2_charge_contracted_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdC = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdC = ref_int1e_dAdC.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdC = int1e_grids_ip1ip2(mol, grid_points, charges = charges)

        assert isinstance(test_int1e_dAdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdC, test_int1e_dAdC, atol = integral_threshold)

    def test_int1e_grids_ip1ip2_charge_contracted_gaussian_charge_omega(self):
        np.random.seed(12345)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dAdC = np.einsum('dijq,q->dij', v_nj, charges)
        ref_int1e_dAdC = ref_int1e_dAdC.reshape(3, 3, mol.nao, mol.nao)

        test_int1e_dAdC = int1e_grids_ip1ip2(mol, grid_points, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dAdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dAdC, test_int1e_dAdC, atol = integral_threshold)

    # ^ ip1ip2    v ipip2

    def test_int1e_grids_ipip2_charge_contracted_cart(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_cart.nao, mol_cart.nao))

        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        # Note: we cannot compute ipip2 (dCdC) directly due to numerical problems,
        #       pyscf treat a point charge as a sharp Gaussian, and we cannot take 2nd derivative of it.
        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        v_nj = -v_nj - v_nj.transpose(0, 2, 1, 3) # dCdC = -dAdC - dBdC
        ref_int1e_dCdC = np.einsum('dijq,ij->dq', v_nj, dm)
        ref_int1e_dCdC = ref_int1e_dCdC.reshape(3, 3, grid_points.shape[0])

        test_int1e_dCdC = int1e_grids_ipip2(mol, grid_points, dm = dm)

        assert isinstance(test_int1e_dCdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dCdC, test_int1e_dCdC, atol = integral_threshold)

    def test_int1e_grids_ipip2_charge_contracted_sph(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        # Note: we cannot compute ipip2 (dCdC) directly due to numerical problems,
        #       pyscf treat a point charge as a sharp Gaussian, and we cannot take 2nd derivative of it.
        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        v_nj = -v_nj - v_nj.transpose(0, 2, 1, 3) # dCdC = -dAdC - dBdC
        ref_int1e_dCdC = np.einsum('dijq,ij->dq', v_nj, dm)
        ref_int1e_dCdC = ref_int1e_dCdC.reshape(3, 3, grid_points.shape[0])

        test_int1e_dCdC = int1e_grids_ipip2(mol, grid_points, dm = dm)

        assert isinstance(test_int1e_dCdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dCdC, test_int1e_dCdC, atol = integral_threshold)

    def test_int1e_grids_ipip2_charge_contracted_gaussian_charge(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipip2 = mol._add_suffix('int3c2e_ipip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dCdC = np.einsum('dijq,ij->dq', v_nj, dm)
        ref_int1e_dCdC = ref_int1e_dCdC.reshape(3, 3, grid_points.shape[0])

        test_int1e_dCdC = int1e_grids_ipip2(mol, grid_points, dm = dm, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dCdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dCdC, test_int1e_dCdC, atol = integral_threshold)

    def test_int1e_grids_ipip2_charge_contracted_omega(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        # Note: we cannot compute ipip2 (dCdC) directly due to numerical problems,
        #       pyscf treat a point charge as a sharp Gaussian, and we cannot take 2nd derivative of it.
        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        v_nj = -v_nj - v_nj.transpose(0, 2, 1, 3) # dCdC = -dAdC - dBdC
        ref_int1e_dCdC = np.einsum('dijq,ij->dq', v_nj, dm)
        ref_int1e_dCdC = ref_int1e_dCdC.reshape(3, 3, grid_points.shape[0])

        test_int1e_dCdC = int1e_grids_ipip2(mol, grid_points, dm = dm)

        assert isinstance(test_int1e_dCdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dCdC, test_int1e_dCdC, atol = integral_threshold)

    def test_int1e_grids_ipip2_charge_contracted_gaussian_charge_omega(self):
        np.random.seed(12345)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ipip2 = mol._add_suffix('int3c2e_ipip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dCdC = np.einsum('dijq,ij->dq', v_nj, dm)
        ref_int1e_dCdC = ref_int1e_dCdC.reshape(3, 3, grid_points.shape[0])

        test_int1e_dCdC = int1e_grids_ipip2(mol, grid_points, dm = dm, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dCdC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dCdC, test_int1e_dCdC, atol = integral_threshold)

if __name__ == "__main__":
    print("Full Tests for One Electron Coulomb Integrals 2nd Derivative")
    unittest.main()
