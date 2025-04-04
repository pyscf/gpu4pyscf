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
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2, int1e_grids_ip2_charge_contracted

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
    def test_int1e_grids_ip_full_tensor_cart(self):
        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        ref_int1e_dA = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        ref_int1e_dC = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points)
        test_int1e_dA = test_int1e_dA.transpose(0, 2, 3, 1)
        test_int1e_dC = test_int1e_dC.transpose(0, 2, 3, 1)

        np.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        np.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_full_tensor_sph(self):
        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        ref_int1e_dA = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        ref_int1e_dC = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points)
        test_int1e_dA = test_int1e_dA.transpose(0, 2, 3, 1)
        test_int1e_dC = test_int1e_dC.transpose(0, 2, 3, 1)

        np.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        np.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_full_tensor_gaussian_charge(self):
        np.random.seed(12345)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        ref_int1e_dA = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        ref_int1e_dC = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, charge_exponents = charge_exponents)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, charge_exponents = charge_exponents)
        test_int1e_dA = test_int1e_dA.transpose(0, 2, 3, 1)
        test_int1e_dC = test_int1e_dC.transpose(0, 2, 3, 1)

        np.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        np.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_full_tensor_omega(self):
        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        ref_int1e_dA = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        ref_int1e_dC = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points)
        test_int1e_dA = test_int1e_dA.transpose(0, 2, 3, 1)
        test_int1e_dC = test_int1e_dC.transpose(0, 2, 3, 1)

        np.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        np.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_full_tensor_gaussian_charge_omega(self):
        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        np.random.seed(12345)
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        ref_int1e_dA = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        ref_int1e_dC = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, charge_exponents = charge_exponents)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, charge_exponents = charge_exponents)
        test_int1e_dA = test_int1e_dA.transpose(0, 2, 3, 1)
        test_int1e_dC = test_int1e_dC.transpose(0, 2, 3, 1)

        np.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        np.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_contracted_cart(self):
        np.random.seed(12346)
        dm = np.random.uniform(-2.0, 2.0, (mol_cart.nao, mol_cart.nao))
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_cart
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dA = np.einsum('xijk,ij,k->xi', v_nj, dm, charges)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dC = np.einsum('xijk,ij,k->xk', q_nj, dm, charges)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm, charges = charges)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, dm = dm, charges = charges)

        assert isinstance(test_int1e_dA, cp.ndarray)
        assert isinstance(test_int1e_dC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_contracted_sph(self):
        np.random.seed(12346)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dA = np.einsum('xijk,ij,k->xi', v_nj, dm, charges)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dC = np.einsum('xijk,ij,k->xk', q_nj, dm, charges)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm, charges = charges)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, dm = dm, charges = charges)

        assert isinstance(test_int1e_dA, cp.ndarray)
        assert isinstance(test_int1e_dC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_contracted_gaussian_charge(self):
        np.random.seed(12347)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dA = np.einsum('xijk,ij,k->xi', v_nj, dm, charges)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dC = np.einsum('xijk,ij,k->xk', q_nj, dm, charges)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm, charges = charges, charge_exponents = charge_exponents)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, dm = dm, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dA, cp.ndarray)
        assert isinstance(test_int1e_dC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_contracted_omega(self):
        np.random.seed(12348)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        omega = 1.2
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dA = np.einsum('xijk,ij,k->xi', v_nj, dm, charges)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dC = np.einsum('xijk,ij,k->xk', q_nj, dm, charges)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm, charges = charges)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, dm = dm, charges = charges)

        assert isinstance(test_int1e_dA, cp.ndarray)
        assert isinstance(test_int1e_dC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip_contracted_gaussian_charge_omega(self):
        np.random.seed(12349)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])
        charge_exponents = np.random.uniform(0.5, 1.0, grid_points.shape[0])

        omega = 0.8
        mol_sph_omega = mol_sph.copy()
        mol_sph_omega.set_range_coulomb(omega)

        mol = mol_sph_omega
        fakemol = gto.fakemol_for_charges(grid_points, expnt=charge_exponents)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        ref_int1e_dA = np.einsum('xijk,ij,k->xi', v_nj, dm, charges)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        ref_int1e_dC = np.einsum('xijk,ij,k->xk', q_nj, dm, charges)

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm, charges = charges, charge_exponents = charge_exponents)
        test_int1e_dC = int1e_grids_ip2(mol, grid_points, dm = dm, charges = charges, charge_exponents = charge_exponents)

        assert isinstance(test_int1e_dA, cp.ndarray)
        assert isinstance(test_int1e_dC, cp.ndarray)
        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)
        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip2_charge_contracted(self):
        np.random.seed(12346)
        charges = np.random.uniform(-2.0, 2.0, grid_points.shape[0])

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)

        ngrids = grid_points.shape[0]
        n_atom = mol.natm
        nao = mol.nao
        gridslice = [[ngrids * i // n_atom, ngrids * (i + 1) // n_atom] for i in range(n_atom)]
        ref_int1e_dC = np.zeros([n_atom, 3, nao, nao])
        for i_atom in range(n_atom):
            g0,g1 = gridslice[i_atom]
            ref_int1e_dC[i_atom, :, :, :] += np.einsum('dijq,q->dij', q_nj[:, :, :, g0:g1], charges[g0:g1])

        test_int1e_dC = cp.zeros([n_atom, 3, nao, nao])
        int1e_grids_ip2_charge_contracted(mol, grid_points, charges, gridslice, test_int1e_dC)

        cp.testing.assert_allclose(ref_int1e_dC, test_int1e_dC, atol = integral_threshold)

    def test_int1e_grids_ip1_density_contracted(self):
        np.random.seed(12347)
        dm = np.random.uniform(-2.0, 2.0, (mol_sph.nao, mol_sph.nao))

        mol = mol_sph
        fakemol = gto.fakemol_for_charges(grid_points)

        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)

        v_nj = np.einsum('dijq,ij->dqi', v_nj, dm)

        ngrids = grid_points.shape[0]
        aoslice = np.array(mol.aoslice_by_atom())
        ref_int1e_dA = np.empty([mol.natm, 3, ngrids])
        for i_atom in range(mol.natm):
            p0,p1 = aoslice[i_atom, 2:]
            ref_int1e_dA[i_atom,:,:] = np.einsum('dqi->dq', v_nj[:,:,p0:p1])

        test_int1e_dA = int1e_grids_ip1(mol, grid_points, dm = dm)

        cp.testing.assert_allclose(ref_int1e_dA, test_int1e_dA, atol = integral_threshold)

if __name__ == "__main__":
    print("Full Tests for One Electron Coulomb Integrals 1st Derivative")
    unittest.main()
