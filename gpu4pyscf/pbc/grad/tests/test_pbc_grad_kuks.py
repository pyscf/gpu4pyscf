#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf.pbc import gto, dft
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-3

def setUpModule():
    global cell, cell_no_pseudo, cell_be_no_pseudo, cell_be
    cell = gto.Cell()
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.basis = [[0, [1.3, 1]], [1, [0.8, 1]]]
    cell.verbose = 5
    cell.pseudo = 'gth-pade'
    cell.unit = 'bohr'
    cell.output = '/dev/null'
    cell.build()

    cell_no_pseudo = pyscf.M(
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]],
        a = '''
            0.000000000, 3.370137329, 3.370137329
            3.370137329, 0.000000000, 3.370137329
            3.370137329, 3.370137329, 0.000000000
        ''',
        unit = 'bohr',
        basis = [[0, [1.3, 1]], [0, [1.0, 1]], [1, [0.8, 1]]],
        # pseudo = 'gth-pade',
        verbose = 5,
        output = '/dev/null',
    )

    cell_be = pyscf.M(
        atom = [['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0]]],
        a = '''
            0.00, 3.37, 3.37
            3.37, 0.00, 3.37
            3.37, 3.37, 0.00
        ''',
        unit = 'bohr',
        basis = [[0, [3.3, 1]], [0, [0.9, 1]], [1, [0.8, 1]]],
        pseudo = 'gth-pade',
        verbose = 0,
    )

    cell_be_no_pseudo = pyscf.M(
        atom = [['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0]]],
        a = '''
            0.00, 3.37, 3.37
            3.37, 0.00, 3.37
            3.37, 3.37, 0.00
        ''',
        unit = 'bohr',
        basis = [[0, [3.3, 1]], [0, [0.9, 1]], [1, [0.8, 1]]],
        verbose = 0,
    )


def tearDownModule():
    global cell, cell_no_pseudo
    cell.stdout.close()
    cell_no_pseudo.stdout.close()
    del cell, cell_no_pseudo

def numerical_gradient(cell, xc, kpts):
    def get_energy(cell):
        mf = cell.KUKS(xc=xc, kpts=kpts)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        E = mf.kernel()
        assert mf.converged
        return E

    gradient = np.zeros([cell.natm, 3])
    cell_copy = cell.copy()
    for i_atom in range(cell.natm):
        for i_xyz in range(3):
            print(f"i_atom = {i_atom}, i_xyz = {i_xyz}")

            xyz_p = cell.atom_coords()
            xyz_p[i_atom, i_xyz] += disp
            cell_copy.set_geom_(xyz_p, unit='Bohr')
            cell_copy.build()
            Ep = get_energy(cell_copy)

            xyz_m = cell.atom_coords()
            xyz_m[i_atom, i_xyz] -= disp
            cell_copy.set_geom_(xyz_m, unit='Bohr')
            cell_copy.build()
            Em = get_energy(cell_copy)

            gradient[i_atom, i_xyz] = (Ep - Em) / (2 * disp)
    print(f"ref = np.{repr(gradient)}")
    return gradient


class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell, 'svwn', kpts)
        g_ref = np.array([[-0.05717802, -0.05717802,  0.05717802],
                          [ 0.05719081,  0.05719081, -0.05719081]])
        mf = cell.KUKS(xc='svwn', kpts=kpts).to_gpu()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_rsjk_lda_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell, 'svwn', kpts)
        g_ref = np.array([[-0.05717802, -0.05717802,  0.05717802],
                          [ 0.05719081,  0.05719081, -0.05719081]])
        mf = cell.KUKS(xc='svwn', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 5)

    def test_df_lda_grad(self):
        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc='svwn', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=3e-5)

        mf = cell_be.KUKS(xc='svwn', kpts=kpts).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g1 = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g-g1).max(), 0, 8)

    def test_lda_grad_multigrid_v2(self):
        kpts = cell.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell, 'svwn', kpts)
        g_ref = np.array([[-0.05717802, -0.05717802,  0.05717802],
                          [ 0.05719081,  0.05719081, -0.05719081]])
        mf = cell.KUKS(xc='svwn', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell, 'pbe', kpts)
        g_ref = np.array([[-0.05642876, -0.05642876,  0.05642876],
                          [ 0.05644314,  0.05644314, -0.05644314]])
        mf = cell.KUKS(xc='pbe', kpts=kpts).to_gpu()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_multigrid_v2(self):
        kpts = cell.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell, 'pbe', kpts)
        g_ref = np.array([[-0.05642876, -0.05642876,  0.05642876],
                          [ 0.05644314,  0.05644314, -0.05644314]])
        mf = cell.KUKS(xc='pbe', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell, 'r2scan', kpts)
        g_ref = np.array([[-0.05357594, -0.05357594,  0.05357594],
                          [ 0.05361277,  0.05361277, -0.05361277]])
        mf = cell.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - g_ref).max(), 0, delta=2e-6)

    def test_mgga_grad_multigrid_v2(self):
        kpts = cell.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell, 'r2scan', kpts)
        g_ref = np.array([[-0.05357594, -0.05357594,  0.05357594],
                          [ 0.05361277,  0.05361277, -0.05361277]])
        mf = cell.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf = mf.multigrid_numint()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - g_ref).max(), 0, delta=2e-6)

    def test_mgga_grad_without_pseudo(self):
        kpts = cell_no_pseudo.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell_no_pseudo, 'r2scan')
        g_ref = np.array([[ 0.05745937,  0.05745937, -0.05745937],
                          [-0.05767467, -0.05767467,  0.05767467]])
        mf = cell_no_pseudo.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad_multigrid_v2_without_pseudo(self):
        kpts = cell_no_pseudo.make_kpts([1,1,3])
        # g_ref = numerical_gradient(cell_no_pseudo, 'r2scan')
        g_ref = np.array([[ 0.05745937,  0.05745937, -0.05745937],
                          [-0.05767467, -0.05767467,  0.05767467]])
        mf = cell_no_pseudo.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf = mf.multigrid_numint()
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_hybrid_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell, 'pbe0', kpts)
        ref = np.array([[-0.05144472, -0.05144472,  0.05144472],
                        [ 0.05145642,  0.05145642, -0.05145642]])
        mf = cell.KUKS(kpts=kpts, xc='pbe0').to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_pbe0_grad(self):
        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc='pbe0', kpts=kpts).to_gpu()
        mf.exxdiv = None
        mf.rsjk = PBCJKMatrixOpt(cell_be)
        mf.j_engine = PBCJMatrixOpt(cell_be)
        mf = mf.multigrid_numint()
        mf.conv_tol = 1e-10
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_df_pbe0_grad(self):
        kpts = cell_be_no_pseudo.make_kpts([1,1,3])
        mf = cell_be_no_pseudo.KUKS(xc='pbe0', kpts=kpts).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=1e-5)

        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc='pbe0', kpts=kpts).to_gpu().density_fit()
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_sr_rsh_grad(self):
        xc = 'SR_HF(0.33)*.5 + 0.5*B88'
        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc=xc, kpts=kpts).to_gpu().density_fit()
        mf.rsjk = PBCJKMatrixOpt(cell_be)
        mf.j_engine = PBCJMatrixOpt(cell_be)
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=4e-6)

    def test_wb97_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell, 'wb97', kpts)
        ref = np.array([[-0.04358021, -0.04358021,  0.04358021],
                        [ 0.04392264,  0.04392264, -0.04392264]])
        mf = cell.KUKS(xc='wb97', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, delta=1e-6)

    def test_df_wb97_grad(self):
        kpts = cell_be_no_pseudo.make_kpts([1,1,3])
        mf = cell_be_no_pseudo.KUKS(xc='wb97', kpts=kpts).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=2e-6)

    def test_rsh_grad(self):
        xc = 'RSH(0.5, 0.8, -0.4)*.6 + 0.5*LDA'
        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc=xc, kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_be)
        mf.j_engine = PBCJMatrixOpt(cell_be)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_be)[1]

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=4e-6)

    def test_df_rsh_grad(self):
        xc = 'RSH(0.5, 0.8, -0.4)*.5 + 0.5*LDA'
        kpts = cell_be.make_kpts([1,1,3])
        mf = cell_be.KUKS(xc=xc, kpts=kpts).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=4e-6)

    def test_camb3lyp_grad(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell, 'camb3lyp', kpts)
        ref = np.array([[-0.04803599, -0.04803599,  0.04803599],
                        [ 0.04886935,  0.04886935, -0.04886935]])
        mf = cell.KUKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_camb3lyp_grad_without_pseudo(self):
        kpts = cell.make_kpts([1,1,3])
        # ref = numerical_gradient(cell_no_pseudo, xc='camb3lyp')
        ref = np.array([[ 0.05796211,  0.05796211, -0.05796211],
                        [-0.05796231, -0.05796231,  0.05796231]])
        mf = cell_no_pseudo.KUKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_no_pseudo)
        mf.j_engine = PBCJMatrixOpt(cell_no_pseudo)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, delta=5e-6)

class BeckeMGGAResponse_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu4pyscf.pbc.dft.gen_grid import BeckeGrids
        cls.cell = gto.M(
            atom='He .2 .3 .4; He 1.4 1.2 1.6', unit='Bohr',
            a=[[5., .3, .2], [.1, 5.5, .4], [.2, -.1, 6.]],
            basis=[[0, [1.3, 1.]], [1, [.8, 1.]]], spin=2,
            precision=1e-11, verbose=0)
        cls.kpts = cls.cell.make_kpts([2, 1, 1])
        cls.scaled_kpts = cls.cell.get_scaled_kpts(cls.kpts)
        densities = []
        for overlap in cls.cell.pbc_intor('int1e_ovlp', kpts=cls.kpts):
            eigenvalues, vectors = np.linalg.eigh(overlap)
            orbitals = vectors / np.sqrt(eigenvalues)
            # Three alpha and one beta electron, in orthonormal orbitals.
            densities.append([
                orbitals[:, :3] @ orbitals[:, :3].conj().T,
                orbitals[:, 3:4] @ orbitals[:, 3:4].conj().T])
        cls.dm = np.asarray(densities).transpose(1, 0, 2, 3).copy()
        cls.grids = cls._make_grids(cls.cell)
        assert isinstance(cls.grids, BeckeGrids)

    @staticmethod
    def _make_grids(cell):
        from gpu4pyscf.pbc.dft.gen_grid import BeckeGrids
        grids = BeckeGrids(cell)
        grids.atom_grid = (10, 38)
        grids.prune = None
        return grids.build()

    def _cpu_energy(self, cell, xc):
        from pyscf.pbc.dft.gen_grid import BeckeGrids as CpuGrids
        from pyscf.pbc.dft.numint import KNumInt as CpuNumInt
        grids = self._make_grids(cell)
        # The quadrature is deliberately identical, while AO, density and XC
        # evaluation use CPU implementations. Rebuilding includes moving centers
        # and partition weights, rather than just translating a frozen grid.
        cpu_grids = CpuGrids(cell)
        cpu_grids.coords = grids.coords.get()
        cpu_grids.weights = grids.weights.get()
        kpts = cell.get_abs_kpts(self.scaled_kpts)
        return CpuNumInt().nr_uks(cell, cpu_grids, xc, self.dm, kpts=kpts)[1]

    def _displaced_cell(self, row, axis, displacement):
        cell = self.cell.copy()
        coordinates = self.cell.atom_coords()
        if row < self.cell.natm:
            coordinates[row, axis] += displacement
        else:
            deformation = np.eye(3)
            deformation[row - self.cell.natm, axis] += displacement
            coordinates = coordinates @ deformation.T
            cell.a = self.cell.a @ deformation.T
        cell.set_geom_(coordinates, unit='Bohr')
        cell.build(False, False)
        return cell

    def test_r2scan_becke_gradient_and_strain(self):
        import cupy as cp
        from gpu4pyscf.pbc.dft.numint import KNumInt
        from gpu4pyscf.pbc.grad.kuks import get_vxc_full_response
        xc = 'r2SCAN'
        density = cp.asarray(self.dm)
        actual = get_vxc_full_response(
            KNumInt(), self.cell, self.grids, xc, density, self.kpts)
        self.assertEqual(actual.shape, (self.cell.natm + 3, 3))
        # All nuclear and strain components, including both off-diagonal
        # orientations. These are energy derivatives, not volume-scaled stress.
        step = 1e-4
        reference = np.empty_like(actual)
        for row, axis in np.ndindex(actual.shape):
            plus = self._displaced_cell(row, axis, step)
            minus = self._displaced_cell(row, axis, -step)
            reference[row, axis] = (
                self._cpu_energy(plus, xc) - self._cpu_energy(minus, xc)
            ) / (2 * step)
        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=0)
        np.testing.assert_allclose(actual[:self.cell.natm].sum(axis=0),
                                   0, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(density.get(), self.dm)

    def test_becke_mgga_spin_swap_and_restricted_limit(self):
        import cupy as cp
        from gpu4pyscf.pbc.dft.numint import KNumInt
        from gpu4pyscf.pbc.grad import kuks, krks
        xc = 'r2SCAN'
        polarized = kuks.get_vxc_full_response(
            KNumInt(), self.cell, self.grids, xc,
            cp.asarray(self.dm), self.kpts)
        swapped = kuks.get_vxc_full_response(
            KNumInt(), self.cell, self.grids, xc,
            cp.asarray(self.dm[::-1].copy()), self.kpts)
        np.testing.assert_allclose(polarized, swapped, atol=1e-11, rtol=0)
        total = self.dm.sum(axis=0)
        unpolarized = kuks.get_vxc_full_response(
            KNumInt(), self.cell, self.grids, xc,
            cp.asarray(np.stack([total / 2, total / 2])), self.kpts)
        restricted = krks.get_vxc_full_response(
            KNumInt(), self.cell, self.grids, xc,
            cp.asarray(total), self.kpts)
        np.testing.assert_allclose(unpolarized, restricted,
                                   atol=1e-10, rtol=0)


if __name__ == "__main__":
    print("Full Tests for KUKS Gradients")
    unittest.main()
