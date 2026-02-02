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
from pyscf.pbc import gto
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-4

def setUpModule():
    global cell, cell_no_pseudo, kpts
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

    cell_no_pseudo = gto.Cell(
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
    cell_no_pseudo.build()

    kpts = cell.make_kpts([1,1,3])

def tearDownModule():
    global cell, cell_no_pseudo
    cell.stdout.close()
    del cell
    cell_no_pseudo.stdout.close()
    del cell_no_pseudo

def numerical_gradient(cell, xc, kpts):
    def get_energy(cell):
        mf = cell.KRKS(xc=xc, kpts=kpts)
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
        # g_ref = numerical_gradient(cell, 'svwn', kpts)
        g_ref = np.array([[-0.05717807, -0.05717807,  0.05717807],
                          [ 0.05719087,  0.05719087, -0.05719087]])
        mf = cell.KRKS(xc='svwn', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_rsjk_lda_grad(self):
        # ref = numerical_gradient(cell, 'svwn', kpts)
        ref = np.array([[-0.05717807, -0.05717807,  0.05717807],
                        [ 0.05719087,  0.05719087, -0.05719087]])
        mf = cell.KRKS(kpts=kpts, xc='svwn').to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_lda_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'svwn', kpts)
        g_ref = np.array([[-0.05717807, -0.05717807,  0.05717807],
                          [ 0.05719087,  0.05719087, -0.05719087]])
        mf = cell.KRKS(xc='svwn', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad(self):
        # g_ref = numerical_gradient(cell, 'pbe', kpts)
        g_ref = np.array([[-0.05642881, -0.05642881,  0.05642881],
                          [ 0.05644319,  0.05644319, -0.05644319]])
        mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'pbe', kpts)
        g_ref = np.array([[-0.05642881, -0.05642881,  0.05642881],
                          [ 0.05644319,  0.05644319, -0.05644319]])
        mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_without_pseudo(self):
        # g_ref = numerical_gradient(cell_no_pseudo, 'pbe', kpts)
        g_ref = np.array([[ 0.05625033,  0.05625033, -0.05625033],
                          [-0.0562508 , -0.0562508 ,  0.0562508 ]])
        mf = cell_no_pseudo.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_multigrid_v2_without_pseudo(self):
        # g_ref = numerical_gradient(cell_no_pseudo, 'pbe', kpts)
        g_ref = np.array([[ 0.05625033,  0.05625033, -0.05625033],
                          [-0.0562508 , -0.0562508 ,  0.0562508 ]])
        mf = cell_no_pseudo.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad(self):
        # g_ref = numerical_gradient(cell, 'r2scan', kpts)
        g_ref = np.array([[-0.05357598, -0.05357598,  0.05357598],
                          [ 0.05361285,  0.05361285, -0.05361285]])
        mf = cell.KRKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'r2scan', kpts)
        g_ref = np.array([[-0.05357598, -0.05357598,  0.05357598],
                          [ 0.05361285,  0.05361285, -0.05361285]])
        mf = cell.KRKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_hybrid_grad(self):
        # ref = numerical_gradient(cell, 'pbe0', kpts)
        ref = np.array([[-0.05144472, -0.05144472,  0.05144472],
                        [ 0.05145642,  0.05145642, -0.05145642]])
        mf = cell.KRKS(kpts=kpts, xc='pbe0').to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skip('Insufficient GPU memory for rsjk.q_cond')
    def test_hse_grad(self):
        # ref = numerical_gradient(cell, 'hse06', kpts)
        ref = np.array([[-0.05104506, -0.05104506,  0.05104506],
                        [ 0.05201861,  0.05201861, -0.05201861]])
        mf = cell.RKS(xc='hse06', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_wb97_grad(self):
        # ref = numerical_gradient(cell, 'wb97', kpts)
        ref = np.array([[-0.04358029, -0.04358029,  0.04358029],
                        [ 0.04392258,  0.04392258, -0.04392258]])
        mf = cell.KRKS(xc='wb97', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_wb97_grad_without_pseudo(self):
        # ref = numerical_gradient(cell_no_pseudo, 'wb97', kpts)
        ref = np.array([[ 0.0625855 ,  0.0625855 , -0.0625855 ],
                        [-0.06288699, -0.06288699,  0.06288699]])
        mf = cell_no_pseudo.KRKS(xc='wb97', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell_no_pseudo)
        mf.rsjk = PBCJKMatrixOpt(cell_no_pseudo)
        mf.j_engine = PBCJMatrixOpt(cell_no_pseudo)
        mf.conv_tol = 1e-10
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_camb3lyp_grad(self):
        # ref = numerical_gradient(cell, 'camb3lyp', kpts)
        ref = np.array([[-0.04803599, -0.04803599,  0.04803599],
                        [ 0.04886935,  0.04886935, -0.04886935]])
        mf = cell.KRKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for KRKS Gradients")
    unittest.main()
