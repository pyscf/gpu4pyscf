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
from pyscf.pbc import gto, dft
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
    cell.verbose = 4
    cell.pseudo = 'gth-pade'
    cell.unit = 'bohr'
    cell.mesh = [15] * 3
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

def numerical_gradient(cell, xc):
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
        # g_ref = numerical_gradient(cell, 'lda,vwn')
        g_ref = np.array([[-0.0570564 , -0.0570564 ,  0.0570564 ],
                          [ 0.05723334,  0.05723334, -0.05723334]])
        mf = dft.KUKS(cell, kpts).to_gpu()
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_lda_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'lda,vwn')
        g_ref = np.array([[-0.0570564 , -0.0570564 ,  0.0570564 ],
                          [ 0.05723334,  0.05723334, -0.05723334]])
        mf = dft.KUKS(cell, kpts).to_gpu()
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad(self):
        # g_ref = numerical_gradient(cell, 'pbe,pbe')
        g_ref = np.array([[-0.05592252, -0.05592252,  0.05592252],
                          [ 0.05671207,  0.05671207, -0.05671207]])
        mf = dft.KUKS(cell, kpts).to_gpu()
        mf.xc = 'pbe,pbe'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'pbe,pbe')
        g_ref = np.array([[-0.05592252, -0.05592252,  0.05592252],
                          [ 0.05671207,  0.05671207, -0.05671207]])
        mf = dft.KUKS(cell, kpts).to_gpu()
        mf.xc = 'pbe,pbe'
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad(self):
        # g_ref = numerical_gradient(cell, 'r2scan')
        g_ref = np.array([[-0.05286804, -0.05286804,  0.05286804],
                          [ 0.05365182,  0.05365182, -0.05365181]])
        mf = cell.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'r2scan')
        g_ref = np.array([[-0.05286804, -0.05286804,  0.05286804],
                          [ 0.05365182,  0.05365182, -0.05365181]])
        mf = cell.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad_without_pseudo(self):
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
        # g_ref = numerical_gradient(cell_no_pseudo, 'r2scan')
        g_ref = np.array([[ 0.05745937,  0.05745937, -0.05745937],
                          [-0.05767467, -0.05767467,  0.05767467]])
        mf = cell_no_pseudo.KUKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell_no_pseudo)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_hybrid_grad(self):
        # ref = numerical_gradient(cell, xc='pbe0')
        ref = np.array([[-0.05102351, -0.05102351,  0.05102351],
                        [ 0.05168613,  0.05168613, -0.05168613]])
        mf = cell.KUKS(xc='pbe0', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skip('Insufficient GPU memory for rsjk.q_cond')
    def test_hse_grad(self):
        # ref = numerical_gradient(cell, xc='hse06')
        ref = np.array([[-0.05104506, -0.05104506,  0.05104506],
                        [ 0.05201861,  0.05201861, -0.05201861]])
        mf = cell.KUKS(xc='hse06', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_wb97_grad(self):
        # ref = numerical_gradient(cell, xc='wb97')
        ref = np.array([[-0.04605253, -0.04605253,  0.04605252],
                        [ 0.04315768,  0.04315768, -0.04315768]])
        mf = cell.KUKS(xc='wb97', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    def test_camb3lyp_grad(self):
        # ref = numerical_gradient(cell, xc='camb3lyp')
        ref = np.array([[-0.04507094, -0.04507094,  0.04507094],
                        [ 0.0487566 ,  0.0487566 , -0.0487566 ]])
        mf = cell.KUKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_camb3lyp_grad_without_pseudo(self):
        # ref = numerical_gradient(cell_no_pseudo, xc='camb3lyp')
        ref = np.array([[ 0.05796211,  0.05796211, -0.05796211],
                        [-0.05796231, -0.05796231,  0.05796231]])
        mf = cell_no_pseudo.KUKS(xc='camb3lyp', kpts=kpts).to_gpu()
        mf._numint = multigrid_v2.MultiGridNumInt(cell_no_pseudo)
        mf.rsjk = PBCJKMatrixOpt(cell_no_pseudo)
        mf.j_engine = PBCJMatrixOpt(cell_no_pseudo)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for KUKS Gradients")
    unittest.main()
