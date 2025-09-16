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

disp = 1e-5

def setUpModule():
    global cell, kpts
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

    kpts = cell.make_kpts([1,1,3])

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

def numerical_gradient(cell, xc):
    def get_energy(cell):
        mf = cell.KRKS(xc=xc, kpts=kpts).to_gpu()
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
    return gradient


class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        # g_ref = numerical_gradient(cell, 'svwn')
        g_ref = np.array([[-0.0570564 , -0.0570564 ,  0.0570564 ],
                          [ 0.05723334,  0.05723334, -0.05723334]])
        mf = cell.KRKS(xc='svwn', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_lda_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'svwn')
        g_ref = np.array([[-0.0570564 , -0.0570564 ,  0.0570564 ],
                          [ 0.05723334,  0.05723334, -0.05723334]])
        mf = cell.KRKS(xc='svwn', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad(self):
        # g_ref = numerical_gradient(cell, 'pbe')
        g_ref = np.array([[-0.05592252, -0.05592252,  0.05592252],
                          [ 0.05671208,  0.05671207, -0.05671207]])
        mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_gga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'pbe')
        g_ref = np.array([[-0.05592252, -0.05592252,  0.05592252],
                          [ 0.05671208,  0.05671207, -0.05671207]])
        mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
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
        mf = cell.KRKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    def test_mgga_grad_multigrid_v2(self):
        # g_ref = numerical_gradient(cell, 'r2scan')
        g_ref = np.array([[-0.05286804, -0.05286804,  0.05286804],
                          [ 0.05365182,  0.05365182, -0.05365181]])
        mf = cell.KRKS(xc='r2scan', kpts=kpts).to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        mf._numint = multigrid_v2.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

    @unittest.skip('hybrid funcitonal deriviatives')
    def test_hybrid_grad(self):
        g_ref = numerical_gradient(cell, 'pbe')
        # TODO: save the g_ref
        mf = cell.KRKS(xc='b3lyp', kpts=kpts).to_gpu()
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        np.testing.assert_almost_equal(g, g_ref, 7)

if __name__ == "__main__":
    print("Full Tests for KRKS Gradients")
    unittest.main()
