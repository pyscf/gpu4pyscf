#!/usr/bin/env python
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

import unittest
import numpy as np
import pyscf
from gpu4pyscf.pbc.dft import multigrid_v2 as multigrid
from pyscf.pbc.grad import krks as krks_cpu

disp = 1e-5

def setUpModule():
    global cell, cell_orth
    cell = pyscf.M(
        # The original geometry of second carbon is [1.685068664391,1.685068664391,1.685068664391]
        # Henry distorted it to make the gradient non-zero
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.885068664391,1.685068664391,1.585068664391]]],
        a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000''',
        basis = 'gth-szv',
        pseudo = 'gth-pade',
        unit = 'bohr')

    cell_orth = pyscf.M(
        atom = 'H 0 0 0; H 1. 1. 1.',
        a = np.eye(3) * 2.5,
        basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
        verbose = 5,
        pseudo = 'gth-pade',
        unit = 'bohr',
        mesh = [13] * 3,
        output = '/dev/null')

def tearDownModule():
    global cell_orth, cell
    cell_orth.stdout.close()
    del cell_orth, cell

def numerical_gradient(cell, xc):
    def get_energy(cell):
        mf = cell.RKS(xc=xc).to_gpu()
        mf.conv_tol = 1e-10
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
        kmf = cell_orth.KRKS(xc='svwn').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    def test_lda_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='lda,vwn')
        ref = np.array([[-0.2574167 , -0.03094249,  0.12969496],
                        [ 0.25741799,  0.03094078, -0.12969543]])
        mf = cell.RKS(xc='lda,vwn').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_gga_grad(self):
        kmf = cell_orth.KRKS(xc='pbe').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    def test_gga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='pbe,pbe')
        ref = np.array([[-0.25588534, -0.03079455,  0.12893533],
                        [ 0.25585186,  0.03078769, -0.12891839]])
        mf = cell.RKS(xc='pbe,pbe').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_mgga_grad(self):
        # ref = numerical_gradient(cell_orth, xc='r2scan')
        ref = np.array([[-0.01026366, -0.01026366, -0.01026366],
                        [ 0.01026374,  0.01026374,  0.01026374]])
        mf = cell_orth.RKS(xc='r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_mgga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='r2scan')
        ref = np.array([[-0.26574535, -0.03127805,  0.13378557],
                        [ 0.265949  ,  0.03133622, -0.13367209]])
        mf = cell.RKS(xc='r2scan,r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skip('gradients for hybrid functional not avaiable')
    def test_hybrid_grad(self):
        ref = numerical_gradient(cell_orth, xc='b3lyp')
        # TODO: save the ref
        mf = cell_orth.RKS(xc='b3lyp').to_gpu()
        mf.exxdiv = None
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skip('gradients for hybrid functional not avaiable')
    def test_hybrid_grad_nonorth(self):
        ref = numerical_gradient(cell, xc='b3lyp')
        # TODO: save the ref
        mf = cell.RKS(xc='b3lyp').to_gpu()
        mf.exxdiv = None
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for RKS Gradients")
    unittest.main()
