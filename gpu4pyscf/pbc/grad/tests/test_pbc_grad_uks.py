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
from pyscf.pbc.grad import kuks as kuks_cpu
from gpu4pyscf.lib.multi_gpu import num_devices
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-4

def setUpModule():
    global cell, cell_orth, cell_no_pseudo
    cell = pyscf.M(
        # The original geometry of second carbon is [1.685068664391,1.685068664391,1.685068664391]
        # Henry distorted it to make the gradient non-zero
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.585068664391,1.685068664391,1.885068664391]]],
        a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000''',
        basis = 'gth-szv',
        pseudo = 'gth-pade',
        unit = 'bohr',
        output = '/dev/null',
    )

    cell_orth = pyscf.M(
        atom = 'H 0 0 0; H 1. 1. 1.',
        a = np.eye(3) * 3.5,
        basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
        verbose = 5,
        pseudo = 'gth-pade',
        unit = 'bohr',
        output = '/dev/null',
    )

    cell_no_pseudo = pyscf.M(
        atom = 'H 0 0 0; H 1. 1. 1.',
        a = np.eye(3) * 3.5,
        basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
        verbose = 5,
        # pseudo = 'gth-pade',
        unit = 'bohr',
        output = '/dev/null',
    )

def tearDownModule():
    global cell, cell_orth, cell_no_pseudo
    cell.stdout.close()
    del cell
    cell_orth.stdout.close()
    del cell_orth
    cell_no_pseudo.stdout.close()
    del cell_no_pseudo

def numerical_gradient(cell, xc):
    def get_energy(cell):
        mf = cell.UKS(xc=xc).to_gpu()
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
    print(f"ref = np.{repr(gradient)}")
    return gradient

class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        kmf = cell_orth.KUKS(xc='svwn').run()
        ref = kuks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.UKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skipIf(num_devices > 1, '')
    def test_lda_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='lda,vwn')
        ref = np.array([[ 0.12969496, -0.03094249, -0.2574167 ],
                        [-0.12969543,  0.03094078,  0.25741799]])
        mf = cell.UKS(xc='lda,vwn').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf._numint = multigrid.MultiGridNumInt(cell)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_gga_grad(self):
        kmf = cell_orth.KUKS(xc='pbe').run()
        ref = kuks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.UKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skipIf(num_devices > 1, '')
    def test_gga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='pbe,pbe')
        ref = np.array([[ 0.12893533, -0.03079455, -0.25588534],
                        [-0.12891839,  0.03078769,  0.25585186]])
        mf = cell.UKS(xc='pbe,pbe').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf._numint = multigrid.MultiGridNumInt(cell)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_mgga_grad(self):
        # ref = numerical_gradient(cell_orth, xc='r2scan')
        ref = np.array([[-0.22997632, -0.22997632, -0.22997632],
                        [ 0.22997655,  0.22997655,  0.22997655]])
        mf = cell_orth.UKS(xc='r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skipIf(num_devices > 1, '')
    def test_mgga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='r2scan')
        ref = np.array([[ 0.13378557, -0.03127805, -0.26574535],
                        [-0.13367209,  0.03133622,  0.265949  ]])
        mf = cell.UKS(xc='r2scan,r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf._numint = multigrid.MultiGridNumInt(cell)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_mgga_grad_without_pseudo(self):
        # ref = numerical_gradient(cell_no_pseudo, xc='r2scan')
        ref = np.array([[-0.23043204, -0.23043204, -0.23043204],
                        [ 0.23043227,  0.23043227,  0.23043227]])
        mf = cell_no_pseudo.UKS(xc='r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = multigrid.MultiGridNumInt(cell_no_pseudo)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    def test_hybrid_grad(self):
        # ref = numerical_gradient(cell_orth, xc='pbe0')
        ref = np.array([[-0.23059506, -0.23059506, -0.23059506],
                        [ 0.23059781,  0.23059781,  0.23059781]])
        mf = cell_orth.UKS(xc='pbe0').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skip('Insufficient GPU memory for rsjk.q_cond')
    def test_hse_grad(self):
        # ref = numerical_gradient(cell_orth, xc='hse06')
        ref = np.array([[-0.23039771, -0.23039771, -0.23039771],
                        [ 0.23043268,  0.23043268,  0.23043268]])
        mf = cell_orth.UKS(xc='hse06').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_wb97_grad(self):
        # ref = numerical_gradient(cell_orth, xc='wb97')
        ref = np.array([[-0.22096546, -0.22096546, -0.22096546],
                        [ 0.22118384,  0.22118384,  0.22118384]])
        mf = cell_orth.UKS(xc='wb97').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_wb97_grad_without_pseudo(self):
        # ref = numerical_gradient(cell_no_pseudo, xc='wb97')
        ref = np.array([[-0.22143687, -0.22143687, -0.22143687],
                        [ 0.22165506,  0.22165506,  0.22165506]])
        mf = cell_no_pseudo.UKS(xc='wb97').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_no_pseudo)
        mf.rsjk = PBCJKMatrixOpt(cell_no_pseudo)
        mf.j_engine = PBCJMatrixOpt(cell_no_pseudo)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_no_pseudo)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_camb3lyp_grad(self):
        # ref = numerical_gradient(cell_orth, xc='camb3lyp')
        ref = np.array([[-0.22851045, -0.22851045, -0.22851045],
                        [ 0.22850896,  0.22850896,  0.22850896]])
        mf = cell_orth.UKS(xc='camb3lyp').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for UKS Gradients")
    unittest.main()
