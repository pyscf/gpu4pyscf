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
from pyscf.pbc.grad import krks as krks_cpu
from gpu4pyscf.lib.multi_gpu import num_devices
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-4

def setUpModule():
    global cell, cell_orth, cell_be, cell_no_pseudo
    cell = pyscf.M(
        # The original geometry of second carbon is [1.685068664391,1.685068664391,1.685068664391]
        # Henry distorted it to make the gradient non-zero
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.885,1.685,1.585]]],
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

    cell_no_pseudo = pyscf.M(
        atom = [['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0]]],
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
    global cell, cell_orth
    cell.stdout.close()
    del cell
    cell_orth.stdout.close()
    del cell_orth

def numerical_gradient(cell, xc):
    def get_energy(cell):
        mf = cell.RKS(xc=xc)
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
            Ep = get_energy(cell_copy)

            xyz_m = cell.atom_coords()
            xyz_m[i_atom, i_xyz] -= disp
            cell_copy.set_geom_(xyz_m, unit='Bohr')
            Em = get_energy(cell_copy)

            gradient[i_atom, i_xyz] = (Ep - Em) / (2 * disp)
    print(f"ref = np.{repr(gradient)}")
    return gradient

class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        kmf = cell_orth.KRKS(xc='svwn').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='svwn').to_gpu()
        mf = mf.multigrid_numint()
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    def test_lda_grad_no_pseudo(self):
        mf = cell_no_pseudo.RKS(xc='svwn').to_gpu()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

        mf = cell_no_pseudo.RKS(xc='svwn').to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g1 = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g-g1).max(), 0, 8)

    @unittest.skipIf(num_devices > 1, '')
    def test_rsjk_lda_grad(self):
        # ref = numerical_gradient(cell_orth, xc='svwn')
        ref = np.array([[-0.22454693, -0.22454693, -0.22454693],
                        [ 0.22454697,  0.22454697,  0.22454697]])
        mf = cell_orth.RKS().to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_lda_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='lda,vwn')
        ref = np.array([[-0.25731923, -0.03086362,  0.1297622 ],
                        [ 0.25732052,  0.0308619 , -0.12976267]])
        mf = cell.RKS(xc='lda,vwn').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_gga_grad(self):
        kmf = cell_orth.KRKS(xc='pbe').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='pbe').to_gpu()
        mf = mf.multigrid_numint()
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skipIf(num_devices > 1, '')
    def test_gga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='pbe,pbe')
        ref = np.array([[-0.25578848, -0.03071615,  0.12900215],
                        [ 0.25575498,  0.03070929, -0.12898519]])
        mf = cell.RKS(xc='pbe,pbe').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_mgga_grad(self):
        # ref = numerical_gradient(cell_orth, xc='r2scan')
        ref = np.array([[-0.22997619, -0.22997619, -0.22997619],
                        [ 0.22997641,  0.22997641,  0.22997641]])
        mf = cell_orth.RKS(xc='r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf = mf.multigrid_numint()
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    @unittest.skipIf(num_devices > 1, '')
    def test_mgga_grad_nonorth(self):
        # ref = numerical_gradient(cell, xc='r2scan')
        ref = np.array([[-0.26564486, -0.03119648,  0.13385543],
                        [ 0.26584843,  0.03125564, -0.13374247]])
        mf = cell.RKS(xc='r2scan,r2scan').to_gpu()
        mf.conv_tol = 1e-10
        mf.run()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_hybrid_grad(self):
        # ref = numerical_gradient(cell_orth, xc='pbe0')
        ref = np.array([[-0.23059492, -0.23059492, -0.23059492],
                        [ 0.23059768,  0.23059768,  0.23059768]])
        mf = cell_orth.RKS(xc='pbe0').to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_pbe0_grad(self):
        mf = cell_no_pseudo.RKS(xc='pbe0').to_gpu()
        mf.exxdiv = None
        mf.rsjk = PBCJKMatrixOpt(cell_no_pseudo)
        mf.j_engine = PBCJMatrixOpt(cell_no_pseudo)
        mf = mf.multigrid_numint()
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_no_pseudo)[1]

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

        mf = cell_be.RKS(xc='pbe0').to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell_be)
        mf.j_engine = PBCJMatrixOpt(cell_be)
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_df_pbe0_grad(self):
        mf = cell_no_pseudo.RKS(xc='pbe0').to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

        mf = cell_be.RKS(xc='pbe0').to_gpu().density_fit()
        mf.exxdiv = None
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_sr_rsh_grad(self):
        xc = 'SR_HF(0.33)*.5 + 0.5*B88'
        mf = cell_be.RKS(xc=xc).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell_be)
        mf.j_engine = PBCJMatrixOpt(cell_be)
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_wb97_grad(self):
        # ref = numerical_gradient(cell_orth, xc='wb97')
        ref = np.array([[-0.22096546, -0.22096546, -0.22096546],
                        [ 0.22118384,  0.22118384,  0.22118384]])
        mf = cell_orth.RKS(xc='wb97').to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_df_wb97_grad(self):
        mf = cell_no_pseudo.RKS(xc='wb97').to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [1.5,1.0,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_camb3lyp_grad(self):
        # ref = numerical_gradient(cell_orth, xc='camb3lyp')
        ref = np.array([[-0.22851045, -0.22851045, -0.22851045],
                        [ 0.22850896,  0.22850896,  0.22850896]])
        mf = cell_orth.RKS(xc='camb3lyp').to_gpu()
        mf = mf.multigrid_numint()
        mf.rsjk = PBCJKMatrixOpt(cell_orth)
        mf.j_engine = PBCJMatrixOpt(cell_orth)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 6)

    def test_df_rsh_grad(self):
        xc = 'RSH(0.5, 0.8, -0.4)*.6 + 0.5*LDA'
        mf = cell_be.RKS(xc=xc).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    def test_df_sr_rsh_grad(self):
        xc = 'SR_HF(0.33)*.5 + 0.5*B88'
        mf = cell_be.RKS(xc=xc).to_gpu().density_fit()
        mf = mf.multigrid_numint()
        g = mf.Gradients().kernel()

        mfs = mf.as_scanner()
        e1 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0+disp/2.0]]])
        e2 = mfs([['Be', [0.0, 0.0, 0.0]], ['Be', [0.5,0.2,1.0-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=2e-6)

if __name__ == "__main__":
    print("Full Tests for RKS Gradients")
    unittest.main()
