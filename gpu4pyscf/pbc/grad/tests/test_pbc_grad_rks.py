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

disp = 1e-3

def setUpModule():
    global cell, cell_orth
    cell = pyscf.M(
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]],
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

class KnownValues(unittest.TestCase):

    def test_lda_grad(self):
        kmf = cell_orth.KRKS(xc='svwn').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skip('pyscf multigrid bug')
    def test_lda_grad_nonorth(self):
        mf = cell.RKS(xc='lda,vwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 4)

    def test_gga_grad(self):
        kmf = cell_orth.KRKS(xc='pbe').run()
        ref = krks_cpu.Gradients(kmf).kernel()
        mf = cell_orth.RKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell_orth)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        self.assertAlmostEqual(abs(g - ref).max(), 0, 5)

    @unittest.skip('pyscf multigrid bug')
    def test_gga_grad_nonorth(self):
        mf = cell.RKS(xc='pbe,pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 4)

    @unittest.skip('gradients for hybrid functional not avaiable')
    def test_hybrid_grad(self):
        mf = cell_orth.RKS(xc='b3lyp').to_gpu()
        mf.exxdiv = None
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell_orth)[1]
        mfs = g_scan.base.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.,1.,1.+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.,1.,1.-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 4)

    @unittest.skip('gradients for hybrid functional not avaiable')
    def test_hybrid_grad_nonorth(self):
        mf = cell.RKS(xc='b3lyp').to_gpu()
        mf.exxdiv = None
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(cell)[1]
        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 4)

if __name__ == "__main__":
    print("Full Tests for RKS Gradients")
    unittest.main()
