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
from pyscf import lib
from pyscf.pbc import gto
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-3

def setUpModule():
    global cell, kpts
    cell = gto.Cell()
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680]]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.basis = [[0, [3.3, 1]], [0, [1.1, 1]], [1, [0.8, 1]]]
    cell.verbose = 5
    cell.pseudo = 'gth-pade'
    cell.unit = 'bohr'
    cell.output = '/dev/null'
    cell.build()

    kpts = cell.make_kpts([1,1,2])

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):
    @unittest.skip('Gradients without pseudo potential')
    def test_uhf_grad(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680]]]
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]]]
        cell.unit = 'bohr'
        cell.build()
        mf = cell.UHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], 0.00016272817818590305)
        self.assertAlmostEqual(lib.fp(g), 0.00010682200307755532, 6)

        mf = cell.UHF()
        mfs = mf.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_uhf_with_pseudo_grad(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680]]]
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]]]
        cell.pseudo = 'gth-pbe'
        cell.unit = 'bohr'
        cell.build()
        mf = cell.UHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], 0.0001656583785769376)
        self.assertAlmostEqual(lib.fp(g), 0.00010874300386520308, 6)

        mf = cell.UHF()
        mfs = mf.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_kuhf_grad(self):
        mf = cell.KUHF(kpts=kpts, exxdiv=None).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], 0.1211648308588867, 5)
        self.assertAlmostEqual(lib.fp(g), 0.4940831933171378, 6)
        mf = cell.KUHF(kpts=kpts, exxdiv=None).to_gpu()
        mfs = mf.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

if __name__ == "__main__":
    print("Full Tests for KUHF Gradients")
    unittest.main()
