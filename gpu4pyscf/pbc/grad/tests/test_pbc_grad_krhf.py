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
from pyscf.pbc.grad import krhf as krhf_cpu
from gpu4pyscf.pbc.grad import krhf as krhf_gpu
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-3

def setUpModule():
    global cell, kpts
    cell = gto.Cell()
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]]
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
    def test_krhf_grad(self):
        mf = cell.KRHF(exxdiv=None).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.00493036507528758, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

        mf = cell.KRHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.00493036507528758, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_krhf_grad1(self):
        mf = cell.KRHF(kpts=kpts, exxdiv=None).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.00493036507528758, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

        mf = cell.KRHF(kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(lib.fp(g), -0.00493036507528758, 6)

        mfs = g_scan.base.as_scanner()
        e1 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_hcore(self):
        with lib.temporary_env(numint, MIN_BLK_SIZE=1024):
            dat = krhf_gpu.get_hcore(cell, kpts)
            ref = krhf_cpu.get_hcore(cell, kpts)
            assert abs(dat.get() - ref).max() < 1e-8

            hcore_generator_gpu = krhf_gpu.Gradients(cell.KRHF(kpts=kpts)).hcore_generator()
            hcore_generator_cpu = krhf_cpu.Gradients(cell.KRHF(kpts=kpts)).hcore_generator()
            dat = hcore_generator_gpu(0)
            ref = hcore_generator_cpu(0)
            assert abs(dat.get() - ref.transpose(1,0,2,3)).max() < 1e-8
            dat = hcore_generator_gpu(1)
            ref = hcore_generator_cpu(1)
            assert abs(dat.get() - ref.transpose(1,0,2,3)).max() < 1e-8

if __name__ == "__main__":
    print("Full Tests for KRHF Gradients")
    unittest.main()
