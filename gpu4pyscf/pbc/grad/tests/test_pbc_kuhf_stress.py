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
from pyscf.pbc import gto
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt

def setUpModule():
    global cell
    a = np.eye(3) * 4
    np.random.seed(5)
    a -= np.random.rand(3, 3)
    cell = gto.M(atom='H 1 1 1; H 3 2.5 2.4',
                 basis=[[0, [1.5, 1]], [0, [.5, 1]], [1, [.8, 1]]],
                 pseudo='''
H GTH-PBE-q1 GTH-PBE
1
  0.20000000    2    -4.17890044     0.72446331
0
                 ''',
                 precision=1e-9,
                 verbose=6, output='/dev/null', a=a, unit='Bohr')

def tearDownModule():
    global cell
    del cell

def _check_vs_finite_diff(dat, mf_scanner, disp=1e-3, tol=1e-7):
    cell = mf_scanner.cell
    vol = cell.vol
    for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        e1 = mf_scanner(cell1)
        e2 = mf_scanner(cell2)
        assert abs(dat[i,j] - (e1-e2)/2/disp/vol) < tol

class KnownValues(unittest.TestCase):
    def test_kuhf_vs_finite_difference(self):
        kmesh = [3, 1, 1]
        mf = cell.KUHF(kpts=cell.make_kpts(kmesh)).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell).build()
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.KUHF(kpts=cell.make_kpts(kmesh)).to_gpu().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_uhf_vs_finite_difference(self):
        mf = cell.UHF().to_gpu()
        mf.run()
        mf.rsjk = PBCJKMatrixOpt(cell).build()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.UHF().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner, disp=.5e-3)

    def test_gdf_uhf_vs_finite_difference(self):
        mf = cell.UHF().to_gpu().density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_kuhf_vs_finite_difference(self):
        kmesh = [3, 1, 1]
        mf = cell.KUHF(kpts=cell.make_kpts(kmesh)).to_gpu().density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

if __name__ == "__main__":
    print("Full Tests for KUHF Stress tensor")
    unittest.main()
