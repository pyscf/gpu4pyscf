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
from pyscf.gto import ATOM_OF, intor_cross
from pyscf.pbc import dft, gto, grad
from pyscf.pbc.tools import pbc
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.numint import NumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.grad import uks_stress, uks
from gpu4pyscf.pbc.grad.uks_stress import _finite_diff_cells
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.lib.multi_gpu import num_devices
import pytest

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

def _check_vs_finite_diff(dat, mf_scanner):
    cell = mf_scanner.cell
    vol = cell.vol
    disp = 1e-3
    for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
        cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
        e1 = mf_scanner(cell1)
        e2 = mf_scanner(cell2)
        assert abs(dat[i,j] - (e1-e2)/2/disp/vol) < .5e-6

class KnownValues(unittest.TestCase):
    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        xc = 'lda,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc).to_gpu())
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 2e-9

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        dm *= .5
        xc = 'pbe,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc).to_gpu())
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-8

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        xc = 'm06,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc).to_gpu())
        dat = uks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) - (.5+.2j)
        dm = np.einsum('spi,sqi->spq', dm, dm.conj())
        dm *= .5
        xc = 'lda,'
        mf_grad = uks.Gradients(cell.UKS(xc=xc).to_gpu())
        dat = uks_stress.get_vxc(mf_grad, cell, dm, with_j=True)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vj1 = FFTDF(cell1).get_jk(dm.sum(axis=0), with_k=False)[0]
            exc1 = ni.nr_uks(cell1, UniformGrids(cell1), xc, dm)[1]
            vj2 = FFTDF(cell2).get_jk(dm.sum(axis=0), with_k=False)[0]
            exc2 = ni.nr_uks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('sij,ji->', dm, (vj1-vj2)) * .5
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_lda_vs_finite_difference(self):
        xc = 'svwn'
        mf0 = cell.UKS(xc=xc).to_gpu()
        mf = mf0.multigrid_numint().run()
        mf_grad = uks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

        ref = dat
        dat = mf0.reset(cell).run().Gradients().get_stress()
        assert abs(dat - ref).max() < 1e-8

    def test_gga_vs_finite_difference(self):
        xc = 'pbe'
        mf = cell.UKS(xc=xc).to_gpu().multigrid_numint().run()
        mf_grad = uks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_mgga_vs_finite_difference(self):
        xc = 'rscan'
        mf = cell.UKS(xc=xc).to_gpu().multigrid_numint().run()
        mf_grad = uks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    @unittest.skipIf(num_devices > 1, '')
    def test_pbe0_vs_finite_difference(self):
        xc = 'pbe0'
        mf = cell.UKS(xc=xc).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.UKS(xc=xc).to_gpu().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    @pytest.mark.slow
    def test_hse_vs_finite_difference(self):
        xc = 'hse06'
        mf = cell.UKS(xc=xc).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.UKS(xc=xc).to_gpu().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_pbe0_vs_finite_difference(self):
        xc = 'pbe0'
        mf = cell.UKS(xc=xc).to_gpu().density_fit()
        mf = mf.multigrid_numint().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_pbe_vs_finite_difference(self):
        xc = 'pbe'
        mf = cell.UKS(xc=xc).to_gpu().density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_hse_vs_finite_difference(self):
        xc = 'hse06'
        mf = cell.UKS(xc=xc).to_gpu().density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

if __name__ == "__main__":
    print("Full Tests for UKS Stress tensor")
    unittest.main()
