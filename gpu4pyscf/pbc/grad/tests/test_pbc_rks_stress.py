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
from pyscf.pbc import dft, gto
from pyscf.pbc.tools import pbc
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.numint import NumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.grad import rks_stress, rks
from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
import pytest

class KnownValues(unittest.TestCase):
    def test_coulG(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.5, 1]]], a=a, unit='Bohr')
        coulG0, coulG1 = rks_stress._get_coulG_strain_derivatives(cell, cell.Gv)
        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-5)
        assert abs(coulG1[0,0].get() - (pbc.get_coulG(cell1) - pbc.get_coulG(cell2)) / 2e-5).max() < 1e-9
        cell1, cell2 = _finite_diff_cells(cell, 0, 1, disp=1e-5)
        assert abs(coulG1[0,1].get() - (pbc.get_coulG(cell1) - pbc.get_coulG(cell2)) / 2e-5).max() < 1e-9

    def test_eval_ao_cart(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]], a=a, unit='Bohr', cart=True)
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)[0]
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = ni.eval_ao(cell1, coords)
            ao2 = ni.eval_ao(cell2, coords)
            assert abs(ao_value[i,j,0] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_cart(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [3, [.7, 1]]], a=a, unit='Bohr', cart=True)
        coords = np.random.rand(10, 3)
        ao_value = rks_stress._eval_ao_strain_derivatives(cell, coords, deriv=1)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)[0]
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = ni.eval_ao(cell1, coords, deriv=1)
            ao2 = ni.eval_ao(cell2, coords, deriv=1)
            assert abs(ao_value[i,j] - (ao1 - ao2) / 2e-5).max() < 1e-9

    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-8

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]],
                     precision=1e-9, a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'pbe,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-12
            cell2.precision = 1e-12
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]],
                     precision=1e-9, a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'm06,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-12
            cell2.precision = 1e-12
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            assert abs(dat[i,j] - (exc1 - exc2)/2e-5) < 1e-9

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_j=True)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vj1 = FFTDF(cell1).get_jk(dm, with_k=False)[0]
            vj1 *= .5
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vj2 = FFTDF(cell2).get_jk(dm, with_k=False)[0]
            vj2 *= .5
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vj1-vj2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_get_nuc(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_nuc=True)
        kpt = np.zeros(3)
        ni = NumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vne1 = FFTDF(cell1).get_nuc(kpt)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vne2 = FFTDF(cell2).get_nuc(kpt)
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vne1-vne2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_get_pp(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='Si 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]],
                     precision=1e-9, pseudo='gth-pade', a=a, unit='Bohr')
        nao = cell.nao
        dm = np.random.rand(nao, nao) - .5
        dm = dm.dot(dm.T)
        xc = 'lda,'
        mf_grad = rks.Gradients(cell.RKS(xc=xc).to_gpu())
        dat = rks_stress.get_vxc(mf_grad, cell, dm, with_nuc=True)
        ni = NumInt()
        kpt = np.zeros(3)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vne1 = FFTDF(cell1).get_pp(kpt)
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm)[1]
            vne2 = FFTDF(cell2).get_pp(kpt)
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm)[1]
            de = np.einsum('ij,ji', dm, (vne1-vne2))
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-5) < 1e-8

    def test_lda_vs_finite_difference(self):
        a = np.eye(3) * 3
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        mf = cell.RKS(xc='svwn').to_gpu().run()
        mf_grad = rks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_gga_vs_finite_difference(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='C 1 1 1; C 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     pseudo='gth-pade', a=a, unit='Bohr', verbose=0)
        mf = cell.RKS(xc='pbe').to_gpu().run()
        mf_grad = rks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    @pytest.mark.slow
    def test_mgga_vs_finite_difference(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        mf = cell.RKS(xc='rscan').to_gpu().run()
        mf_grad = rks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-6

    def test_pbe0_vs_finite_difference(self):
        a = np.eye(3) * 3.5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [0, [.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        xc = 'pbe0'
        mf = cell.RKS(xc=xc).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.RKS(xc=xc).to_gpu().as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 1e-7

    @pytest.mark.slow
    def test_hse_vs_finite_difference(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='H 1 1 1; H 2 1.5 2.4',
                     basis=[[0, [1.5, 1]], [0, [.5, 1]], [1, [.8, 1]]],
                     a=a, unit='Bohr', verbose=0)
        xc = 'hse06'
        mf = cell.RKS(xc=xc).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.RKS(xc=xc).to_gpu().as_scanner()
        vol = cell.vol
        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-3)
            e1 = mf_scanner(cell1)
            e2 = mf_scanner(cell2)
            assert abs(dat[i,j] - (e1-e2)/2e-3/vol) < 2e-7

if __name__ == "__main__":
    print("Full Tests for RKS Stress tensor")
    unittest.main()
