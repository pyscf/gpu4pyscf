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
import cupy as cp
from pyscf.gto import ATOM_OF, intor_cross
from pyscf.pbc import dft, gto, grad
from pyscf.pbc.tools import pbc
from pyscf.pbc.dft.numint import KNumInt
from pyscf.pbc.dft.gen_grid import UniformGrids
from gpu4pyscf.pbc.df import FFTDF
from gpu4pyscf.pbc.dft import krkspu
from gpu4pyscf.pbc.grad import pp as pp_grad
from gpu4pyscf.pbc.grad import krkspu as krkspu_grad
from gpu4pyscf.pbc.grad import krks
from gpu4pyscf.pbc.grad.krks_stress import _eval_ao_strain_derivatives
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.dft.multigrid_v3 import MultiGridNumInt
from gpu4pyscf.lib.multi_gpu import num_devices
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix
import pytest

def setUpModule():
    global cell
    a = np.eye(3) * 4
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
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
        assert abs(dat[i,j] - (e1-e2)/2/disp/vol) < 1e-6

class KnownValues(unittest.TestCase):
    def test_eval_ao_cart_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [4, [.8, 1]]], a=a, unit='Bohr', cart=True)
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = _eval_ao_strain_derivatives(cell, coords, kpts)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh))
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh))
            assert abs(ao_value[0][i,j,0] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j,0] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j,0] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_eval_ao_sph_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [4, [.8, 1]]], a=a, unit='Bohr')
        assert not cell.cart
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = _eval_ao_strain_derivatives(cell, coords, kpts)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh))
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh))
            assert abs(ao_value[0][i,j,0] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j,0] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j,0] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_cart_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [4, [.8, 1]]], a=a, unit='Bohr', cart=True)
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=1)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh), deriv=1)
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh), deriv=1)
            assert abs(ao_value[0][i,j] - (ao1[0] - ao2[0]) / 2e-5).max() < 1e-9
            assert abs(ao_value[1][i,j] - (ao1[1] - ao2[1]) / 2e-5).max() < 1e-9
            assert abs(ao_value[2][i,j] - (ao1[2] - ao2[2]) / 2e-5).max() < 1e-9

    def test_eval_ao_deriv1_sph_kpts(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]],
                            [1, [1.5, 1], [.5, 1]],
                            [2, [.8, 1]],
                            [4, [.8, 1]]], a=a, unit='Bohr')
        assert not cell.cart
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        coords = np.random.rand(10, 3)
        ao_value = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=1)
        ao_value = ao_value.get().transpose(0,1,2,3,5,4)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-5)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            ao1 = dft.numint.eval_ao_kpts(cell1, coords, cell1.make_kpts(kmesh), deriv=1)
            ao2 = dft.numint.eval_ao_kpts(cell2, coords, cell2.make_kpts(kmesh), deriv=1)
            assert abs(ao_value[0][i,j] - (ao1[0] - ao2[0]) / 2e-5).max() < 2e-9
            assert abs(ao_value[1][i,j] - (ao1[1] - ao2[1]) / 2e-5).max() < 2e-9
            assert abs(ao_value[2][i,j] - (ao1[2] - ao2[2]) / 2e-5).max() < 2e-9

    def test_get_vxc_lda(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat2 = ni.energy_strain_gradient(xc, dm, kpts, spin=0, with_j=False, with_nuc=False)

        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat2[i,j] - (exc1 - exc2)/2e-4) < 1e-7

        ni = MultiGridNumInt(cell)
        dat2 = ni.energy_strain_gradient(xc, dm, kpts, spin=0, with_j=False, with_nuc=False)
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = MultiGridNumInt(cell1).nr_rks(cell1, None, xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = MultiGridNumInt(cell2).nr_rks(cell2, None, xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat2[i,j] - (exc1 - exc2)/2e-4) < 3e-7

    def test_get_vxc_gga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'pbe,'
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat2 = ni.energy_strain_gradient(xc, dm, kpts, spin=0, with_j=False, with_nuc=False)

        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat2[i,j] - (exc1 - exc2)/2e-4) < 1e-7

    def test_get_vxc_mgga(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a,
                     unit='Bohr', precision=1e-9)
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'm06,'
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat2 = ni.energy_strain_gradient(xc, dm, kpts, spin=0, with_j=False, with_nuc=False)

        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            assert abs(dat2[i,j] - (exc1 - exc2)/2e-4) < 1e-7

    def test_multigrid_image_boundary_strain_deriv(self):
        # The structure is from t-HfO2, all elements are changed to H.
        cell = gto.Cell()
        cell.atom = [
            ('H', (1.767935803700000, 1.767935803726000, 2.595423152457000)),
            ('H', (-0.000000000170000, -0.000000000413000, -0.000000000009000)),
            ('H', (-0.000000000217000, 1.767935803226000, 3.539517351764000)),
            ('H', (1.767935803466000, -0.000000000847000, 4.246760404410000)),
            ('H', (1.767935803449000, -0.000000000675000, 1.651328953277000)),
            ('H', (-0.000000000265000, 1.767935803396000, 0.944085900686000)),
        ]
        cell.a = [
            [3.535871610000000, 0.000000000293102, -0.000000000005598],
            [0.000000000293102, 3.535871610000000, -0.000000000129270],
            [-0.000000000007998, -0.000000000189320, 5.190846310000000],
        ]
        cell.unit = 'Angstrom'
        cell.basis = [[0, [1.0, 1.0]]]
        cell.rcut = 34.71833632109211
        cell.precision = 1e-10
        cell.mesh = [15, 15, 21]
        cell.build()

        cell_plus, cell_minus = _finite_diff_cells(cell, 0, 0, disp=1e-4)
        assert len(gto.eval_gto.get_lattice_Ls(cell_plus)) == 946
        # The minus will have one extra image.
        assert len(gto.eval_gto.get_lattice_Ls(cell_minus)) == 947

        kpts = cell.make_kpts([1, 1, 1], with_gamma_point=True)
        dm = np.eye(cell.nao)[None]
        xc = 'r2scan'

        from gpu4pyscf.pbc.dft.multigrid_v2 import (
            MultiGridNumInt as MultiGridNumIntV2,
            _rks_exc_strain_deriv,
        )
        ni_v2 = MultiGridNumIntV2(cell)
        ni_v2.mesh = np.asarray(cell.mesh)
        ni_v2.build('MGGA', gamma_point=False)
        dat = _rks_exc_strain_deriv(
            ni_v2, xc, dm, kpts, with_j=False, with_nuc=False)
        assert abs(dat / cell.vol).max() < 1e-3

        # V3 evaluates strain derivatives analytically without rebuilding
        # neighboring images for displaced cells.
        for enable_aft in (True, False):
            ni_v3 = MultiGridNumInt(cell)
            ni_v3.enable_aft = enable_aft
            ni_v3.mesh = np.asarray(cell.mesh)
            dat = ni_v3.energy_strain_gradient(
                xc, dm, kpts, spin=0, with_j=False, with_nuc=False)
            assert abs(dat / cell.vol).max() < 1e-3

    def test_get_j(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 3]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat = ni.energy_derivatives(xc, dm, spin=0, kpts=kpts, with_j=True, with_nuc=False)[-3:]
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            vj1 = FFTDF(cell1).get_jk(dm, kpts=cell1.make_kpts(kmesh), with_k=False)[0]
            vj1 *= .5
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vj2 = FFTDF(cell2).get_jk(dm, kpts=cell2.make_kpts(kmesh), with_k=False)[0]
            vj2 *= .5
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vj1-vj2)) / len(kpts)
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-4) < 1e-7

    def test_get_nuc(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='He 1 1 1; He 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]], a=a, unit='Bohr')
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat = ni.energy_derivatives(xc, dm, spin=0, kpts=kpts, with_nuc=True, with_j=False)[-3:]
        dat += pp_grad.ppnl_derivatives(cell, cp.array(dm), kpts)[-3:]
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vne1 = FFTDF(cell1).get_nuc(kpts=cell1.make_kpts(kmesh))
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vne2 = FFTDF(cell2).get_nuc(kpts=cell2.make_kpts(kmesh))
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vne1-vne2)) / len(kpts)
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-4) < 1e-7

    def test_get_pp(self):
        a = np.eye(3) * 5
        np.random.seed(5)
        a += np.random.rand(3, 3) - .5
        cell = gto.M(atom='C 1 1 1; Si 2 1.5 2.4',
                     basis=[[0, [.5, 1]], [1, [.8, 1]], [2, [.6, 1]]],
                     pseudo='gth-pade', a=a, unit='Bohr', precision=1e-9)
        kmesh = [3, 1, 1]
        nao = cell.nao
        dm = np.random.rand(np.prod(kmesh), nao, nao) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('kpi,kqi->kpq', dm, dm.conj())
        xc = 'lda,'
        kpts = cell.make_kpts(kmesh)
        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        dat = ni.energy_derivatives(xc, dm, spin=0, kpts=kpts, with_nuc=True, with_j=False)[-3:]
        dat += pp_grad.ppnl_derivatives(cell, cp.array(dm), kpts)[-3:]
        ni = KNumInt()
        for (i, j) in [(0, 0), (0, 1), (0, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            cell1.precision = 1e-10
            cell2.precision = 1e-10
            vne1 = FFTDF(cell1).get_pp(kpts=cell1.make_kpts(kmesh))
            exc1 = ni.nr_rks(cell1, UniformGrids(cell1), xc, dm, kpts=cell1.make_kpts(kmesh))[1]
            vne2 = FFTDF(cell2).get_pp(kpts=cell2.make_kpts(kmesh))
            exc2 = ni.nr_rks(cell2, UniformGrids(cell2), xc, dm, kpts=cell2.make_kpts(kmesh))[1]
            de = np.einsum('kij,kji', dm, (vne1-vne2)) / len(kpts)
            de += exc1 - exc2
            assert abs(dat[i,j] - de/2e-4) < 3e-7

    def test_lda_vs_finite_difference(self):
        xc = 'svwn'
        kmesh = [3, 1, 1]
        mf0 = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf0.multigrid_numint().run()
        mf_grad = krks.Gradients(mf)
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

        ref = dat
        dat = mf0.reset(cell).run().Gradients().get_stress()
        assert abs(dat - ref).max() < 1e-8

    def test_gga_vs_finite_difference(self):
        xc = 'pbe'
        kmesh = [3, 1, 1]
        mf0 = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf0.multigrid_numint()
        mf._numint.allow_mesh_reduction = False
        mf.run()
        mf_grad = mf.Gradients()
        ref = mf_grad.get_stress()

        mf = mf0.multigrid_numint()
        mf._numint.allow_mesh_reduction = True
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        assert abs(dat - ref).max() < 1e-6

        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

        ref = dat
        dat = mf0.reset(cell).run().Gradients().get_stress()
        assert abs(dat - ref).max() < 1e-8

    def test_mgga_vs_finite_difference(self):
        xc = 'scan'
        kmesh = [3, 1, 1]
        mf0 = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf0.multigrid_numint()
        mf._numint.allow_mesh_reduction = False
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

        ref = dat
        dat = mf0.reset(cell).run().Gradients().get_stress()
        assert abs(dat - ref).max() < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_pbe0_vs_finite_difference(self):
        xc = 'pbe0'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    @pytest.mark.slow
    def test_hse_vs_finite_difference(self):
        xc = 'hse06'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu().as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_pbe0_vs_finite_difference(self):
        xc = 'pbe0'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf.density_fit().multigrid_numint().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_pbe_vs_finite_difference(self):
        xc = 'pbe'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf.density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_gdf_hse_vs_finite_difference(self):
        xc = 'hse06'
        kmesh = [3, 1, 1]
        mf = cell.KRKS(xc=xc, kpts=cell.make_kpts(kmesh)).to_gpu()
        mf = mf.density_fit().run()
        mf_grad = mf.Gradients()
        dat = mf_grad.get_stress()
        mf_scanner = mf.as_scanner()
        _check_vs_finite_diff(dat, mf_scanner)

    def test_hubbard_U(self):
        cell = gto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            pseudo = 'gth-pbe')
        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = krkspu.KRKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao)
        mf.__dict__.update(cell.KRKS(kpts=kpts).to_gpu().run(max_cycle=1).__dict__)
        sigma = krkspu_grad._hubbard_U_derivatives(mf)[-3:]

        for (i, j) in [(1, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            mf.reset(cell1)
            e1 = mf.get_veff().E_U.real
            mf.reset(cell2)
            e2 = mf.get_veff().E_U.real
            assert abs(sigma[i,j] - (e1 - e2) / 2e-4) < 1e-7

    @pytest.mark.slow
    def test_krkspu_finite_diff(self):
        cell = gto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            pseudo = 'gth-pbe')
        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = krkspu.KRKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao).run()
        sigma = mf.Gradients().get_stress()
        mf_scanner = mf.as_scanner()

        cell1, cell2 = _finite_diff_cells(cell, 0, 0, disp=1e-3)
        e1 = mf_scanner(cell1)
        e2 = mf_scanner(cell2)
        assert abs(sigma[0,0] - (e1 - e2)/2e-3/cell.vol) < 1e-6

class StrainVXC_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = gto.M(
            atom='He .2 .3 .4; He 1.4 1.2 1.6', unit='Bohr',
            a=[[5., .3, .2], [.1, 5.5, .4], [.2, -.1, 6.]],
            mesh=[25, 27, 29], basis=[[0, [.8, 1.]], [1, [.6, 1.]]],
            pseudo='gth-pade', precision=1e-11, verbose=0)
        cls.scaled_kpts = np.array([[.25, 0., 0.], [-.25, 0., 0.]])
        cls.coeff = np.random.default_rng(51).normal(size=(2, cls.cell.nao, 3))
        cls.coeff /= np.sqrt(cls.cell.nao)

    def _cpu_energy(self, cell, density, xc, mesh, with_j, with_nuc):
        from pyscf.pbc.df import FFTDF as CpuFFTDF
        from pyscf.pbc.gto import pseudo
        kpts = cell.get_abs_kpts(self.scaled_kpts)
        grids = UniformGrids(cell)
        grids.mesh = mesh
        ni = KNumInt()
        energy = ni.nr_rks(cell, grids, xc, density, kpts=kpts)[1]
        df = CpuFFTDF(cell, kpts=kpts)
        df.mesh = mesh
        if with_j:
            vj = df.get_jk(density, kpts=kpts, with_k=False)[0]
            energy += .5 * np.einsum('kij,kji->', vj, density).real / len(kpts)
        if with_nuc:
            if cell._pseudo:
                gv = cell.get_Gv(mesh)
                vlocG = -np.einsum('ag,ag->g', cell.get_SI(mesh=mesh),
                                   pseudo.get_vlocG(cell, gv))
                rho = ni.get_rho(cell, density, grids, kpts=kpts)
                # Convert the Fourier potential to physical units, then integrate.
                vloc = np.fft.ifftn(vlocG.reshape(mesh)).real.ravel()
                vloc *= np.prod(mesh) / cell.vol
                energy += np.dot(rho, vloc) * cell.vol / np.prod(mesh)
            else:
                vne = df.get_nuc(kpts=kpts)
                energy += np.einsum('kij,kji->', vne, density).real / len(kpts)
        return float(energy)

    def _check_public_vxc(self, xc, with_j=False, with_nuc=False,
                          bare=False, grid_override=False, implicit_dm=False):
        from gpu4pyscf.pbc.dft import KRKS
        from gpu4pyscf.pbc.dft.numint import KNumInt as GpuNumInt
        from gpu4pyscf.pbc.dft.gen_grid import UniformGrids as GpuGrids
        from gpu4pyscf.pbc.grad import krks_stress
        cell = self.cell.copy()
        if bare:
            # Rebuilding a copied cell with pseudo=None retains its parsed _pseudo.
            cell = gto.M(atom=self.cell.atom, a=self.cell.a, unit=self.cell.unit,
                         mesh=self.cell.mesh, basis=self.cell.basis,
                         precision=self.cell.precision, verbose=0)
            self.assertFalse(cell._pseudo)
        kpts = cell.get_abs_kpts(self.scaled_kpts)
        mf = KRKS(cell, kpts=kpts, xc=xc)
        mf._numint = GpuNumInt()
        mf.grids = GpuGrids(cell)
        # A genuine positive orbital density suffices; no SCF is needed.
        mf.mo_coeff = cp.asarray(self.coeff)
        mf.mo_occ = cp.ones((2, 3))
        density = mf.make_rdm1()
        density_cpu = density.get()
        gradient = mf.Gradients()
        if grid_override:
            gradient.grids = GpuGrids(cell)
            gradient.grids.mesh = np.asarray(cell.mesh) + 2
        selected_grids = gradient.grids if grid_override else mf.grids
        mesh = np.asarray(selected_grids.mesh).copy()
        actual = krks_stress.get_vxc(
            gradient, cell, None if implicit_dm else density, kpts,
            with_j=with_j, with_nuc=with_nuc)
        self.assertEqual(actual.shape, (3, 3))
        step = 1e-4
        reference = np.empty((3, 3))
        for x in range(3):
            for y in range(3):
                plus, minus = _finite_diff_cells(cell, x, y, step)
                ep = self._cpu_energy(plus, density_cpu, xc, mesh, with_j, with_nuc)
                em = self._cpu_energy(minus, density_cpu, xc, mesh, with_j, with_nuc)
                reference[x, y] = (ep-em) / (2*step)
        # get_vxc returns dE/dstrain, without the stress tensor's 1/volume.
        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(density.get(), density_cpu)
        np.testing.assert_array_equal(mf.mo_coeff.get(), self.coeff)
        np.testing.assert_array_equal(selected_grids.mesh, mesh)

    def test_lda_uniform_strain(self):
        self._check_public_vxc('LDA,VWN', implicit_dm=True)

    def test_pbe_coulomb_gradient_grid_override(self):
        self._check_public_vxc('PBE', with_j=True, grid_override=True)

    def test_r2scan_local_pseudopotential_strain(self):
        self._check_public_vxc('R2SCAN', with_nuc=True)

    def test_pbe_coulomb_bare_nuclear_strain(self):
        self._check_public_vxc('PBE', with_j=True, with_nuc=True, bare=True)


if __name__ == "__main__":
    print("Full Tests for KRKS Stress tensor")
    unittest.main()
