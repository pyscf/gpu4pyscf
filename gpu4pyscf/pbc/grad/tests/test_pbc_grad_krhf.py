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
from packaging.version import Version
import pyscf
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.grad import krhf as krhf_cpu
from gpu4pyscf.pbc.grad import krhf as krhf_gpu
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc.df import AFTDF
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

disp = 1e-3

def setUpModule():
    global cell
    cell = gto.Cell()
    cell.atom= [['C', [0.0, 0.0, 0.0]], ['C', [1.685,1.685,1.6]]]
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

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):
    def test_rhf_grad(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [1.5,1.5,1.1]]]
        cell.a = '''
        0.00, 3.37, 3.37
        3.37, 0.00, 3.37
        3.37, 3.37, 0.00'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]], [1, [1., 1]]]
        cell.unit = 'bohr'
        cell.build()
        mf = cell.RHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(g[1,2], 0.01669204581120408, 6)
        self.assertAlmostEqual(lib.fp(g), -0.004299739901011966, 6)

        mfs = mf.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.5,1.5,1.1+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.5,1.5,1.1-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_rhf_with_pseudo_grad(self):
        mf = cell.RHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(g[1,2], -0.125769199473623, 6)
        self.assertAlmostEqual(lib.fp(g), -0.087662458760762, 6)

        atom_coords = cell.atom_coords()
        mfs = mf.as_scanner()
        atom_coords[1,2] += disp/2.0
        e1 = mfs(atom_coords)
        atom_coords[1,2] -= disp
        e2 = mfs(atom_coords)
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

        mf = cell.RHF().to_gpu()
        mf.with_df = AFTDF(cell)
        g1 = mf.Gradients().kernel()
        self.assertAlmostEqual(g1[1,2], -0.125769199473623, 6)
        self.assertAlmostEqual(lib.fp(g1), -0.087662458760762, 6)
        self.assertAlmostEqual(abs(g1 - g).max(), 0, 8)

    def test_df_rhf_grad(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [0.5,1.0,1.1]]]
        cell.a = '''
        0.00, 3.37, 3.37
        3.37, 0.00, 3.37
        3.37, 3.37, 0.00'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]]]
        cell.unit = 'bohr'
        cell.build()
        mf = cell.RHF().to_gpu().density_fit()
        mf.conv_tol_grad = 1e-9
        g = mf.Gradients().kernel()

        # numerical_gradient = np.empty((cell.natm, 3))
        # mfs = mf.as_scanner()
        # dx = 1e-6
        # for i_atom in range(cell.natm):
        #     for i_xyz in range(3):
        #         xyz_p = cell.atom_coords()
        #         xyz_p[i_atom, i_xyz] += dx
        #         e_p = mfs(xyz_p)

        #         xyz_m = cell.atom_coords()
        #         xyz_m[i_atom, i_xyz] -= dx
        #         e_m = mfs(xyz_m)

        #         numerical_gradient[i_atom, i_xyz] = (e_p - e_m) / (2.0 * dx)
        # np.set_printoptions(precision = 16)
        # print(repr(numerical_gradient))

        numerical_gradient = np.array([
            [-0.0168897392183176, -0.0260050767586506, -0.0327835129598775],
            [ 0.0168897393848511,  0.0260050764255837,  0.0327835128488552],
        ])

        assert np.max(np.abs(g - numerical_gradient)) < 2e-8

    def test_df_rhf_grad_with_pseudo(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [0.5,1.0,1.1]]]
        cell.a = '''
        0.00, 3.37, 3.37
        3.37, 0.00, 3.37
        3.37, 3.37, 0.00'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]]]
        cell.pseudo = 'gth-pbe'
        cell.unit = 'bohr'
        cell.build()
        mf = cell.RHF().to_gpu().density_fit()
        mf.conv_tol_grad = 1e-9
        g = mf.Gradients().kernel()

        # numerical_gradient = np.empty((cell.natm, 3))
        # mfs = mf.as_scanner()
        # dx = 1e-6
        # for i_atom in range(cell.natm):
        #     for i_xyz in range(3):
        #         xyz_p = cell.atom_coords()
        #         xyz_p[i_atom, i_xyz] += dx
        #         e_p = mfs(xyz_p)

        #         xyz_m = cell.atom_coords()
        #         xyz_m[i_atom, i_xyz] -= dx
        #         e_m = mfs(xyz_m)

        #         numerical_gradient[i_atom, i_xyz] = (e_p - e_m) / (2.0 * dx)
        # np.set_printoptions(precision = 16)
        # print(repr(numerical_gradient))

        numerical_gradient = np.array([
            [-0.0175469374585902, -0.0273011047657867, -0.0341998128705612],
            [ 0.0175469373475678,  0.027301104488231 ,  0.0341998133146504],
        ])

        assert np.max(np.abs(g - numerical_gradient)) < 2e-8

    def test_krhf_grad_with_pseudo(self):
        kpts = cell.make_kpts([1,1,2])
        mf = cell.KRHF(kpts=kpts).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], 0.020092574683078568, 5)
        self.assertAlmostEqual(lib.fp(g), 0.46776574928545617, delta=1e-6)

        atom_coords = cell.atom_coords()
        mfs = mf.as_scanner()
        atom_coords[1,2] += disp/2.0
        e1 = mfs(atom_coords)
        atom_coords[1,2] -= disp
        e2 = mfs(atom_coords)
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, delta=5e-6)

        mf = cell.KRHF(kpts=kpts).to_gpu()
        mf.with_df = AFTDF(cell, kpts)
        g1 = mf.Gradients().kernel()
        self.assertAlmostEqual(g1[1,2], 0.020092574683078568, 5)
        self.assertAlmostEqual(lib.fp(g1), 0.46776574928545617, delta=2e-6)
        self.assertAlmostEqual(abs(g1 - g).max(), 0, delta=1e-6)

    def test_krhf_grad1(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680]]]
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]]]
        cell.unit = 'bohr'
        cell.build()
        kpts = cell.make_kpts([1,1,2])
        mf = cell.KRHF(kpts=kpts, exxdiv='ewald').to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], -0.14946206095754058, 6)
        self.assertAlmostEqual(lib.fp(g), -0.5827692518230428, 6)

        mf = cell.KRHF(kpts=kpts, exxdiv='ewald').to_gpu()
        mfs = mf.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [1.685,1.685,1.680-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_df_krhf_grad(self):
        cell = gto.Cell()
        cell.atom= [['H', [0.0, 0.0, 0.0]], ['H', [0.5,1.0,1.1]]]
        cell.a = '''
        0.00, 3.37, 3.37
        3.37, 0.00, 3.37
        3.37, 3.37, 0.00'''
        cell.basis = [[0, [3., 1]], [0, [.8, 1]], [1, [.8, 1]]]
        cell.unit = 'bohr'
        cell.build()
        kpts = cell.make_kpts([1,1,3])
        mf = cell.KRHF(kpts=kpts).to_gpu().density_fit()
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(g[1,2], 0.04237479171431455, 6)
        self.assertAlmostEqual(lib.fp(g), -0.16258737449267388, 6)

        mfs = mf.as_scanner()
        e1 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [0.5,1.0,1.1+disp/2.0]]])
        e2 = mfs([['H', [0.0, 0.0, 0.0]], ['H', [0.5,1.0,1.1-disp/2.0]]])
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 6)

    def test_df_krhf_grad_with_pseudo(self):
        kpts = cell.make_kpts([1,1,3])
        mf = cell.KRHF(kpts=kpts).to_gpu().density_fit()
        g_scan = mf.Gradients().as_scanner()
        g = g_scan(cell)[1]
        self.assertAlmostEqual(g[1,2], 0.015171575401821263, 6)
        self.assertAlmostEqual(lib.fp(g), 0.48661781675093696, 6)

        atom_coords = cell.atom_coords()
        mfs = mf.as_scanner()
        atom_coords[1,2] += disp/2.0
        e1 = mfs(atom_coords)
        atom_coords[1,2] -= disp
        e2 = mfs(atom_coords)
        self.assertAlmostEqual(g[1,2], (e1-e2)/disp, 5)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.12'),
                     'The meaning of get_hcore in *.pbc.grad has been changed in pyscf==2.12. It doesn\'t include pseudopotential nonlocal term anymore.')
    def test_hcore(self):
        kpts = cell.make_kpts([1,1,3])
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
