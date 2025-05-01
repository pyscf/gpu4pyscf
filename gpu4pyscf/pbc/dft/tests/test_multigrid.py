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
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import multigrid as multigrid_cpu
from gpu4pyscf.pbc.dft import multigrid
from gpu4pyscf.pbc.tools import ifft, fft

diamond = '''
C     0.      0.      0.
C     0.8917  0.8917  0.8917
C     1.7834  1.7834  0.
C     2.6751  2.6751  0.8917
C     1.7834  0.      1.7834
C     2.6751  0.8917  2.6751
C     0.      1.7834  1.7834
C     0.8917  2.6751  2.6751'''

def setUpModule():
    global cell_orth
    global kpts, dm, dm1
    np.random.seed(2)
    cell_orth = gto.M(
        verbose = 7,
        output = '/dev/null',
        a = np.diag([3.6, 3.2, 4.5]),
        atom = '''C     0.      0.      0.
                  C     1.8     1.8     1.8   ''',
        basis = ('gth-dzv', [[3, [2., 1.]], [4, [1., 1.]]]),
        pseudo = 'gth-pade',
        precision = 1e-9,
    )

    kptsa = np.random.random((2,3))
    kpts = kptsa.copy()
    kpts[1] = -kpts[0]
    nao = cell_orth.nao_nr()
    dm = np.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + np.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)

def tearDownModule():
    global cell_orth
    del cell_orth

class KnownValues(unittest.TestCase):
    def test_get_pp(self):
        ref = multigrid_cpu.MultiGridFFTDF(cell_orth).get_pp()
        out = multigrid.MultiGridNumInt(cell_orth).get_pp().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc(self):
        ref = multigrid_cpu.MultiGridFFTDF(cell_orth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_eval_nucG(self):
        mesh = cell_orth.mesh
        SI = cell_orth.get_SI(mesh=mesh)
        ref = np.einsum('i,ij->j', -cell_orth.atom_charges(), SI)
        dat = multigrid.eval_nucG(cell_orth, mesh)
        self.assertAlmostEqual(abs(ref - dat.get()).max(), 0, 12)

    def test_eval_vpplocG(self):
        mesh = cell_orth.mesh
        Gv = cell_orth.get_Gv(mesh)
        SI = cell_orth.get_SI(Gv)
        ref = -np.einsum('ij,ij->j', pseudo.get_vlocG(cell_orth, Gv), SI)
        dat = multigrid.eval_vpplocG(cell_orth, mesh)
        self.assertAlmostEqual(abs(ref - dat.get()).max(), 0, 12)

    @unittest.skip('MultiGrid for kpts not implemented')
    def test_get_nuc_kpts(self):
        ref = multigrid.MultiGridFFTDF(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_rho(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        ref = multigrid_cpu.MultiGridFFTDF(cell_orth).get_rho(dm)
        out = multigrid.MultiGridNumInt(cell_orth).get_rho(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_j(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        ref = multigrid_cpu.MultiGridFFTDF(cell_orth).get_jk(dm[None], with_k=False)[0]
        out = multigrid.MultiGridNumInt(cell_orth).get_j(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_vxc_lda(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        n0, exc0, ref = multigrid_cpu.nr_rks(multigrid_cpu.MultiGridFFTDF(pcell), xc, dm, with_j=True)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        self.assertAlmostEqual(abs(n0-n1).max(), 0, 8)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 8)
        self.assertAlmostEqual(abs(ref-vxc.get()).max(), 0, 8)

    def test_get_vxc_gga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        n0, exc0, ref = multigrid_cpu.nr_rks(multigrid_cpu.MultiGridFFTDF(pcell), xc, dm, with_j=True)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        self.assertAlmostEqual(abs(n0-n1).max(), 0, 8)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 8)
        self.assertAlmostEqual(abs(ref-vxc.get()).max(), 0, 8)

    def test_get_vxc_mgga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'scan'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        mf = pcell.RKS()
        ni = mf._numint
        grids = mf.grids.build()
        n0, exc0, ref = ni.nr_rks(pcell, grids, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(
            cell_orth, None, xc, dm, with_j=False)
        self.assertAlmostEqual(abs(n0-n1).max(), 0, 8)
        self.assertAlmostEqual(abs(exc0-exc1).max(), 0, 8)
        self.assertAlmostEqual(abs(ref-vxc.get()).max(), 0, 8)

    def test_eval_tauG(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)

        mg = multigrid.MultiGridNumInt(cell_orth)
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        mf = pcell.RKS()
        grids = mf.grids.build()
        tau = multigrid._eval_tauG(mg, dm, kpts=np.zeros((1,3)))[0]
        ngrids = np.prod(cell_orth.mesh)
        weight = pcell.vol/ngrids
        tau = ifft(tau, cell_orth.mesh).real / weight
        ao = mf._numint.eval_ao(pcell, grids.coords, deriv=1)
        ref = mf._numint.eval_rho(pcell, ao, dm, xctype='MGGA', with_lapl=False)
        # TODO: adjust threshold in create_tasks to get 1e-8 precision
        self.assertAlmostEqual(abs(ref[4]-tau.get()).max(), 0, 7)

    def test_mat_tau(self):
        from pyscf.dft.numint import _tau_dot
        np.random.seed(2)
        vR = np.random.random(cell_orth.mesh).ravel() * .1
        vG = fft(cp.asarray(vR), cell_orth.mesh)

        mg = multigrid.MultiGridNumInt(cell_orth)
        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        mf = pcell.RKS()
        grids = mf.grids.build()
        mat = multigrid._get_tau_pass2(mg, vG[None], kpts=np.zeros((1,3)))
        ao = mf._numint.eval_ao(pcell, grids.coords, deriv=1)
        ngrids = vR.size
        mask = np.ones((ngrids, pcell.nbas), dtype=np.uint8)
        ref = _tau_dot(pcell, ao, ao, vR, mask, (0, pcell.nbas), pcell.ao_loc) * .5
        # TODO: adjust threshold in create_tasks to get 1e-8 precision
        self.assertAlmostEqual(abs(ref-mat.get()).max(), 0, 7)

    def test_rks_lda(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.777337612, 8)

    def test_uks_lda(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.UKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.777337612, 8)

    def test_rks_gga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.87059063524272, 8)

    def test_uks_gga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.UKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.87059063524272, 8)

    def test_rks_mgga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='scan').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.7542917283246, 8)

    def test_uks_mgga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = diamond,
            basis = 'gth-szv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.UKS(xc='tpss').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.6180709444, 8)

    @unittest.skip('MultiGrid for KRKS not implemented')
    def test_krks_lda(self):
        pass

    @unittest.skip('MultiGrid for KUKS not implemented')
    def test_kuks_lda(self):
        pass

    @unittest.skip('MultiGrid for GGA not implemented')
    def test_krks_gga(self):
        pass

    @unittest.skip('MultiGrid for GGA not implemented')
    def test_kuks_gga(self):
        pass

if __name__ == '__main__':
    print("Full Tests for multigrid")
    unittest.main()
