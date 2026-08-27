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
import pyscf
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import multigrid as multigrid_cpu
if hasattr(multigrid_cpu, 'MultiGridNumInt'):
    MultiGridNumInt_cpu = multigrid_cpu.MultiGridNumInt
else:
    MultiGridNumInt_cpu = multigrid_cpu.MultiGridFFTDF
from gpu4pyscf.pbc.dft import multigrid_v3 as multigrid
from gpu4pyscf.pbc.tools import ifft, fft
from gpu4pyscf.pbc.dft import numint, UniformGrids
from gpu4pyscf.pbc.dft import KRKS as KRKS_gpu
from gpu4pyscf.pbc.dft import KUKS as KUKS_gpu
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix

import pytest

def setUpModule():
    global cell_orth, cell_nonorth, cell_he
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
        unit = 'Bohr',
    )

    kptsa = np.random.random((2,3))
    kpts = kptsa.copy()
    kpts[1] = -kpts[0]
    nao = cell_orth.nao_nr()
    dm = np.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + np.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)

    cell_nonorth = pyscf.M(
        atom = [['C', [0.0, 0.0, 0.0]], ['C', [1.685068664391,1.685068664391,1.685068664391]]],
        a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000''',
        basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
        pseudo = 'gth-pade',
        unit = 'bohr',
        mesh = [18] * 3) # GGA needs dense mesh for derivative in reciprocal space

    cell_he = pyscf.M(atom='He 0 0 0',
                      basis=[[0, ( 1, 1, .1), (.5, .1, 1)],
                             [1, (.8, 1)],
                             [2, (.6, 1)]],
                      unit='B',
                      precision = 1e-10,
                      a=np.eye(3)*5)

def tearDownModule():
    global cell_orth, cell_nonorth, cell_he
    del cell_orth, cell_nonorth, cell_he

def _get_vpplocG_derivatives(cell, mesh, rhoG):
    assert cell.dimension == 3
    Gv_bases = multigrid._get_Gv_bases(mesh, cell.reciprocal_vectors())
    coords = cp.asarray(cell.atom_coords())
    SIx = cp.exp(-1j * coords.dot(Gv_bases[0]))
    SIy = cp.exp(-1j * coords.dot(Gv_bases[1]))
    SIz = cp.exp(-1j * coords.dot(Gv_bases[2]))

    ngrids = np.prod(mesh)
    Gx, Gy, Gz = Gv_bases
    GvT = Gx[:,:,None,None] + Gy[:,None,:,None] + Gz[:,None,None,:]
    GvT = GvT.reshape(3, ngrids)
    G2 = cp.einsum('xg,xg->g', GvT, GvT)
    coulG = 4 * np.pi / G2
    coulG[0] = 0
    xyG = cp.einsum('xg,yg->xyg', GvT, GvT)

    dSI_prefactor = -1j * GvT * rhoG.conj()

    charges = cell.atom_charges()

    grad = cp.zeros((cell.natm, 3))
    vlocG_0 = cp.zeros(ngrids, dtype=np.complex128)
    vlocG_1 = cp.zeros((3, 3, ngrids), dtype=np.complex128)

    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        pp = cell._pseudo[symb]
        rloc, nexp, cexp = pp[1:3+1]

        SI = (SIx[ia,:,None,None] * SIy[ia,:,None] * SIz[ia]).ravel()
        x = G2 * rloc**2
        expx = cp.exp(-0.5*x)
        SI *= expx
        Z = charges[ia]

        coef1 = -Z * coulG * SI * (2/G2 + rloc**2)
        coef1[0] = 0

        cfacs = 0
        dcfacs = 0
        if nexp >= 1:
            cfacs += cexp[0]
        if nexp >= 2:
            cfacs += cexp[1] * (3 - x)
            dcfacs -= cexp[1]
        if nexp >= 3:
            cfacs += cexp[2] * (15 - 10*x + x*x)
            dcfacs += cexp[2] * (-10 + 2*x)
        if nexp >= 4:
            cfacs += cexp[3] * (105 - 105*x + 21*x*x - x*x*x)
            dcfacs += cexp[3] * (-105 + 42*x - 3*x*x)

        coef2 = (
            (2*np.pi)**1.5
            * rloc**5
            * SI
            * (cfacs - 2 * dcfacs)
        )

        v = -Z * coulG * SI
        v += (2*np.pi)**(3/2.)*rloc**3 * cfacs * SI
        v[0] += 2*np.pi*Z*rloc**2

        grad[ia, :] = (dSI_prefactor @ v).real

        vlocG_0 += v
        vlocG_1 += (coef1 + coef2) * xyG

    return grad/cell.vol, vlocG_0, vlocG_1

def eval_nucG_SI_gradient(cell, mesh, rho_g):
    from gpu4pyscf.pbc.gto.cell import get_Gv_weights
    from gpu4pyscf.pbc.tools import get_coulG
    ngrids = np.prod(mesh)
    assert rho_g.shape == (ngrids,)

    assert cell.dimension == 3
    Gv, (basex, basey, basez) = get_Gv_weights(cell, mesh)[:2]
    b = cell.reciprocal_vectors()
    coords = cell.atom_coords()
    rb = cp.asarray(coords.dot(b.T))
    SIx = cp.exp(-1j*rb[:,0,None] * basex)
    SIy = cp.exp(-1j*rb[:,1,None] * basey)
    SIz = cp.exp(-1j*rb[:,2,None] * basez)
    dSI_prefactor = -1j * Gv.T * rho_g.conj()
    charges = -cell.atom_charges()
    coulG = get_coulG(cell, Gv=Gv)

    ZSI = 0
    de = cp.empty([cell.natm, 3], dtype = cp.complex128)

    for i_atom in range(cell.natm):
        SI = (SIx[i_atom,:,None,None] * SIy[i_atom,:,None] * SIz[i_atom]).ravel()
        de[i_atom, :] = charges[i_atom] * (dSI_prefactor @ (coulG * SI))
        ZSI -= charges[i_atom] * SI

    coulGxy = cp.einsum('gx,gy->xyg', Gv, Gv)
    G2 = cp.einsum('gx,gx->g', Gv, Gv)
    G2[0] = np.inf
    coulGxy *= coulG
    coulG_1 = coulGxy * -2/G2
    sigma = cp.einsum('xyg,g,g->xy', coulG_1, rho_g.conj(), ZSI).real

    de = de.real
    de /= cell.vol
    sigma /= cell.vol
    return de, sigma

class KnownValues(unittest.TestCase):
    def test_get_pp(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_pp()
        out = multigrid.MultiGridNumInt(cell_orth).get_pp().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_nuc_nonorth(self):
        ref = MultiGridNumInt_cpu(cell_nonorth).get_nuc()
        out = multigrid.MultiGridNumInt(cell_nonorth).get_nuc().get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)

    def test_get_nuc_kpts(self):
        ref = MultiGridNumInt_cpu(cell_orth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_orth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, delta=1e-8)

    def test_get_nuc_kpts_nonorth(self):
        ref = MultiGridNumInt_cpu(cell_nonorth).get_nuc(kpts)
        out = multigrid.MultiGridNumInt(cell_nonorth).get_nuc(kpts).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)

    def test_get_rho(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        ref = multigrid_cpu.multigrid.get_rho(MultiGridNumInt_cpu(cell_orth), dm)
        out = multigrid.MultiGridNumInt(cell_orth).get_rho(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_j(self):
        nao = cell_orth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        ref = MultiGridNumInt_cpu(cell_orth).get_jk(dm[None], with_k=False)[0]
        out = multigrid.MultiGridNumInt(cell_orth).get_j(dm, hermi=0).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_j_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        dm = np.random.random((nao,nao)) - .5
        ref = MultiGridNumInt_cpu(cell_nonorth).get_jk(dm[None], with_k=False)[0]
        out = multigrid.MultiGridNumInt(cell_nonorth).get_j(dm, hermi=0).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 8)

    def test_get_vxc_lda(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-7
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-7

        xc = 'lda,'
        dm = cp.array([dm[None]] * 2)
        mf = pcell.RKS(xc=xc).to_gpu()
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max().get() < 1e-7
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max().get() < 1e-7

    def test_get_vxc_lda_kpts(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'lda,'
        kmesh = [3, 2, 1]
        kpts = cell_orth.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = np.random.random((2,nkpts,nao,nao)) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('sLpq,Lk->skpq', dm, phase.conj())
        dm = np.einsum('sLpr,sLqr->sLpq', dm, dm.conj())
        dm = cp.asarray(dm)

        pcell = cell_orth.copy()
        pcell.precision = 1e-10

        mf = pcell.KUKS(xc=xc).to_gpu()
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]

        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max() < 5e-8
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max() < 5e-8

    def test_get_vxc_gga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

        xc = 'pbe,'
        dm = cp.array([dm[None]] * 2)
        mf = pcell.RKS(xc=xc).to_gpu()
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8

    def test_get_vxc_gga_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'pbe,'
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10
        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(MultiGridNumInt_cpu(pcell), xc, dm, with_j=True)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(pcell, None, xc, dm)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(cell_nonorth, None, xc, dm, with_j=True)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    def test_get_vxc_gga_kpts(self):
        nao = cell_orth.nao
        np.random.seed(20)
        xc = 'pbe,'
        kmesh = [3, 2, 1]
        kpts = cell_orth.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = np.random.random((2,nkpts,nao,nao)) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('sLpq,Lk->skpq', dm, phase.conj())
        dm = np.einsum('sLpr,sLqr->sLpq', dm, dm.conj())

        pcell = cell_orth.copy()
        pcell.precision = 1e-10
        mf = pcell.KRKS(xc=xc)
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_orth).nr_uks(
            cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-7

    def test_get_vxc_gga_kpts_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'pbe,'
        kmesh = [3, 2, 1]
        kpts = cell_nonorth.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = np.random.random((nkpts,nao,nao)) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('Lpr,Lqr->Lpq', dm, dm.conj())

        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10

        if hasattr(multigrid_cpu, 'nr_rks'):
            n0, exc0, ref = multigrid_cpu.nr_rks(
                MultiGridNumInt_cpu(pcell), xc, dm, with_j=True, kpts=kpts)
        else:
            n0, exc0, ref = MultiGridNumInt_cpu(pcell).nr_rks(
                pcell, None, xc, dm, kpts=kpts)
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(
            cell_nonorth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc.get()).max() < 1e-8

    def test_get_vxc_mgga(self):
        nao = cell_orth.nao
        np.random.seed(2)
        xc = 'r2scan'
        dm = np.random.random((nao,nao)) - .5
        dm = cp.asarray(dm.dot(dm.T))
        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        mf = pcell.RKS(xc=xc).to_gpu()

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm[None])
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc).max().get() < 1e-7
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-7
        assert abs(ref-vxc).max().get() < 1e-7

        dm = cp.array([dm[None], dm[None]])
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm)
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj[0] + vj[1]
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8

    def test_get_vxc_mgga_kpts(self):
        nao = cell_orth.nao
        np.random.seed(3)
        xc = 'r2scan'
        kmesh = [3, 2, 1]
        kpts = cell_orth.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = np.random.random((nkpts,nao,nao)) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('Lpr,Lqr->Lpq', dm, dm.conj())

        pcell = cell_orth.copy()
        pcell.precision = 1e-11
        mf = pcell.KRKS(xc=xc).to_gpu()

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm, kpts=kpts)
        # vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        # ref += vj
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=False, kpts=kpts)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_rks(cell_orth, None, xc, dm, with_j=False, kpts=kpts)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8

        dm = cp.array([dm, dm])
        n0, exc0, ref = mf._numint.nr_uks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj[0] + vj[1]
        ni = multigrid.MultiGridNumInt(cell_orth)
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8
        ni.enable_aft = False
        n1, exc1, vxc = ni.nr_uks(cell_orth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1).max() < 1e-8
        assert abs(exc0-exc1).max() < 1e-8
        assert abs(ref-vxc).max().get() < 1e-8

    def test_get_vxc_mgga_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(2)
        xc = 'r2scan'
        dm = np.random.random((nao,nao)) - .5
        dm = cp.array(dm.dot(dm.T))
        pcell = cell_nonorth.copy()
        pcell.precision = 1e-10
        mf = pcell.RKS(xc=xc).to_gpu()

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm[None])
        vj = mf.with_df.get_jk(dm, with_k=False)[0]
        ref += vj
        n1, exc1, vxc = multigrid.MultiGridNumInt(cell_nonorth).nr_rks(cell_nonorth, None, xc, dm, with_j=True)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max() < 5e-8

    def test_get_vxc_mgga_kpts_nonorth1(self):
        cell = pyscf.M(
            a = '''
    3.86599305 0.         0.        
    1.93299652 3.34804819 0.        
    1.93299652 1.11601606 3.15657011
            ''',
            atom='''
    Si 0.         0.         0.        
    Si 1.93299652 1.11601606 0.78914253''',
        pseudo = 'gth-pade',
        basis={
            'Si': ('gth-dzv', [[3, [1.1, 1.]], [4, [.8, 1.]]]),
        })
        nao = cell.nao
        cp.random.seed(1)
        dm = cp.random.rand(nao, nao) * .5 - .2
        dm = dm.dot(dm.T)

        xc = 'r2scan,'
        mf = cell.KRKS(xc=xc).to_gpu()
        n0, exc0, ref = mf._numint.nr_rks(cell, mf.grids, xc, dm[None])

        ni = multigrid.MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        n1, exc1, dat = ni.nr_rks(cell, None, xc, dm, with_j=False)
        assert abs(n0 - n1) < 3e-7
        assert abs(exc0 - exc1) < 3e-7
        assert abs(ref - dat).max() < 3e-7

    def test_get_vxc_mgga_kpts_nonorth(self):
        nao = cell_nonorth.nao
        np.random.seed(4)
        xc = 'r2scan'
        kmesh = [3, 2, 1]
        kpts = cell_nonorth.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = np.random.random((nkpts,nao,nao)) - .5
        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm = np.einsum('Lpq,Lk->kpq', dm, phase.conj())
        dm = np.einsum('Lpr,Lqr->Lpq', dm, dm.conj())

        pcell = cell_nonorth.copy()
        mf = pcell.KRKS(xc=xc).to_gpu()

        n0, exc0, ref = mf._numint.nr_rks(pcell, mf.grids, xc, dm, kpts=kpts)
        vj = mf.with_df.get_jk(dm, kpts=kpts, with_k=False)[0]
        ref += vj
        ni = multigrid.MultiGridNumInt(cell_nonorth)
        ni.allow_mesh_reduction = False
        n1, exc1, vxc = ni.nr_rks(cell_nonorth, None, xc, dm, with_j=True, kpts=kpts)
        assert abs(n0-n1) < 1e-8
        assert abs(exc0-exc1) < 1e-8
        assert abs(ref-vxc).max() < 5e-8

    @pytest.mark.slow
    def test_rks_lda(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='svwn').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf._numint.allow_mesh_reduction = False
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.777337612, 8)

    @pytest.mark.slow
    def test_rks_gga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='pbe').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf._numint.allow_mesh_reduction = False
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.87059063524272, 8)

    @pytest.mark.slow
    def test_rks_mgga(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''C     0.      0.      0.
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            basis = 'gth-dzv',
            pseudo = 'gth-pbe',
            precision = 1e-9,
        )
        mf = cell.RKS(xc='scan').to_gpu()
        mf._numint = multigrid.MultiGridNumInt(cell)
        mf._numint.allow_mesh_reduction = False
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -44.7542917283246, 8)

    def test_compact_basis_functions(self):
        cell = gto.M(
            a = np.diag([4., 8., 7.]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = [[0, [2e4, 1.]], [0, [1e2, 1.]], [0, [2., 1.]],
                     [1, [2e2, 1.]], [1, [1., 1.]]],
            mesh = [7, 7, 7],
        )
        np.random.seed(2)
        nao = cell.nao
        dm = np.random.random((nao,nao)) - .5
        dm = dm.dot(dm.T)
        ref = cell.RKS().get_rho(dm)
        ni = multigrid.MultiGridNumInt(cell)
        ni.enable_aft = False
        out = ni.get_rho(dm).get()
        self.assertAlmostEqual(abs(ref-out).max(), 0, 7)

    def test_band_rks_gamma(self):
        cell = gto.M(
            verbose = 0,
            a = np.diag([3.6, 3.2, 4.5]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        np.random.seed(1)
        kpts_band = np.random.random((4,3))

        test_mf = cell.RKS(xc='r2scan').to_gpu()
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.RKS(xc='r2scan')
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-7
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_band_krks_kpts(self):
        cell = gto.M(
            verbose = 0,
            a = np.array([[3.6, 0, 0], [0, 3.2, 0.2], [0, 0, 4.5]]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        kpts = cell.make_kpts([1,3,1])

        np.random.seed(1)
        kpts_band = np.random.random((1,3)) # Yes, one non-zero k point, as an edge case

        test_mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.KRKS(xc='pbe', kpts=kpts)
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-7
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_band_kuks_kpts(self):
        cell = gto.M(
            verbose = 0,
            a = np.diag([3.6, 3.2, 4.5]),
            atom = '''C     0.      0.      0.
                      C     1.8     1.8     1.8   ''',
            basis = """
                C DZVP-GTH-no-d-one-p-no-first-exp
                  1
                  2  0  1  3  2  1
                        1.2881838513  -0.0292640031   0.0000000000  -0.2775560300
                        0.4037767149  -0.6882040510   0.0000000000  -0.4712295093
                        0.1187877657  -0.3964426906   1.0000000000  -0.4058039291
                    """,
            pseudo = 'gth-pade',
            precision = 1e-8,
        )

        kpts = cell.make_kpts([1,1,3])

        np.random.seed(1)
        kpts_band = np.random.random((2,3))


        test_mf = cell.KUKS(xc='lda', kpts=kpts).to_gpu()
        test_mf._numint = multigrid.MultiGridNumInt(cell)
        test_mf.conv_tol = 1e-10
        test_mf.kernel()
        test_band_e, test_band_c = test_mf.get_bands(kpts_band)

        ref_mf = cell.KUKS(xc='lda', kpts=kpts)
        ref_mf.mo_coeff = test_mf.mo_coeff.get()
        ref_mf.mo_energy = test_mf.mo_energy.get()
        ref_mf.mo_occ = test_mf.mo_occ.get()
        ref_band_e, ref_band_c = ref_mf.get_bands(kpts_band)
        assert abs(test_band_e.get() - ref_band_e).max() < 1e-6
        assert abs(abs(test_band_c.get()) - abs(np.array(ref_band_c))).max() < 1e-3

    def test_nr_rks_fxc(self):
        cell = cell_he
        np.random.seed(9)
        nao = cell.nao
        dm_he = np.random.rand(nao, nao)
        dm_he = dm_he + dm_he.T
        dm_he = dm_he * .2 + np.eye(nao)
        dm_he = cp.asarray(dm_he)

        dm1 = np.random.rand(2,1,nao,nao)
        dm1 = dm1 + dm1.transpose(0,1,3,2)
        dm1 = cp.asarray(dm1)
        grids = UniformGrids(cell)

        ni = numint.NumInt()
        mg = multigrid.MultiGridNumInt(cell)
        mg.allow_mesh_reduction = False

        xc = 'lda,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'b88,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'r2scan,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        dm_he = np.random.rand(len(kpts), nao, nao) * .2 + np.eye(nao)

        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm_he = np.einsum('Lpq,Lk->kpq', dm_he, phase.conj())
        dm_he = dm_he + dm_he.transpose(0,2,1).conj()
        dm_he = cp.asarray(dm_he)

        dm1 = np.random.rand(2,len(kpts),nao,nao)
        dm1 = np.einsum('nLpq,Lk->nkpq', dm1, phase.conj())
        dm1 = dm1 + dm1.transpose(0,1,3,2).conj()
        dm1 = cp.asarray(dm1)

        ni = numint.KNumInt()

        xc = 'lda,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'b88,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'r2scan,'
        ref = ni.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_rks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

    def test_nr_uks_fxc(self):
        cell = cell_he
        np.random.seed(9)
        nao = cell.nao
        dm_he = np.random.rand(2, nao, nao)
        dm_he = dm_he + dm_he.transpose(0,2,1)
        dm_he = dm_he * .2 + np.eye(nao)
        dm_he = cp.asarray(dm_he)

        dm1 = np.random.rand(2,3,1,nao,nao)
        dm1 = dm1 + dm1.transpose(0,1,2,4,3)
        dm1 = cp.asarray(dm1)
        grids = UniformGrids(cell)

        ni = numint.NumInt()
        mg = multigrid.MultiGridNumInt(cell)

        xc = 'lda,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'b88,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'r2scan,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        kmesh = [3,1,1]
        kpts = cell.make_kpts(kmesh)
        dm_he = np.random.rand(2, len(kpts), nao, nao) * .2 + np.eye(nao)

        phase = fft_matrix(kmesh).get() / np.prod(kmesh)
        dm_he = np.einsum('nLpq,Lk->nkpq', dm_he, phase.conj())
        dm_he = dm_he + dm_he.transpose(0,1,3,2).conj()
        dm_he = cp.asarray(dm_he)

        dm1 = np.random.rand(2,3, len(kpts),nao,nao)
        dm1 = np.einsum('snLpq,Lk->snkpq', dm1, phase.conj())
        dm1 = dm1 + dm1.transpose(0,1,2,4,3).conj()
        dm1 = cp.asarray(dm1)

        ni = numint.KNumInt()

        xc = 'lda,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'b88,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

        xc = 'r2scan,'
        ref = ni.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        v = mg.nr_uks_fxc(cell, grids, xc, dm_he, dm1, hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(v-ref).max().get(), 0, 10)

    def test_shell_splitting_for_large_fock_in_imagediff_space_gamma(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8817  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 0,
            output = '/dev/null',
        )

        kpts = cell.make_kpts([1,1,1])
        mf = KRKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**28)):
            mf = mf.multigrid_numint()
            mf._numint.allow_mesh_reduction = False

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -44.93180128532909
        ref_gradient = np.array([
            [ 2.87614262e-03,  1.33298682e-03,  1.33298682e-03],
            [-8.42690061e-03,  1.01735391e-05,  1.01735391e-05],
            [ 2.82851274e-03,  1.28135252e-03, -1.28134877e-03],
            [-1.00471252e-04, -8.51551903e-06,  8.51502092e-06],
            [ 2.82851274e-03, -1.28134877e-03,  1.28135252e-03],
            [-1.00471252e-04,  8.51502093e-06, -8.51551903e-06],
            [ 2.87618947e-03, -1.33322853e-03, -1.33322853e-03],
            [-2.78640919e-03, -8.51738633e-06, -8.51738633e-06],
        ])

        assert abs(test_energy - ref_energy) < 3e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_shell_splitting_for_large_fock_in_imagediff_space_k(self):
        cell = gto.M(
            a = np.eye(3)*3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8817  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 5,
            output = '/dev/null',
        )

        kpts = cell.make_kpts([1,1,3])
        mf = KRKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**28)):
            mf = mf.multigrid_numint()
            mf._numint.allow_mesh_reduction = False

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -45.30199423477792
        ref_gradient = np.array([
            [ 2.36897360e-03,  1.19228739e-03,  6.67961844e-04],
            [-9.11593616e-03,  1.03369618e-05,  8.42224689e-06],
            [ 2.32829121e-03,  1.14450844e-03, -6.20435413e-04],
            [ 2.38310045e-04, -8.52269883e-06,  6.81929906e-06],
            [ 2.32828725e-03, -1.14450700e-03,  6.20436720e-04],
            [ 1.14957254e-03,  8.52837460e-06, -6.81790444e-06],
            [ 2.36729361e-03, -1.19142895e-03, -6.64432085e-04],
            [-1.66792993e-03, -8.52961493e-06, -6.81679641e-06],
        ])

        assert abs(test_energy - ref_energy) < 3e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_shell_splitting_for_large_fock_in_imagediff_space_unrestricted(self):
        cell = gto.M(
            a = '''
                0.000000000, 3.370137329, 3.370137329
                3.370137329, 0.000000000, 3.370137329
                3.370137329, 3.370137329, 0.000000000
            ''',
            atom = '''
                C 0 0 0
                C 1.675068664391 1.685068664391 1.685068664391
            ''',
            basis = "gth-dzvp",
            pseudo = 'gth-pbe',
            precision = 1e-8,
            verbose = 5,
            output = '/dev/null',
        )

        kpts = cell.make_kpts([1,1,3])
        mf = KUKS_gpu(cell, xc = 'pbe', kpts = kpts)
        mf.conv_tol = 1e-10

        # mf = mf.multigrid_numint()
        # assert type(mf._numint) is multigrid.MultiGridNumInt

        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_gradient = mf.Gradients().kernel()
        # print(repr(ref_energy))
        # print(repr(ref_gradient))

        with lib.temporary_env(multigrid, get_avail_mem=(lambda **kw: 2**25)):
            mf = mf.multigrid_numint()
            mf._numint.allow_mesh_reduction = False

            test_energy = mf.kernel()
            assert mf.converged
            test_gradient = mf.Gradients().kernel()

        ref_energy = -10.82283467913058
        ref_gradient = np.array([
            [-0.00776978, -0.00816341,  0.00816341],
            [ 0.00777063,  0.00816369, -0.00816369],
        ])

        assert abs(test_energy - ref_energy) < 3e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_eval_vpplocG(self):
        from gpu4pyscf.pbc.dft.multigrid import eval_vpplocG
        np.random.seed(8)
        cell = pyscf.M(
            atom='C 0 0 0; O 1.2 1.7 .1; C .2 .3 .7; O 1.4 0.5 1.8',
            basis=[[0, [0.4, 1]]],
            pseudo={'C': [[2, 2], 0.38, 4, [-8.8, 1.33, 0.85, 0.55]],
                    'O': [[2, 2], 0.8, 3, [1., 1.5, 0.3]]},
            a=np.eye(3) * 2.5 + np.random.rand(3,3)*.5)
        mesh = [11]*3
        Gv = cell.get_Gv(mesh)
        SI = cell.get_SI(Gv)
        ref = -np.einsum('ij,ij->j', pseudo.get_vlocG(cell, Gv), SI)
        dat = eval_vpplocG(cell, mesh)
        self.assertAlmostEqual(abs(ref - dat.get()).max(), 0, 12)

    def test_pploc_derivatives(self):
        from gpu4pyscf.pbc.dft.multigrid_v3 import _pploc_derivatives, _get_Gv_bases
        np.random.seed(8)
        cell = pyscf.M(
            atom='C 0 0 0; O 1.2 1.7 .1; C .2 .3 .7; O 1.4 0.5 1.8',
            basis=[[0, [0.4, 1]]],
            pseudo={'C': [[2, 2], 0.38, 4, [-8.8, 1.33, 0.85, 0.55]],
                    'O': [[2, 2], 0.8, 3, [1., 1.5, 0.3]]},
            a=np.eye(3) * 2.5 + np.random.rand(3,3)*.5)
        mesh = [11]*3
        ngrids = np.prod(mesh)

        rho = cp.array(np.random.rand(*mesh))
        rhoG = cp.fft.ifft(rho).ravel()

        grad_ref, vlocG0, vlocG1 = _get_vpplocG_derivatives(cell, mesh, rhoG)
        sigma_ref = cp.einsum('g,xyg->xy', rhoG.conj(), vlocG1).real / cell.vol

        Gv_bases = _get_Gv_bases(mesh, cell.reciprocal_vectors())
        grad, sigma = _pploc_derivatives(cell, rhoG, Gv_bases)
        assert abs(grad_ref - grad).max().get() < 1e-12
        assert abs(sigma_ref - sigma).max().get() < 1e-12

    def test_ne_derivatives(self):
        from gpu4pyscf.pbc.dft.multigrid_v3 import _ne_derivatives, _get_Gv_bases
        np.random.seed(8)
        cell = pyscf.M(
            atom='C 1.0 1.0 0; C .2 .3 .7',
            basis=[[0, [0.4, 1]]],
            a=np.eye(3) * 2.5 + np.random.rand(3,3)*.5)
        mesh = [11]*3
        ngrids = np.prod(mesh)

        rho = cp.array(np.random.rand(*mesh))
        rhoG = cp.fft.ifft(rho).ravel()

        grad_ref, sigma_ref = eval_nucG_SI_gradient(cell, mesh, rhoG)

        Gv_bases = _get_Gv_bases(mesh, cell.reciprocal_vectors())
        grad, sigma = _ne_derivatives(cell, rhoG, Gv_bases)
        assert abs(grad_ref - grad).max().get() < 1e-12
        assert abs(sigma_ref - sigma).max().get() < 1e-12

if __name__ == '__main__':
    print("Full Tests for multigrid v3")
    unittest.main()
