# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import fft as fft_cpu
from pyscf.pbc import dft as dft_cpu
from gpu4pyscf.pbc.dft import gen_grid
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.lib.cupy_helper import contract


def setUpModule():
    global cell, grids
    cell = pbcgto.Cell()
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.a = np.eye(3) * 2.5
    cell.mesh = [15]*3
    cell.atom = [['C', (1., .8, 1.9)],
                 ['C', (.1, .2,  .3)],]
    cell.basis = 'ccpvdz'
    cell.precision = 1e-11
    cell.build(False, False)
    grids = gen_grid.UniformGrids(cell).build()

    cell_he = pbcgto.M(atom='He 0 0 0',
                    basis=[[0, ( 1, 1, .1), (.5, .1, 1)],
                           [1, (.8, 1)]],
                    unit='B',
                    precision = 1e-9,
                    a=np.eye(3)*5)

    kpts = np.random.random((2,3))
    kpts[1] = -kpts[0]
    nao = cell_orth.nao_nr()
    dm = np.random.random((len(kpts),nao,nao)) * .2
    dm1 = dm + np.eye(nao)
    dm = dm1 + dm1.transpose(0,2,1)

    he_nao = cell_he.nao
    dm_he = np.random.random((len(kpts), he_nao, he_nao))
    dm_he = dm_he + dm_he.transpose(0,2,1)
    dm_he = dm_he * .2 + np.eye(he_nao)

def tearDownModule():
    global cell, grids
    cell.stdout.close()
    del cell, grids


class KnownValues(unittest.TestCase):
    def test_eval_ao(self):
        ni = numint.NumInt()
        ao = numint.eval_ao(cell, grids.coords)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get())
        self.assertAlmostEqual(abs(ao.get()-ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao.get()), -1.0151790074469322, 8)

        ao = numint.eval_ao(cell, grids.coords, deriv=1)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get(), deriv=1)
        self.assertAlmostEqual(abs(ao.get()-ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao.get()), -1.6507836971790972, 8)

    # issue 675
    def test_eval_ao1(self):
        cell = pbcgto.M(a=np.eye(3)*3., atom = 'He 0.0 0.0 0.0', basis = '''
S
1.607   0.6400
0.569   0.2900
0.076   0.0007
''')
        grids = cell.get_uniform_grids()
        dat = numint.eval_ao(cell, grids)

        c = cell.bas_ctr_coeff(0)
        cell1 = pbcgto.M(a=np.eye(3)*3., atom = 'He 0.0 0.0 0.0', basis = '''
S
1.607   1.
S
0.569   1.
S
0.076   1.
''')
        ref = cell1.pbc_eval_gto('GTOval', grids)
        ref = np.einsum('pi,gp->gi', c, ref)
        self.assertAlmostEqual(abs(dat.get() - ref).max(), 0, 8)

    def test_eval_ao_kpt(self):
        np.random.seed(1)
        kpt = np.random.random(3)
        ni = numint.NumInt()
        ao = numint.eval_ao(cell, grids.coords, kpt)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get(), kpt)
        self.assertAlmostEqual(abs(ao.get()-ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao.get()), -1.301832342768873-0.2417141694175898j, 8)

        ao = numint.eval_ao(cell, grids.coords, kpt, deriv=1)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get(), kpt, deriv=1)
        self.assertAlmostEqual(abs(ao.get()-ref).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao.get()), -1.9473900325707074-0.5644459560348523j, 8)

    def test_eval_ao_kpts(self):
        cell_ref = cell.copy()
        cell_ref.precision = 1e-16
        weight = cell.vol
        np.random.seed(1)
        kpts = np.random.random((4,3))
        k411 = cell.make_kpts([4,1,1])
        ni = numint.KNumInt()
        ao = ni.eval_ao(cell, grids.coords, kpts)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.301832342768873-0.2417141694175898j, 8)

        ao = ni.eval_ao(cell, grids.coords, kpts, deriv=1)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts, deriv=1)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.9473900325707074-0.5644459560348523j, 8)

        ao = ni.eval_ao(cell, grids.coords, kpts, deriv=2)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts, deriv=2)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -255.2319247513107-0.29712941019664596j, 8)

        ao = ni.eval_ao(cell, grids.coords, k411)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.0151790074499552, 8)

        ao = ni.eval_ao(cell, grids.coords, k411, deriv=1)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411, deriv=1)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.6507836970726455, 8)

        ao = ni.eval_ao(cell, grids.coords, k411, deriv=2)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411, deriv=2)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -254.41150917759416, 8)

        pcell = cell.copy()
        pcell.cart = True
        cell_ref.cart = True
        ao = ni.eval_ao(pcell, grids.coords, kpts)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

        ao = ni.eval_ao(pcell, grids.coords, kpts, deriv=1)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts, deriv=1)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

        ao = ni.eval_ao(pcell, grids.coords, kpts, deriv=2)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), kpts, deriv=2)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

        ao = ni.eval_ao(pcell, grids.coords, k411)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

        ao = ni.eval_ao(pcell, grids.coords, k411, deriv=1)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411, deriv=1)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

        ao = ni.eval_ao(pcell, grids.coords, k411, deriv=2)
        ref = ni.to_cpu().eval_ao(cell_ref, grids.coords.get(), k411, deriv=2)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max()*weight, 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max()*weight, 0, 9)

    def test_nr_rks(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((2,3))
        nao = cell.nao
        dms = cp.random.random((nao,nao)) - .5
        dms = dms.dot(dms.T)
        ni = numint.NumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'lda', dms, hermi=1, kpts=kpts[0])
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'lda', dms.get(), hermi=1, kpt=kpts[0])
        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_nr_uks(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((2,3))
        nao = cell.nao
        dms = cp.random.random((2,nao,nao)) - .5
        dms = contract('npi,nqi->npq', dms, dms)
        ni = numint.NumInt()
        ne, exc, vmat = ni.nr_uks(cell, grids, 'lda', dms, hermi=1, kpts=kpts[0])
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'lda', dms.get(), hermi=1, kpt=kpts[0])
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_knumint_nr_rks(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((2,3))
        nao = cell.nao
        dms = cp.random.random((2,nao,nao)) - .5
        dms = contract('kpi,kqi->kpq', dms, dms)
        ni = numint.KNumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'm06', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'm06', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ne, exc, vmat = ni.nr_rks(cell, grids, 'blyp', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'blyp', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ne, exc, vmat = ni.nr_rks(cell, grids, 'lda', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'lda', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_knumint_nr_uks(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((2,3))
        nao = cell.nao
        dms = cp.random.random((2,2,nao,nao)) - .5
        dms = contract('nkpi,nkqi->nkpq', dms, dms)
        ni = numint.KNumInt()
        ne, exc, vmat = ni.nr_uks(cell, grids, 'm06', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'm06', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(exc, ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ne, exc, vmat = ni.nr_uks(cell, grids, 'blyp', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'blyp', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(exc, ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ne, exc, vmat = ni.nr_uks(cell, grids, 'lda', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'lda', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(exc, ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_eval_rho(self):
        cp.random.seed(10)
        nao = cell.nao
        dm = cp.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        ao = numint.eval_ao(cell, grids.coords, deriv=1)
        rho = numint.eval_rho(cell, ao, dm, xctype='MGGA')
        ao_cpu = ao.get()
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='MGGA', with_lapl=False)
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

        ao = numint.eval_ao(cell, grids.coords, deriv=1)
        rho = numint.eval_rho(cell, ao, dm, xctype='GGA')
        ao_cpu = ao.get()
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='GGA')
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

        ao = numint.eval_ao(cell, grids.coords, deriv=0)
        rho = numint.eval_rho(cell, ao, dm, xctype='LDA')
        ao_cpu = ao.get()
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='LDA')
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

    def test_knumint_eval_rho(self):
        cp.random.seed(10)
        nao = cell.nao
        dm = cp.random.random((nao,nao))
        dm = dm + dm.T
        ni = numint.NumInt()
        kpts = cell.make_kpts([2,1,1])
        dm = cp.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        ni = numint.KNumInt()
        ao = ni.eval_ao(cell, grids.coords, kpts=kpts, deriv=1)
        rho = ni.eval_rho(cell, ao, dm, xctype='MGGA')
        ao_cpu = [x.get() for x in ao]
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='MGGA', with_lapl=False)
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

        ao = ni.eval_ao(cell, grids.coords, kpts=kpts, deriv=1)
        rho = ni.eval_rho(cell, ao, dm, xctype='GGA')
        ao_cpu = [x.get() for x in ao]
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='GGA')
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

        ao = ni.eval_ao(cell, grids.coords, kpts=kpts, deriv=0)
        rho = ni.eval_rho(cell, ao, dm, xctype='LDA')
        ao_cpu = [x.get() for x in ao]
        ref = ni.to_cpu().eval_rho(cell, ao_cpu, dm.get(), xctype='LDA')
        self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 12)

    def test_uniform_grid_division_mode(self):
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = [20, 20, 20]

        assert not grids.lowmem_mode

        grids.max_grid_mesh_block = [3, 10, 7]
        assert grids.lowmem_mode

        ref_coords = grids.coords.get()
        ref_weight = grids.weights[0]

        test_coords = []
        for frag_grids in grids.loop_grids():
            test_coords.append(frag_grids.coords)
            assert np.max(np.abs(frag_grids.weights - ref_weight)) < 1e-14
        test_coords = cp.vstack(test_coords).get()

        ref_coords = ref_coords[np.lexsort((ref_coords[:, 2], ref_coords[:, 1], ref_coords[:, 0])), :]
        test_coords = test_coords[np.lexsort((test_coords[:, 2], test_coords[:, 1], test_coords[:, 0])), :]

        assert np.max(np.abs(ref_coords - test_coords)) < 1e-15

    def test_nr_rks_grid_division_mode(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((2,3))
        nao = cell.nao
        dms = cp.random.random((2,nao,nao)) - .5
        dms = contract('kpi,kqi->kpq', dms, dms)

        grids = gen_grid.UniformGrids(cell).build()
        grids.mesh = [20, 20, 20]
        grids.max_grid_mesh_block = [3, 10, 7]
        assert grids.lowmem_mode

        ni = numint.NumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'pbe', dms[0], hermi=1, kpts=kpts[0])
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'pbe', dms[0].get(), hermi=1, kpt=kpts[0])

        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ni = numint.KNumInt()
        ne, exc, vmat = ni.nr_rks(cell, grids, 'm06', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_rks(cell, grids.to_cpu(), 'm06', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(float(ne), ref[0], 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_nr_uks_division_mode(self):
        np.random.seed(1)
        cp.random.seed(1)
        kpts = np.random.random((3,3))
        nao = cell.nao
        dms = cp.random.random((2,3,nao,nao)) - .5
        dms = contract('nkpi,nkqi->nkpq', dms, dms)

        grids = gen_grid.UniformGrids(cell).build()
        grids.mesh = [20, 20, 20]
        grids.max_grid_mesh_block = [3, 100, 7]
        assert grids.lowmem_mode

        ni = numint.NumInt()
        ne, exc, vmat = ni.nr_uks(cell, grids, 'lda', dms[:,0], hermi=1, kpts=kpts[0])
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'lda', dms[:,0].get(), hermi=1, kpt=kpts[0])
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(float(exc), ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

        ni = numint.KNumInt()
        ne, exc, vmat = ni.nr_uks(cell, grids, 'm06', dms, hermi=1, kpts=kpts)
        ref = ni.to_cpu().nr_uks(cell, grids.to_cpu(), 'm06', dms.get(), hermi=1, kpts=kpts)
        self.assertAlmostEqual(abs(ne.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(exc, ref[1], 9)
        self.assertAlmostEqual(abs(vmat.get() - ref[2]).max(), 0, 9)

    def test_cache_xc_kernel(self):
        np.random.seed(9)
        dm0 = np.random.random(dm_he.shape) + np.random.random(dm_he.shape)*1j
        dm1 = dm1 + dm1.transpose(0,2,1)
        grids_cpu = dft_cpu.UniformGrids(cell_he)
        grids_cpu.mesh = [11]*3
        ni_cpu = dft_cpu.numint.NumInt()
        kni_cpu = dft_cpu.numint.KNumInt()

        xc = 'lda,'
        ref = ni_cpu.cache_xc_kernel1(cell_he, grids_cpu, xc, dm_he[0])
        ref = np.einsum('g,xyg->', ref[0], ref[2])

        grids = gen_grid.UniformGrids(cell_he)
        grids.mesh = [11]*3
        ni = numint.NumInt()
        kni = numint.KNumInt()
        dat = ni.cache_xc_kernel1(cell_he, grids, xc, cp.array(dm_he[0]), kpts=np.zeros(3))
        dat = cp.einsum('xg,xyg->', dat[0], dat[2])
        self.assertAlmostEqual(dat.get(), ref)
        dat = kni.cache_xc_kernel1(cell_he, grids, xc, cp.array(dm_he[:1]))
        dat = cp.einsum('xg,xyg->', dat[0], dat[2])
        self.assertAlmostEqual(dat.get(), ref)

        ref = ni_cpu.cache_xc_kernel1(cell_he, grids_cpu, xc, dm_he, spin=1)
        ref = np.einsum('ag,axbyg->', ref[0], ref[2])
        dat = ni.cache_xc_kernel1(cell_he, grids, xc, cp.array(dm_he), spin=1)
        dat = cp.einsum('axg,axbyg->', dat[0], dat[2])
        self.assertAlmostEqual(dat.get(), ref)

        xc = 'm06,'
        ref = kni_cpu.cache_xc_kernel1(cell_he, grids_cpu, xc, dm_he, kpts=kpts, spin=1)
        ref = np.einsum('axg,axbyg->', ref[0], ref[2])
        dat = kni.cache_xc_kernel1(cell_he, grids, xc, cp.array(dm_he), kpts=kpts, spin=1)
        dat = cp.einsum('axg,axbyg->', dat[0], dat[2])
        self.assertAlmostEqual(dat.get(), ref)

        xc = 'pbe,'
        ref = kni_cpu.cache_xc_kernel1(cell_he, grids_cpu, xc, np.array([dm_he]*2), kpts=kpts, spin=1)
        ref = np.einsum('axg,axbyg->', ref[0], ref[2])
        dat = kni.cache_xc_kernel1(cell_he, grids, xc, cp.array([dm_he]*2), kpts=kpts, spin=1)
        dat = cp.einsum('axg,axbyg->', dat[0], dat[2])
        self.assertAlmostEqual(dat.get(), ref)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    #unittest.main()
    setUpModule()
    #np.random.seed(2)
    #cell_orth = pbcgto.M(
    #    verbose = 7,
    #    output = '/dev/null',
    #    a = np.eye(3)*3.5668,
    #    atom = '''C     0.      0.      0.
    #              C     1.8     1.8     1.8   ''',
    #    basis = 'gth-dzv',
    #    pseudo = 'gth-pade',
    #    precision = 1e-9,
    #    mesh = [48] * 3,
    #)
    #cell_nonorth = pbcgto.M(
    #    a = np.eye(3)*3.5668 + np.random.random((3,3)),
    #    atom = '''C     0.      0.      0.
    #              C     0.8917  0.8917  0.8917''',
    #    basis = 'gth-dzv',
    #    pseudo = 'gth-pade',
    #    precision = 1e-9,
    #    mesh = [44,43,42],
    #)

    def test_gen_rhf_response(self):
        np.random.seed(9)
        nkpts = len(kpts)
        nao = cell_he.nao
        mo = np.random.rand(nkpts, nao, 4)
        mo_occ = np.ones((nkpts, 4))
        dm0 = np.einsum('kpi,kqi->kpq', mo, mo)

        dm1 = np.random.random(dm0.shape)
        dm1 = dm1 + dm1.transpose(0,2,1)
        dm1[1] = dm1[0]
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()

        mf = dft.KRKS(cell_he)
        mf._numint = multigrid.MultiGridNumInt(cell_he)
        mf.kpts = kpts

        mf.xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                    hermi=1, kpts=kpts)
        vj = mydf.get_jk(dm1, with_k=False, kpts=kpts)[0]
        ref += vj
        v = mf.gen_response(mo, mo_occ, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        mf.xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                    hermi=1, kpts=kpts)
        ref += vj
        v = mf.gen_response(mo, mo_occ, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    def test_nr_rks_fxc(self):
        np.random.seed(9)
        dm1 = np.random.random(dm_he.shape) + np.random.random(dm_he.shape)*1j
        dm1 = dm1 + dm1.transpose(0,2,1)
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()
        mg_df = multigrid.MultiGridNumInt(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1,
                                   hermi=1)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1,
                                    hermi=1)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    def test_nr_rks_fxc_hermi0(self):
        np.random.seed(9)
        dm1 = np.random.random(dm_he.shape) + np.random.random(dm_he.shape)*1j
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()
        mg_df = multigrid.MultiGridNumInt(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1, hermi=0)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=0)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_rks_fxc(ni, cell_he, mydf.grids, xc, dm_he[0], dm1, hermi=0)
        v = multigrid.nr_rks_fxc(mg_df, xc, dm_he[0], dm1, hermi=0)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 6)

    def test_nr_rks_fxc_st(self):
        np.random.seed(9)
        nkpts = len(kpts)
        nao = cell_he.nao
        mo = np.random.rand(nkpts, nao, 4)
        mo_occ = np.ones((nkpts, 4))
        dm0 = np.einsum('kpi,kqi->kpq', mo, mo)

        dm1 = np.random.rand(3,nkpts,nao,nao) + np.random.rand(3,nkpts,nao,nao)*1j
        dm1 = dm1 + dm1.transpose(0,1,3,2).conj()
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mf = dft.KRKS(cell_he)
        mf._numint = multigrid.MultiGridNumInt(cell_he)
        mf.kpts = kpts

        mf.xc = 'lda,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                       singlet=True, kpts=kpts)
        v = multigrid.nr_rks_fxc_st(mf._numint, mf.xc, dm0, dm1, singlet=True, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        mf.xc = 'b88,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                       singlet=True, kpts=kpts) * .5
        ref += mf.with_df.get_jk(dm1, hermi=1, kpts=kpts, with_k=False)[0]
        v = mf.gen_response(mo, mo_occ, singlet=True, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        mf.xc = 'lda,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                       singlet=False, kpts=kpts) * .5
        v = mf.gen_response(mo, mo_occ, singlet=False, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        mf.xc = 'b88,'
        ref = dft.numint.nr_rks_fxc_st(ni, cell_he, mydf.grids, mf.xc, dm0, dm1,
                                       singlet=False, kpts=kpts) * .5
        v = mf.gen_response(mo, mo_occ, singlet=False, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

    def test_gen_uhf_response(self):
        np.random.seed(9)
        nkpts = len(kpts)
        nao = cell_he.nao
        mo = np.random.rand(2, nao, 4)
        mo_occ = np.ones((2, 4))
        dm0 = np.einsum('spi,sqi->spq', mo, mo)

        dm1 = np.random.random(dm0.shape)
        dm1 = dm1 + dm1.transpose(0,2,1)
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.NumInt()

        mf = dft.UKS(cell_he)
        mf._numint = multigrid.MultiGridNumInt(cell_he)

        mf.xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, mf.xc, dm0, dm1, hermi=1)
        vj = mydf.get_jk(dm1, with_k=False)[0]
        ref += vj[0] + vj[1]
        v = mf.gen_response(mo, mo_occ, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 7)

        mf.xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, mf.xc, dm0, dm1, hermi=1)
        ref += vj[0] + vj[1]
        v = mf.gen_response(mo, mo_occ, hermi=1)(dm1)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 7)

    # FIXME: is the discrepancy due to problems in precision or threshold estimation?
    def test_nr_uks_fxc(self):
        np.random.seed(9)
        nkpts = len(kpts)
        nao = cell_he.nao
        dm1 = np.random.rand(3,nkpts,nao,nao) + np.random.rand(3,nkpts,nao,nao)*1j
        dm1 = dm1 + dm1.transpose(0,1,3,2).conj()
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mg_df = multigrid.MultiGridNumInt(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

        xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=1, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)

    def test_orth_uks_fxc_hermi0(self):
        np.random.seed(9)
        dm1 = np.random.random(dm_he.shape) + np.random.random(dm_he.shape)*1j
        mydf = df.FFTDF(cell_he)
        ni = dft.numint.KNumInt()
        mg_df = multigrid.MultiGridNumInt(cell_he)

        xc = 'lda,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 9)

        xc = 'b88,'
        ref = dft.numint.nr_uks_fxc(ni, cell_he, mydf.grids, xc,
                                    (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        v = multigrid.nr_uks_fxc(mg_df, xc, (dm_he, dm_he), (dm1, dm1), hermi=0, kpts=kpts)
        self.assertEqual(ref.dtype, v.dtype)
        self.assertEqual(ref.shape, v.shape)
        self.assertAlmostEqual(abs(v-ref).max(), 0, 8)
