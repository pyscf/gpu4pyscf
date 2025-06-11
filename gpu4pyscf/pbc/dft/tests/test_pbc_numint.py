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
        np.random.seed(1)
        kpts = np.random.random((4,3))
        ni = numint.KNumInt()
        ao = ni.eval_ao(cell, grids.coords, kpts)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get(), kpts)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.301832342768873-0.2417141694175898j, 8)

        ao = ni.eval_ao(cell, grids.coords, kpts, deriv=1)
        ref = ni.to_cpu().eval_ao(cell, grids.coords.get(), kpts, deriv=1)
        self.assertAlmostEqual(abs(ao[0].get()-ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[1].get()-ref[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[2].get()-ref[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(ao[3].get()-ref[3]).max(), 0, 9)
        self.assertAlmostEqual(lib.fp(ao[0].get()), -1.9473900325707074-0.5644459560348523j, 8)

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

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()
