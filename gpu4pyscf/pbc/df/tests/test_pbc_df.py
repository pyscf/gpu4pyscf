# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
import tempfile
import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import df as df_cpu
from gpu4pyscf.pbc.df.df import GDF

def setUpModule():
    global cell
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (1., 1)), (1, (.4, 1))],
                  'C' :[[0, [1., 1]]],}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = np.eye(3) * 2.5
    cell.precision = 1e-8
    cell.build()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
    def test_get_pp(self):
        #kpt = cell.make_kpts([9,6,5])[107]
        #ref = df_cpu.GDF(cell, kpt).get_pp()
        #v1 = GDF(cell, kpt).get_pp().get()
        #assert abs(v1 - ref).max() < 1e-8

        ref = df_cpu.GDF(cell).get_pp()
        v1 = GDF(cell).get_pp().get()
        assert abs(v1 - ref).max() < 1e-8

        kpts4 = cell.make_kpts([4,1,1])
        ref = df_cpu.GDF(cell, kpts4).get_pp()
        v1 = GDF(cell, kpts4).get_pp().get()
        assert abs(v1 - ref).max() < 1e-8

    def test_get_nuc(self):
        L = 5.
        n = 11
        cell1 = pgto.Cell()
        cell1.a = np.eye(3) * L
        cell1.mesh = [n] * 3
        cell1.atom = '''He    3.    2.       3.
                       He    1.    1.       1.'''
        cell1.basis = 'ccpvdz'
        cell1.precision=1e-8
        cell1.verbose = 0
        cell1.max_memory = 1000
        cell1.build(0,0)
        ref = df_cpu.GDF(cell1).get_nuc()
        v1 = GDF(cell1).get_nuc().get()
        assert abs(v1 - ref).max() < 1e-8

        kpts4 = cell1.make_kpts([4,1,1])
        ref = df_cpu.GDF(cell1, kpts4).get_nuc()
        v1 = GDF(cell1, kpts4).get_nuc().get()
        assert abs(v1 - ref).max() < 1e-8

    def test_jk(self):
        mydf0 = df_cpu.GDF(cell)
        mydf  = GDF(cell)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nao,nao))
        jref, kref = mydf0.get_jk(dm, hermi=0, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=0, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

        dm = dm + np.random.random((nao,nao)) * 1j
        dm = dm + dm.conj().T
        jref, kref = mydf0.get_jk(dm, hermi=1, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

    def test_jk_gamma_point(self):
        mydf0 = df_cpu.GDF(cell)
        mydf  = GDF(cell)
        mydf.is_gamma_point = True
        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        jref, kref = mydf0.get_jk(dm, hermi=1, exxdiv='ewald')
        assert abs(vj - jref).max() < 1e-8
        assert abs(vk - kref).max() < 1e-8

    def test_jk1(self):
        kpts = cell.make_kpts([1,6,1])
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts)
        mydf  = GDF(cell, kpts)

        nao = cell.nao
        np.random.seed(12)
        dm = (np.random.random((nkpts, nao, nao)) +
              np.random.random((nkpts, nao, nao))*1j)
        jref, kref = mydf0.get_jk(dm, hermi=0, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=0, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

        dm = dm + dm.conj().transpose(0,2,1)
        jref, kref = mydf0.get_jk(dm, hermi=1, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

    @unittest.skip('pbc-gdf only supports Monkhorst-Pack k-mesh')
    def test_jk_complex_dm(self):
        scaled_center = [0.3728,0.5524,0.7672]
        kpt = cell.make_kpts([1,1,1], scaled_center=scaled_center)[0]
        mydf0 = df_cpu.GDF(cell, kpts=[kpt])
        mydf  = GDF(cell, kpts=[kpt])

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nao,nao)) + np.random.random((nao,nao)) * 1j
        jref, kref = mydf0.get_jk(dm, hermi=0, kpts=kpt, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=0, kpts=kpt, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

        dm = dm + dm.conj().T
        jref, kref = mydf0.get_jk(dm, hermi=1, kpts=kpt, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, kpts=kpt, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-8
        assert abs(vk.get() - kref).max() < 1e-8

    @unittest.skip('pbc-gdf only supports Monkhorst-Pack k-mesh')
    def test_get_j(self):
        kpts = np.random.random((4,3))
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts=kpts)
        mydf  = GDF(cell, kpts=kpts)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        jref = mydf0.get_jk(dm, with_k=False)[0]
        vj = mydf.get_jk(dm, with_k=False)[0]
        assert abs(vj.get() - jref).max() < 1e-8

    @unittest.skip('pbc-gdf only supports Monkhorst-Pack k-mesh')
    def test_get_k(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts=kpts)
        mydf  = GDF(cell, kpts=kpts)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        kref = mydf0.get_jk(dm, hermi=0, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=0, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-8

    @unittest.skip('pbc-gdf only supports Monkhorst-Pack k-mesh')
    def test_get_k1(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts=kpts)
        mydf  = GDF(cell, kpts=kpts)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-8

    def test_get_k2(self):
        kpts = cell.make_kpts([3,1,1])
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts=kpts)
        mydf  = GDF(cell, kpts=kpts)

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.einsum('kpi,kqi->kpq', mo, mo.conj())
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)

        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-8

    def test_get_k3(self):
        kpts = cell.make_kpts([6,1,1])
        nkpts = len(kpts)
        mydf0 = df_cpu.GDF(cell, kpts=kpts)
        mydf  = GDF(cell, kpts=kpts)
        #mydf.k_conj_symmetry = False

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.einsum('kpi,kqi->kpq', mo, mo.conj())
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)

        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-8

if __name__ == '__main__':
    print("Full Tests for PBC DF")
    unittest.main()
