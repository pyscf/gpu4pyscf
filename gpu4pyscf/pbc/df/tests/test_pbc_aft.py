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
import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import aft as aft_cpu, aft_jk as aft_jk_cpu
from gpu4pyscf.pbc.df import aft, aft_jk
from gpu4pyscf.lib.cupy_helper import tag_array


def setUpModule():
    global cell, cell1, kpts
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (1., 1)), (1, (.4, 1))],
                  'C' :[[0, [1., 1]]],}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = np.eye(3) * 2.5
    cell.precision = 1e-8
    cell.build()
    kpts = cell.make_kpts([13,1,1])[4:8]

    cell1 = pgto.Cell()
    cell1.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell1.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))]}
    cell1.a = np.eye(3) * 2.5
    cell1.mesh = [21] * 3
    cell1.build()

def tearDownModule():
    global cell, cell1, kpts
    del cell, cell1, kpts

class KnownValues(unittest.TestCase):
    def test_aft_get_pp(self):
        ref = aft_cpu.AFTDF(cell, kpts[0]).get_pp()
        v1 = aft.AFTDF(cell, kpts[0]).get_pp().get()
        assert abs(v1 - ref).max() < 1e-9

        kpts4 = cell.make_kpts([4,1,1])
        ref = aft_cpu.AFTDF(cell, kpts4).get_pp()
        v1 = aft.AFTDF(cell, kpts4).get_pp().get()
        assert abs(v1 - ref).max() < 1e-9

    def test_aft_get_nuc(self):
        ref = aft_cpu.AFTDF(cell, kpts[0]).get_nuc()
        v1 = aft.AFTDF(cell, kpts[0]).get_nuc().get()
        assert abs(v1 - ref).max() < 1e-9

        kpts4 = cell.make_kpts([4,1,1])
        ref = aft_cpu.AFTDF(cell, kpts4).get_nuc()
        v1 = aft.AFTDF(cell, kpts4).get_nuc().get()
        assert abs(v1 - ref).max() < 1e-9

    def test_jk(self):
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nao,nao))
        jref, kref = mydf0.get_jk(dm, hermi=0, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=0, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-9
        assert abs(vk.get() - kref).max() < 1e-9

        dm = dm + np.random.random((nao,nao)) * 1j
        dm = dm + dm.conj().T
        jref, kref = mydf0.get_jk(dm, hermi=1, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-9
        assert abs(vk.get() - kref).max() < 1e-9

    def test_jk_complex_dm(self):
        scaled_center = [0.3728,0.5524,0.7672]
        kpt = cell.make_kpts([1,1,1], scaled_center=scaled_center)[0]
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=[kpt]).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=[kpt]).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nao,nao)) + np.random.random((nao,nao)) * 1j
        jref, kref = mydf0.get_jk(dm, hermi=0, kpts=kpt, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=0, kpts=kpt, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-9
        assert abs(vk.get() - kref).max() < 1e-9

        dm = dm + dm.conj().T
        jref, kref = mydf0.get_jk(dm, hermi=1, kpts=kpt, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, kpts=kpt, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 1e-9
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_j(self):
        kpts = np.random.random((4,3))
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=kpts).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        jref = mydf0.get_jk(dm, with_k=False)[0]
        vj = mydf.get_jk(dm, with_k=False)[0]
        assert abs(vj.get() - jref).max() < 1e-9

    def test_aft_k(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=kpts).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        kref = mydf0.get_jk(dm, hermi=0, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=0, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k1(self):
        kpts = cell.get_abs_kpts([[-.25,-.25,-.25],
                                  [-.25,-.25, .25],
                                  [-.25, .25,-.25],
                                  [-.25, .25, .25],
                                  [ .25,-.25,-.25],
                                  [ .25,-.25, .25],
                                  [ .25, .25,-.25],
                                  [ .25, .25, .25]])
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=kpts).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k2(self):
        kpts = cell.make_kpts([2,1,1])
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=kpts).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.random.rand(nkpts, nao, nao)
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)

        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k3(self):
        kpts = cell.make_kpts([6,1,1])
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf  = aft.AFTDF(cell, kpts=kpts).set(mesh=mesh)
        mydf0.k_conj_symmetry = False
        mydf.k_conj_symmetry = False

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.random.rand(nkpts, nao, nao)
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)

        kref = mydf0.get_jk(dm, hermi=1, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

if __name__ == '__main__':
    print("Full Tests for aft")
    unittest.main()
