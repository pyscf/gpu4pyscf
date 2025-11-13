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
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import aft as aft_cpu, aft_jk as aft_jk_cpu
from pyscf.pbc.df import fft as fft_cpu
from gpu4pyscf.pbc.df import aft, aft_jk
from gpu4pyscf.pbc.df import fft
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.grad import rks_stress
from gpu4pyscf.pbc.grad import krks_stress
from gpu4pyscf.lib.multi_gpu import num_devices
from packaging import version

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
        ref = aft_cpu.AFTDF(cell).get_pp(kpts=kpts[0])
        v1 = aft.AFTDF(cell).get_pp(kpts=kpts[0]).get()
        assert abs(v1 - ref).max() < 1e-9

        kpts4 = cell.make_kpts([4,1,1])
        ref = aft_cpu.AFTDF(cell).get_pp(kpts=kpts4)
        v1 = aft.AFTDF(cell).get_pp(kpts=kpts4).get()
        assert abs(v1 - ref).max() < 1e-9

    def test_aft_get_nuc(self):
        ref = aft_cpu.AFTDF(cell).get_nuc(kpts=kpts[0])
        v1 = aft.AFTDF(cell).get_nuc(kpts=kpts[0]).get()
        assert abs(v1 - ref).max() < 1e-9

        kpts4 = cell.make_kpts([4,1,1])
        ref = aft_cpu.AFTDF(cell).get_nuc(kpts=kpts4)
        v1 = aft.AFTDF(cell).get_nuc(kpts=kpts4).get()
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
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        jref = mydf0.get_jk(dm, kpts=kpts, with_k=False)[0]
        vj = mydf.get_jk(dm, kpts=kpts, with_k=False)[0]
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
        mesh = [13]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        kref = mydf0.get_jk(dm, hermi=0, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=0, kpts=kpts, with_j=False)[1]
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
        mesh = [13]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        dm = np.random.random((nkpts,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k2(self):
        kpts = cell.make_kpts([2,1,1])
        nkpts = len(kpts)
        mesh = [13]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.random.rand(nkpts, nao, nao)
        dm = dm + dm.transpose(0,2,1)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k3(self):
        kpts = cell.make_kpts([6,1,1])
        nkpts = len(kpts)
        mesh = [13]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)
        mydf0.k_conj_symmetry = False
        mydf.k_conj_symmetry = False

        nao = cell.nao
        np.random.seed(12)
        nocc = 2
        mo = (np.random.random((nkpts,nao,nocc)) +
              np.random.random((nkpts,nao,nocc))*1j)
        mo_occ = np.ones((nkpts,nocc))
        dm = np.random.rand(nkpts, nao, nao)
        dm = dm + dm.transpose(0,2,1)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_ej_ip1_gamma_point(self):
        cell = pgto.M(
            atom = '''
            O   0.000    0.    0.1174
            H   1.757    0.    0.4696
            H   0.757    0.    0.4696
            C   1.      1.    0.
            H   4.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        np.random.seed(9)
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) * .5
        dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
        mydf = aft.AFTDF(cell)
        ej = aft_jk.get_ej_ip1(mydf, dm)
        assert abs(ej.sum(axis=0)).max() < 1e-8

        cell.precision = 1e-10
        cell.build(0, 0)
        dm = dm[0] + dm[1]
        vj = fft_cpu.FFTDF(cell).get_j_e1(dm)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xpq,qp->x', vj[:,p0:p1], dm[:,p0:p1])
        assert abs(ej - ref).max() < 1e-8

    def test_ej_ip1_kpts(self):
        cell = pgto.M(
            atom = '''
            O   0.000    0.    0.1174
            H   1.757    0.    0.4696
            H   0.757    0.    0.4696
            C   1.      1.    0.
            H   0.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        kpts = cell.make_kpts([3,2,1])
        dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
        mydf = aft.AFTDF(cell)
        ej = aft_jk.get_ej_ip1(mydf, dm, kpts=kpts)
        assert abs(ej.sum(axis=0)).max() < 1e-8

        cell.precision = 1e-10
        cell.build(0, 0)
        vj = fft_cpu.FFTDF(cell).get_j_e1(dm, kpts=kpts)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xkpq,kqp->x', vj[:,:,p0:p1], dm[:,:,p0:p1]).real
        ref /= len(kpts)
        assert abs(ej - ref).max() < 1e-8

    def test_ek_ip1_gamma_point(self):
        cell = pgto.M(
            atom = '''
            O   0.000    0.    0.1174
            H   1.757    0.    0.4696
            H   0.757    0.    0.4696
            C   1.      1.    0.
            H   4.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        np.random.seed(9)
        nao = cell.nao
        dm = np.random.rand(2, nao, nao) * .5
        dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
        myaft = aft.AFTDF(cell)
        ek = aft_jk.get_ek_ip1(myaft, dm)
        assert abs(ek.sum(axis=0)).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            ek_ewald = aft_jk.get_ek_ip1(myaft, dm, exxdiv='ewald')
            assert abs(ek_ewald.sum(axis=0)).max() < 1e-8

        cell.precision = 1e-10
        cell.build(0, 0)
        myfft = fft_cpu.FFTDF(cell)
        vk = myfft.get_k_e1(dm)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xnpq,nqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1])
        assert abs(ek - ref).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            vk = myfft.get_k_e1(dm, exxdiv='ewald')
            for i in range(cell.natm):
                p0, p1 = aoslices[i, 2:]
                ref[i] = np.einsum('xnpq,nqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1])
            assert abs(ek_ewald - ref).max() < 1e-8

    @unittest.skipIf(num_devices > 1, '')
    def test_ek_ip1_kpts(self):
        cell = pgto.M(
            atom = '''
            O   0.000    0.    0.1174
            H   1.757    0.    0.4696
            H   0.757    0.    0.4696
            C   1.      1.    0.
            H   0.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        kpts = cell.make_kpts([3,2,1])
        dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
        myaft = aft.AFTDF(cell)
        ek = aft_jk.get_ek_ip1(myaft, dm, kpts=kpts)
        assert abs(ek.sum(axis=0)).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            ek_ewald = aft_jk.get_ek_ip1(myaft, dm, kpts=kpts, exxdiv='ewald')
            assert abs(ek_ewald.sum(axis=0)).max() < 1e-8

        cell.precision = 1e-10
        cell.build(0, 0)
        myfft = fft_cpu.FFTDF(cell)
        vk = myfft.get_k_e1(dm, kpts=kpts)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xkpq,kqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1]).real
        ref /= len(kpts)
        assert abs(ek - ref).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            vk = myfft.get_k_e1(dm, kpts=kpts, exxdiv='ewald')
            for i in range(cell.natm):
                p0, p1 = aoslices[i, 2:]
                ref[i] = np.einsum('xkpq,kqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1]).real
            ref /= len(kpts)
            assert abs(ek_ewald - ref).max() < 1e-8

    def test_ej_strain_deriv_gamma_point(self):
        cell = pgto.M(
            atom = '''
            C   1.      1.    0.
            H   4.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        np.random.seed(9)
        nao = cell.nao
        dm = np.random.rand(nao, nao) * .5
        dm = dm.dot(dm.T)
        mydf = aft.AFTDF(cell)
        sigma = aft_jk.get_ej_strain_deriv(mydf, dm)

        xc = 'lda,'
        mf_grad = cell.RKS(xc=xc).to_gpu().Gradients()
        ref = rks_stress.get_vxc(mf_grad, cell, dm, with_j=True, with_nuc=False)
        ref -= rks_stress.get_vxc(mf_grad, cell, dm, with_j=False, with_nuc=False)
        assert abs(ref - sigma).max() < 1e-8

    def test_ej_strain_deriv_kpts(self):
        cell = pgto.M(
            atom = '''
            C   1.      1.    0.
            H   4.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        kmesh = [3,2,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = cp.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
        mydf = aft.AFTDF(cell)
        sigma = aft_jk.get_ej_strain_deriv(mydf, dm, kpts)

        xc = 'lda,'
        mf_grad = cell.KRKS(xc=xc, kpts=kpts).to_gpu().Gradients()
        ref = krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=True, with_nuc=False)
        ref -= krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=False, with_nuc=False)
        assert abs(ref - sigma).max() < 1e-8

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = rks_stress._finite_diff_cells(cell, i, j, disp=1e-4)
            mydf = aft.AFTDF(cell1, kpts=cell1.make_kpts(kmesh))
            vj = aft_jk.get_j_kpts(mydf, dm, hermi=1, kpts=mydf.kpts)
            e1 = .5 * cp.einsum('kij,kji->', vj, dm).real / nkpts
            mydf = aft.AFTDF(cell2, kpts=cell2.make_kpts(kmesh))
            vj = aft_jk.get_j_kpts(mydf, dm, hermi=1, kpts=mydf.kpts)
            e2 = .5 * cp.einsum('kij,kji->', vj, dm).real / nkpts
            assert abs(sigma[i,j] - (e1-e2)/2e-4) < 2e-7

    def test_ek_strain_deriv_gamma_point(self):
        cell = pgto.M(
            atom = '''
            C   1.      1.    0.
            H   4.      0.5   3.
            H   0.5     1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        np.random.seed(9)
        nao = cell.nao
        dm = np.random.rand(nao, nao) * .5
        dm = cp.array(dm.dot(dm.T))
        mydf = aft.AFTDF(cell)
        sigma = aft_jk.get_ek_strain_deriv(mydf, dm)

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = rks_stress._finite_diff_cells(cell, i, j, disp=1e-4)
            mydf = aft.AFTDF(cell1)
            vk = aft_jk.get_jk(mydf, dm, hermi=1, with_j=False, exxdiv=None)[1]
            e1 = .5 * cp.einsum('ij,ji->', vk, dm).real
            mydf = aft.AFTDF(cell2)
            vk = aft_jk.get_jk(mydf, dm, hermi=1, with_j=False, exxdiv=None)[1]
            e2 = .5 * cp.einsum('ij,ji->', vk, dm).real
            assert abs(sigma[i, j] - (e1-e2)/2e-4).max() < 2e-7

    def test_ek_strain_deriv_kpts(self):
        cell = pgto.M(
            atom = '''
            C   1.      1.    0.
            H   4.      0.5   3.
            H   0.5     1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        kmesh = [1,3,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)
        dm = cp.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
        mydf = aft.AFTDF(cell)
        sigma = aft_jk.get_ek_strain_deriv(mydf, dm, kpts, exxdiv='ewald')

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = rks_stress._finite_diff_cells(cell, i, j, disp=1e-4)
            mydf = aft.AFTDF(cell1, kpts=cell1.make_kpts(kmesh))
            vk = aft_jk.get_k_kpts(mydf, dm, hermi=1, kpts=mydf.kpts, exxdiv='ewald')
            e1 = .5 * cp.einsum('kij,kji->', vk, dm).real / nkpts
            mydf = aft.AFTDF(cell2, kpts=cell2.make_kpts(kmesh))
            vk = aft_jk.get_k_kpts(mydf, dm, hermi=1, kpts=mydf.kpts, exxdiv='ewald')
            e2 = .5 * cp.einsum('kij,kji->', vk, dm).real / nkpts
            assert abs(sigma[i, j] - (e1-e2)/2e-4).max() < 5e-7

if __name__ == '__main__':
    print("Full Tests for aft")
    unittest.main()
