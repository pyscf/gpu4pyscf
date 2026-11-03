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
from packaging.version import Version
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import aft as aft_cpu, aft_jk as aft_jk_cpu
from pyscf.pbc.df import fft as fft_cpu
from gpu4pyscf.pbc.df import aft, aft_jk
from gpu4pyscf.pbc.df import fft
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
from gpu4pyscf.pbc.dft.multigrid_v3 import MultiGridNumInt
from gpu4pyscf.lib.multi_gpu import num_devices
from packaging import version

def setUpModule():
    global cell, cell1, kpts
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (1., 1)), (1, (.4, 1)), (1, (.3, 1))],
                  'C' :[[0, [1., 1]], [2, [.3, 1]]],}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = np.eye(3) * 2.5
    cell.precision = 1e-8
    cell.build()
    kpts = cell.make_kpts([13,1,1])[4:8]

    cell1 = pgto.Cell()
    cell1.atom = 'He 1. .5 .5; He .1 1.3 2.1'
    cell1.basis = {'He': [(0, (2.5, 1)), (0, (1., 1)), (2, (.5, 1))]}
    cell1.a = np.eye(3) * 2.5
    cell1.mesh = [21] * 3
    cell1.build()

def tearDownModule():
    global cell, cell1, kpts
    del cell, cell1, kpts

class AFTProtocols(unittest.TestCase):
    def setUp(self):
        self.cell = pgto.M(
            atom='C .2 .3 .4; H 1.4 1.2 1.6', unit='Bohr', spin=1,
            a=np.diag([5., 5.5, 6.]), mesh=[15, 17, 19],
            basis={'C': [[0, [.9, 1.]], [1, [.5, 1.]]],
                   'H': [[0, [.6, 1.]]]}, pseudo='gth-pade',
            precision=1e-9, verbose=0)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.9'),
                         'PySCF 2.8: different AFT shape, reset, and range-separated mesh conventions')
    def test_gamma_matrices_and_cpu_conversion(self):
        cell = self.cell
        mydf = aft.AFTDF(cell)
        mydf.mesh = np.array([13, 15, 17])
        mydf.max_memory = 777
        mydf.time_reversal_symmetry = False
        converted = mydf.to_cpu()
        self.assertIsInstance(converted, aft_cpu.AFTDF)
        self.assertIs(converted.cell, cell)
        self.assertEqual(converted.max_memory, mydf.max_memory)
        self.assertFalse(converted.time_reversal_symmetry)
        np.testing.assert_array_equal(converted.mesh, mydf.mesh)
        np.testing.assert_array_equal(converted.kpts, mydf.kpts)
        reference = aft_cpu.AFTDF(cell).set(mesh=mydf.mesh.copy())
        for name in ('get_nuc', 'get_pp'):
            with self.subTest(operator=name):
                actual = getattr(mydf, name)()
                self.assertIsInstance(actual, cp.ndarray)
                self.assertEqual(actual.shape, (1, cell.nao, cell.nao))
                self.assertEqual(actual.dtype, np.float64)
                np.testing.assert_allclose(actual.get(), getattr(reference, name)(),
                                           atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual.get(), getattr(converted, name)(),
                                           atol=1e-9, rtol=0)
                single = getattr(mydf, name)(kpts=np.zeros(3))
                np.testing.assert_allclose(single.get(), actual[0].get(),
                                           atol=1e-12, rtol=0)
                np.testing.assert_allclose(single.get(), single.T.get(),
                                           atol=1e-11, rtol=0)
        dm = np.linalg.inv(cell.pbc_intor('int1e_ovlp'))
        actual, exchange = mydf.get_jk(cp.asarray(dm), with_k=False)
        self.assertIsNone(exchange)
        for cpu_df in (reference, converted):
            expected = cpu_df.get_jk(dm, with_k=False)[0]
            np.testing.assert_allclose(actual.get(), expected, atol=1e-9, rtol=0)

    def test_reset_scaled_kpts_and_range_cache(self):
        cell = self.cell
        scaled = np.array([[.13, -.07, .09], [-.13, .07, -.09]])
        mydf = aft.AFTDF(cell, cell.get_abs_kpts(scaled))
        mydf.mesh = np.array([13, 15, 17])
        original_mesh = mydf.mesh.copy()
        original_kpts = mydf.kpts.copy()
        with mydf.range_coulomb(.4) as cached:
            self.assertIsNot(cached, mydf)
            self.assertEqual(cached.cell.omega, .4)
        self.assertEqual(cell.omega, 0)
        with mydf.range_coulomb(.4) as reused:
            self.assertIs(reused, cached)
        self.assertEqual(len(mydf._rsh_df), 1)
        moved, _ = _finite_diff_cells(cell, 0, 0, disp=.04)
        moved.mesh = np.array([17, 19, 21])
        self.assertIs(mydf.reset(moved), mydf)
        self.assertIs(mydf.cell, moved)
        self.assertEqual(mydf._rsh_df, {})
        np.testing.assert_allclose(moved.get_scaled_kpts(mydf.kpts), scaled,
                                   atol=1e-14, rtol=0)
        np.testing.assert_allclose(mydf.kpts, moved.get_abs_kpts(scaled),
                                   atol=1e-14, rtol=0)
        self.assertGreater(np.max(abs(mydf.kpts-original_kpts)), 1e-4)
        # reset preserves the explicitly configured quadrature, as on CPU.
        np.testing.assert_array_equal(mydf.mesh, original_mesh)
        with mydf.range_coulomb(.4) as rebuilt:
            self.assertIsNot(rebuilt, cached)
            self.assertIs(rebuilt.cell, moved)
        self.assertEqual(moved.omega, 0)
        self.assertIs(mydf.reset(), mydf)
        self.assertEqual(mydf._rsh_df, {})
        self.assertIs(mydf.cell, moved)
        np.testing.assert_array_equal(mydf.mesh, original_mesh)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.9'),
                         'PySCF 2.8: different AFT shape, reset, and range-separated mesh conventions')
    def test_symmetry_kpoints_reset(self):
        cell = self.cell
        points = cell.make_kpts([3, 1, 1], time_reversal_symmetry=True)
        cpu_points = cell.make_kpts([3, 1, 1], time_reversal_symmetry=True)
        self.assertIsInstance(points, KPoints)
        self.assertLess(points.nkpts_ibz, points.nkpts)
        mydf = aft.AFTDF(cell, points)
        reference = aft_cpu.AFTDF(cell, cpu_points)
        self.assertIs(mydf.kpts, points)
        scaled = points.kpts_scaled.copy()
        original_mesh = np.array(mydf.mesh)
        moved, _ = _finite_diff_cells(cell, 0, 0, disp=.04)
        moved.mesh = np.array([17, 19, 21])
        self.assertIs(mydf.reset(moved), mydf)
        reference.reset(moved)
        self.assertIs(mydf.kpts, points)
        self.assertIs(points.cell, moved)
        np.testing.assert_allclose(points.kpts_scaled, scaled, atol=1e-14, rtol=0)
        np.testing.assert_allclose(points.kpts, moved.get_abs_kpts(scaled),
                                   atol=1e-14, rtol=0)
        np.testing.assert_allclose(points.kpts, reference.kpts, atol=1e-14, rtol=0)
        np.testing.assert_allclose(points.weights_ibz, cpu_points.weights_ibz,
                                   atol=1e-14, rtol=0)
        self.assertLess(points.nkpts_ibz, points.nkpts)
        np.testing.assert_array_equal(mydf.mesh, original_mesh)

    def _check_long_range_mesh_and_weighted_kernels(self, scaled):
        cell = self.cell
        cell.omega = .4
        original_mesh = np.array(cell.mesh)
        mydf = aft.AFTDF(cell)
        reference = aft_cpu.AFTDF(cell)
        expected_mesh = cell.cutoff_to_mesh(
            aft_cpu.estimate_ke_cutoff_for_omega(cell, cell.omega))
        np.testing.assert_array_equal(mydf.mesh, expected_mesh)
        np.testing.assert_array_equal(mydf.mesh, reference.mesh)
        np.testing.assert_array_equal(cell.mesh, original_mesh)
        Gv, _, weight = cell.get_Gv_weights(mydf.mesh)
        kpt = cell.get_abs_kpts(scaled)
        for omega, lr, sr in ((0, .37, .37), (None, .37, .37),
                              (None, .65, 0)):
            with self.subTest(kpt=scaled, omega=omega, lr=lr, sr=sr):
                actual = mydf.weighted_coulG(
                    kpt=kpt, omega=omega, lr_factor=lr, sr_factor=sr)
                expected = pbc_tools.get_coulG(
                    cell, k=kpt, mesh=mydf.mesh, Gv=Gv,
                    omega=cell.omega if omega is None else omega)
                np.testing.assert_allclose(actual.get(), expected * weight * lr,
                                           atol=1e-13, rtol=1e-12)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.9'),
                         'PySCF 2.8: different AFT shape, reset, and range-separated mesh conventions')
    def test_long_range_mesh_and_weighted_kernels_gamma(self):
        self._check_long_range_mesh_and_weighted_kernels(np.zeros(3))

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.9'),
                         'PySCF 2.8: different AFT shape, reset, and range-separated mesh conventions')
    def test_long_range_mesh_and_weighted_kernels_nonzero_kpoint(self):
        self._check_long_range_mesh_and_weighted_kernels(np.array([.13, -.07, .09]))



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
        assert abs(vj.get() - jref).max() < 3e-9
        assert abs(vk.get() - kref).max() < 3e-9

        dm = dm + np.random.random((nao,nao)) * 1j
        dm = dm + dm.conj().T
        jref, kref = mydf0.get_jk(dm, hermi=1, exxdiv='ewald')
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        assert abs(vj.get() - jref).max() < 3e-9
        assert abs(vk.get() - kref).max() < 3e-9

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
        #kpts = np.random.random((4,3))
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
        dm = dm + np.random.rand(nkpts, nao, nao) * 1j
        dm = dm + dm.transpose(0,2,1).conj()
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

        dm = np.einsum('npi,nqi->npq', mo, mo.conj())
        dm = lib.tag_array(dm, mo_coeff=mo, mo_occ=mo_occ)
        kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
        vk = mydf.get_jk(cp.array(dm), hermi=1, kpts=kpts, with_j=False)[1]
        assert abs(vk.get() - kref).max() < 1e-9

    def test_aft_k3(self):
        kpts = cell.make_kpts([6,1,1])
        nkpts = len(kpts)
        mesh = [11]*3
        mydf0 = aft_cpu.AFTDF(cell).set(mesh=mesh)
        mydf  = aft.AFTDF(cell).set(mesh=mesh)
        mydf0.time_reversal_symmetry = False
        mydf.time_reversal_symmetry = False

        try:
            bak = aft_jk.get_avail_mem
            aft_jk.get_avail_mem = lambda **kw: 10000000

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
            # difference is smaller for pyscf-2.13 or newer
            assert abs(vk.get() - kref).max() < 1e-8

            dm = np.einsum('npi,nqi->npq', mo, mo.conj())
            kref = mydf0.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
            dm = tag_array(cp.array(dm), mo_coeff=cp.array(mo), mo_occ=cp.array(mo_occ))
            vk = mydf.get_jk(dm, hermi=1, kpts=kpts, with_j=False)[1]
            # difference is smaller for pyscf-2.13 or newer
            assert abs(vk.get() - kref).max() < 1e-8
        finally:
            aft_jk.get_avail_mem = bak

    def test_vk_kpts_vs_fft(self):
        cell = pyscf.M(
            atom = '''
            C   1.      1.    0.
            H   4.      0.    3.
            H   0.      1.    .6
            ''',
            a=np.eye(3)*4.,
            basis=[[0, [.55, .5, .1], [.35, .6, .2], [.15, .1, .8]], [1, [.3, 1]]],
        )
        kpts = cell.make_kpts([3,2,1])
        dm_kpts = cp.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
        mydf = aft.AFTDF(cell, kpts=kpts)
        vk = aft_jk.get_k_kpts(mydf, dm_kpts, hermi=1, kpts=kpts)
        ref = fft.FFTDF(cell, kpts=kpts).get_jk(dm_kpts, hermi=1, with_j=False, kpts=kpts)[1]
        assert abs(vk - ref).max().get() < 1e-8

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
        dm = np.random.rand(2, nao, nao) - .5
        dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
        mydf = aft.AFTDF(cell)
        grad_sigma = aft_jk.get_ej_derivatives(mydf, dm)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        ej = grad_sigma[:-3]
        assert abs(ej.sum(axis=0)).max() < 1e-9

        cell.precision = 1e-10
        cell.build(0, 0)
        dm = dm[0] + dm[1]
        vj = fft_cpu.FFTDF(cell).get_j_e1(dm)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xpq,qp->x', vj[:,p0:p1], dm[:,p0:p1])
        ref *= 2
        assert abs(ej - ref).max() < 1e-8

        disp = 1e-3
        atom_coords = cell.atom_coords()
        def eval_jk(i, x, disp):
            atom_coords[i,x] += disp
            cell1 = cell.set_geom_(atom_coords, unit='Bohr')
            vj = fft_cpu.FFTDF(cell1).get_jk(dm, with_k=False)[0]
            ref = .5 * np.einsum('ij,ji->', vj, dm)
            atom_coords[i,x] -= disp
            return ref

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp) - ej[i,x]) < 1e-5

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
        grad_sigma = aft_jk.get_ej_derivatives(mydf, dm, kpts=kpts)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        ej = grad_sigma[:-3]
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
        ref *= 2
        assert abs(ej - ref).max() < 1e-8

        nkpts = len(kpts)
        disp = 1e-3
        atom_coords = cell.atom_coords()
        def eval_jk(i, x, disp):
            atom_coords[i,x] += disp
            cell1 = cell.set_geom_(atom_coords, unit='Bohr')
            vj = fft_cpu.FFTDF(cell1).get_jk(dm, kpts=kpts, with_k=False)[0]
            ref = .5/nkpts**2 * np.einsum('kij,kji->', vj, dm)
            atom_coords[i,x] -= disp
            return ref

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp) - ej[i,x]/nkpts) < 2e-6

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
        grad_sigma = aft_jk.get_ek_derivatives(myaft, dm)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        ek = grad_sigma[:-3]
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
        ref *= 2
        assert abs(ek - ref).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            vk = myfft.get_k_e1(dm, exxdiv='ewald')
            for i in range(cell.natm):
                p0, p1 = aoslices[i, 2:]
                ref[i] = np.einsum('xnpq,nqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1])
            ref *= 2
            assert abs(ek_ewald - ref).max() < 5e-8

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
        grad_sigma = aft_jk.get_ek_derivatives(myaft, dm, kpts=kpts)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        ek = grad_sigma[:-3]
        assert abs(ek.sum(axis=0)).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            ek_ewald = aft_jk.get_ek_ip1(myaft, dm, kpts=kpts, exxdiv='ewald')
            assert abs(ek_ewald.sum(axis=0)).max() < 1e-8

        cell.precision = 1e-10
        cell.build(0, 0)
        myfft = fft_cpu.FFTDF(cell, kpts=kpts)
        vk = myfft.get_k_e1(dm, kpts=kpts)
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xkpq,kqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1]).real
        ref /= len(kpts)
        ref *= 2
        assert abs(ek - ref).max() < 1e-8

        if version.parse(pyscf.__version__) > version.parse('2.11.0'):
            vk = myfft.get_k_e1(dm, kpts=kpts, exxdiv='ewald')
            for i in range(cell.natm):
                p0, p1 = aoslices[i, 2:]
                ref[i] = np.einsum('xkpq,kqp->x', vk[:,:,p0:p1], dm[:,:,p0:p1]).real
            ref /= len(kpts)
            ref *= 2
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
        grad_sigma = aft_jk.get_ej_derivatives(mydf, dm)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        sigma = grad_sigma[-3:]

        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        ref = ni.energy_derivatives(
            'HF', dm, with_j=True, with_nuc=False)[-3:]
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
        grad_sigma = aft_jk.get_ej_derivatives(mydf, dm, kpts)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        sigma = grad_sigma[-3:]

        ni = MultiGridNumInt(cell)
        ni.allow_mesh_reduction = False
        ref = ni.energy_derivatives(
            'HF', dm, kpts=kpts, with_j=True, with_nuc=False)[-3:]
        assert abs(ref - sigma).max() < 1e-8

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
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
        grad_sigma = aft_jk.get_ek_derivatives(mydf, dm)
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        sigma = grad_sigma[-3:]

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
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
        grad_sigma = aft_jk.get_ek_derivatives(mydf, dm, kpts, exxdiv='ewald')
        assert isinstance(grad_sigma, np.ndarray)
        assert grad_sigma.shape == (cell.natm+3, 3)
        sigma = grad_sigma[-3:]

        for (i, j) in [(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=1e-4)
            mydf = aft.AFTDF(cell1, kpts=cell1.make_kpts(kmesh))
            vk = aft_jk.get_k_kpts(mydf, dm, hermi=1, kpts=mydf.kpts, exxdiv='ewald')
            e1 = .5 * cp.einsum('kij,kji->', vk, dm).real / nkpts
            mydf = aft.AFTDF(cell2, kpts=cell2.make_kpts(kmesh))
            vk = aft_jk.get_k_kpts(mydf, dm, hermi=1, kpts=mydf.kpts, exxdiv='ewald')
            e2 = .5 * cp.einsum('kij,kji->', vk, dm).real / nkpts
            assert abs(sigma[i, j] - (e1-e2)/2e-4).max() < 5e-7

    def test_ewald_strain_deriv(self):
        def _exxdiv_ewald_strain_deriv(cell, kpts, omega):
            from pyscf.pbc.tools.pbc import madelung
            scaled_kpts = kpts.dot(cell.lattice_vectors().T)
            nkpts = len(kpts)
            ewald_G0_response = np.empty((3,3))
            disp = 1e-5
            for i in range(3):
                for j in range(i+1):
                    cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
                    kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
                    kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
                    e1 = nkpts * madelung(cell1, kpts1, omega=omega)
                    e2 = nkpts * madelung(cell2, kpts2, omega=omega)
                    ewald_G0_response[j,i] = ewald_G0_response[i,j] = (e1-e2)/(2*disp)
            exx_0 = nkpts * madelung(cell, kpts, omega)
            return exx_0, ewald_G0_response

        np.random.seed(1)
        cell = pgto.M(
            atom = '''
            H   0.      0.5   3.
            H   0.5     1.    .6
            ''',
            a=np.eye(3)*4. - np.random.rand(3,3),
            basis=[[0, [.25, 1]], [1, [.3, 1]]],
        )
        kmesh = [3,2,1]
        kpts = cell.make_kpts(kmesh)
        omega = -0.25
        ref = _exxdiv_ewald_strain_deriv(cell, kpts, omega)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell, kpts, omega)
        assert abs(exx_0 - ref[0]) < 1e-12
        assert abs(exx_1 - ref[1]).max() < 1e-8

        omega = 0.25
        ref = _exxdiv_ewald_strain_deriv(cell, kpts, omega)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell, kpts, omega)
        assert abs(exx_0 - ref[0]) < 1e-12
        assert abs(exx_1 - ref[1]).max() < 1e-8

class FFTCoulombDerivative_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = pgto.M(
            atom='He .2 .3 .4; He 1.4 1.2 1.6', unit='Bohr',
            a=np.diag([5., 5.5, 6.]), mesh=[25, 27, 29],
            basis=[[0, [.8, 1.]], [1, [.6, 1.]]], pseudo='gth-pade',
            precision=1e-11, verbose=0)

    def _density(self, nkpts):
        coeff = np.random.default_rng(27).normal(size=(nkpts, self.cell.nao, 3))
        return np.einsum('kip,kjp->kij', coeff, coeff) / self.cell.nao

    def _cpu_gradient(self, density, kpts):
        cell = self.cell
        nkpts = 1 if kpts is None else len(kpts)
        matrices = fft_cpu.FFTDF(cell, kpts=kpts).get_j_e1(density, kpts=kpts)
        matrices = np.asarray(matrices).reshape(3, nkpts, cell.nao, cell.nao)
        dms = density.reshape(nkpts, cell.nao, cell.nao)
        result = np.empty((cell.natm, 3))
        for atom, (_, _, p0, p1) in enumerate(cell.aoslice_by_atom()):
            result[atom] = np.einsum('xkpq,kqp->x', matrices[:, :, p0:p1],
                                     dms[:, :, p0:p1]).real * (2 / nkpts)
        return result

    def _cpu_energy(self, cell, density, kpts):
        nkpts = 1 if kpts is None else len(kpts)
        vj = fft_cpu.FFTDF(cell, kpts=kpts).get_jk(density, kpts=kpts, with_k=False)[0]
        return .5 * np.einsum('kij,kji->', np.asarray(vj).reshape(nkpts, cell.nao, cell.nao),
                              density.reshape(nkpts, cell.nao, cell.nao)).real / nkpts

    def _check_derivative(self, kpts):
        cell = self.cell
        nkpts = 1 if kpts is None else len(kpts)
        density = self._density(nkpts)
        if kpts is None:
            density = density[0]
        reference = self._cpu_gradient(density, kpts)
        for order in ('C', 'F'):
            with self.subTest(order=order):
                dm_gpu = cp.array(density, order=order)
                before = dm_gpu.copy()
                actual = fft.FFTDF(cell, kpts=kpts).get_j_e1(dm_gpu, kpts=kpts)
                self.assertIsInstance(actual, np.ndarray)
                self.assertEqual(actual.shape, (cell.natm, 3))
                np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(dm_gpu.get(), before.get())
        coords = cell.atom_coords()
        # Fixed Cartesian AO DMs and lattice keep Cartesian/fractional k-points fixed.
        step = 1e-4
        fd = np.empty((cell.natm, 3))
        for atom in range(cell.natm):
            for axis in range(3):
                plus, minus = coords.copy(), coords.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                p = cell.set_geom_(plus, unit='Bohr', inplace=False)
                m = cell.set_geom_(minus, unit='Bohr', inplace=False)
                fd[atom, axis] = (self._cpu_energy(p, density, kpts)
                                  - self._cpu_energy(m, density, kpts)) / (2 * step)
        np.testing.assert_allclose(reference, fd, atol=3e-8, rtol=0)

    def test_gamma_coulomb_derivative(self):
        self._check_derivative(None)

    def test_kpoint_coulomb_derivative(self):
        self._check_derivative(self.cell.make_kpts([2, 1, 1]))

    def test_coulomb_derivative_default_kpts(self):
        kpts = self.cell.make_kpts([2, 1, 1])
        density = self._density(len(kpts))
        mydf = fft.FFTDF(self.cell, kpts=kpts)
        result = mydf.get_j_e1(cp.asarray(density))
        self.assertEqual(result.shape, (self.cell.natm, 3))
        np.testing.assert_allclose(result, self._cpu_gradient(density, kpts),
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(result, mydf.get_j_e1(density, kpts=kpts),
                                   atol=1e-12, rtol=0)


class FactorizedExchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scipy.linalg import eigh
        cls.cell = pgto.M(atom='He .3 .4 .2', unit='Bohr',
                          a=np.diag([5., 5.5, 6.]), mesh=[9, 9, 9],
                          basis=[[0, [1.2, 1.]], [1, [.8, 1.]]],
                          precision=1e-10, verbose=0)
        cls.kpts = cls.cell.make_kpts([3, 1, 1])
        cpu = aft_cpu.AFTDF(cls.cell, cls.kpts)
        overlap = cls.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=cls.kpts)
        hcore = (cls.cell.pbc_intor('int1e_kin', hermi=1, kpts=cls.kpts)
                 + cpu.get_nuc(cls.kpts))
        energy, coeff = zip(*(eigh(h, s) for h, s in zip(hcore, overlap)))
        coeff = np.asarray(coeff)
        # Fix the time-reversed gauge using actual one-electron Bloch orbitals.
        coeff[0] = coeff[0].real
        coeff[2] = coeff[1].conj()
        for k, c in enumerate(coeff):
            np.testing.assert_allclose(c.conj().T @ overlap[k] @ c, np.eye(cls.cell.nao),
                                       atol=1e-11, rtol=0)
            np.testing.assert_allclose(hcore[k] @ c, (overlap[k] @ c) * energy[k],
                                       atol=1e-10, rtol=0)
        occ, vir = coeff[:, :, :1], coeff[:, :, 1:2]
        cls.factors = {'occupied': (np.sqrt(2)*occ, np.sqrt(2)*occ.conj()),
                       'transition': (vir, occ.conj())}
        cls.densities = {key: left @ right.transpose(0, 2, 1)
                         for key, (left, right) in cls.factors.items()}
        cls.references = {key: cpu.get_jk(dm, hermi=int(key == 'occupied'),
                                         kpts=cls.kpts, with_j=False, exxdiv=None)[1]
                          for key, dm in cls.densities.items()}

    def _gpu(self, key, time_reversal=True, tagged=False, folded=False):
        from gpu4pyscf.df.df_jk import factorize_dm
        points = self.kpts
        if folded:
            scaled = self.cell.get_scaled_kpts(points)
            points = self.cell.get_abs_kpts(scaled - np.floor(scaled + .5))
        dm = cp.asarray(self.densities[key])
        if tagged:
            left, right = self.factors[key]
            dm = tag_array(dm, factor_l=cp.asarray(left),
                           factor_r=cp.asarray(right))
        left, right = factorize_dm(dm)
        self.assertIsNotNone(right)
        self.assertLess(2*left.shape[-1], self.cell.nao)
        np.testing.assert_allclose((left @ right.swapaxes(-1, -2)).get(), dm.get(),
                                   atol=1e-12, rtol=0)
        before = dm.copy()
        left_before, right_before = left.copy(), right.copy()
        mydf = aft.AFTDF(self.cell, points)
        mydf.time_reversal_symmetry = time_reversal
        result = aft_jk.get_k_kpts(mydf, dm, hermi=int(key == 'occupied'),
                                   kpts=points, exxdiv=None)
        np.testing.assert_array_equal(dm.get(), before.get())
        np.testing.assert_array_equal(left.get(), left_before.get())
        np.testing.assert_array_equal(right.get(), right_before.get())
        return result.get()

    def _check_transition_density_time_reversal(self, symmetry):
        for tagged in (False, True):
            with self.subTest(symmetry=symmetry, tagged=tagged):
                actual = self._gpu('transition', symmetry, tagged)
                np.testing.assert_allclose(actual, self.references['transition'],
                                           atol=1e-9, rtol=0)

    def test_transition_density_time_reversal_enabled(self):
        self._check_transition_density_time_reversal(True)

    def test_transition_density_time_reversal_disabled(self):
        self._check_transition_density_time_reversal(False)

    def _check_occupied_density_time_reversal(self, symmetry):
        for tagged in (False, True):
            with self.subTest(symmetry=symmetry, tagged=tagged):
                actual = self._gpu('occupied', symmetry, tagged)
                np.testing.assert_allclose(actual, self.references['occupied'],
                                           atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual, actual.conj().transpose(0, 2, 1),
                                           atol=1e-10, rtol=0)

    def test_occupied_density_time_reversal_enabled(self):
        self._check_occupied_density_time_reversal(True)

    def test_occupied_density_time_reversal_disabled(self):
        self._check_occupied_density_time_reversal(False)

    def test_exchange_linearity(self):
        from gpu4pyscf.df.df_jk import factorize_dm
        scale = .37 + .2j
        # Both terms share an occupied right factor, so the sum remains rank one.
        density = self.densities['occupied'] + scale*self.densities['transition']
        dm = cp.asarray(density)
        left, right = factorize_dm(dm)
        self.assertLess(2*left.shape[-1], self.cell.nao)
        mydf = aft.AFTDF(self.cell, self.kpts)
        result = aft_jk.get_k_kpts(mydf, dm, hermi=0, kpts=self.kpts, exxdiv=None)
        expected = self.references['occupied'] + scale*self.references['transition']
        np.testing.assert_allclose(result.get(), expected, atol=1e-9, rtol=0)


if __name__ == '__main__':
    print("Full Tests for aft")
    unittest.main()
