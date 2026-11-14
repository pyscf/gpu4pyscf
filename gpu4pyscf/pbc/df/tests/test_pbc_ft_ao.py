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

import pytest
import unittest
import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df import ft_ao as ft_ao_cpu
from pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.pbc.lib.kpts_helper import kpts_to_bvkmesh
from gpu4pyscf.pbc.df import ft_ao as ft_ao_gpu
from gpu4pyscf.pbc.df.ft_ao import ft_aopair, ft_aopair_kpts
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.df.aft import AFTDF
from gpu4pyscf.pbc.tools import get_coulG

def setUpModule():
    global cell, cell_he
    cell = pgto.M(
        verbose=5, output='/dev/null',
        atom='''O1   1.3    .2       .3
                N2   .19   .1      1.1
                N2   .5    .4      .7 ''',
        basis={'O1': [[3, [.5, 1.]], [4, [2., 1.]]], 'N2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2]),
        precision=1e-8)

    cell_he = pgto.M(atom='He .2 .3 .4', a=np.eye(3)*5, unit='Bohr',
                    basis=[[0, [1., 1.]], [0, [.4, 1.]]], pseudo='gth-pade',
                    mesh=[15]*3, precision=1e-8, verbose=0)

def tearDownModule():
    global cell, cell_he
    cell.stdout.close()
    del cell, cell_he

class KnownValues(unittest.TestCase):
    def test_ft_aopair_gamma_point(self):
        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair(cell, Gv).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair(cell, Gv).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpt(self):
        kpts = cell.get_abs_kpts([6/7, 2/15, 3/8])
        kpti = kptj = kpts

        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        np.random.seed(1)
        kpti = kptj = np.random.random(3)
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpts(self):
        kpts = cell.make_kpts([3,4,3])
        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        q = kpts[11]
        dat = ft_aopair_kpts(cell, Gv, q=q, kptjs=kpts).get()
        ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, q=q, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpt_no_aosym(self):
        np.random.seed(1)
        kpti, kptj = kpti_kptj = np.random.random((2,3))
        Gv = cell.get_Gv([3]*3)
        kpts = kptj.reshape(1,3)
        kmesh = kpts_to_bvkmesh(cell, kpts, bound_by_supmol=True)
        ft_opt = ft_ao_gpu.FTOpt(cell, kmesh)
        ft_opt.permutation_symmetry = False
        ft_kern = ft_opt.gen_ft_kernel()
        q = kptj - kpti
        dat = ft_kern(Gv, q=q, kpts=kpts).get()
        ref = ft_ao_cpu.ft_aopair(cell, Gv, kpti_kptj=kpti_kptj)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpts_nosym(self):
        np.random.seed(2)
        kpts = np.random.random((4,3))
        Gv = cell.get_Gv([3]*3)
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_ao(self):
        Gv = cell.get_Gv(mesh=[9,7,7])
        dat = ft_ao_gpu.ft_ao(cell, Gv).get()
        ref = ft_ao_cpu.ft_ao(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        pcell = cell.copy()
        pcell.cart = True
        dat = ft_ao_gpu.ft_ao(pcell, Gv).get()
        ref = ft_ao_cpu.ft_ao(pcell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_ao_batch_evaluation(self):
        Gv = cell.get_Gv(mesh=[9,7,7])

        ft_opt = ft_ao_gpu.FTOpt(cell).build()
        eval_ft, ao_pair_offsets = ft_opt.ft_evaluator()
        ref = eval_ft(Gv)

        batch_size = int(ref.shape[0] *.23)
        eval_ft, ao_pair_offsets = ft_opt.ft_evaluator(batch_size=batch_size)
        dat = cp.empty_like(ref)
        for i, (p0, p1) in enumerate(zip(ao_pair_offsets[:-1],
                                         ao_pair_offsets[1:])):
            dat[p0:p1] = eval_ft(Gv, i)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 12)

        ft_opt = ft_ao_gpu.FTOpt(cell, bvkmesh=[3,1,2]).build()
        eval_ft, ao_pair_offsets = ft_opt.ft_evaluator()
        ref = eval_ft(Gv)

        batch_size = int(ref.shape[0] *.23)
        eval_ft, ao_pair_offsets = ft_opt.ft_evaluator(batch_size=batch_size)
        dat = cp.empty_like(ref)
        for i, (p0, p1) in enumerate(zip(ao_pair_offsets[:-1],
                                         ao_pair_offsets[1:])):
            dat[p0:p1] = eval_ft(Gv, i)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 12)

    @pytest.mark.slow
    def test_ft_ao_large(self):
        Gv = cell.get_Gv(mesh=[129,128,128])
        dat = ft_ao_gpu.ft_ao(cell, Gv).get()
        ref = ft_ao_cpu.ft_ao(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        pcell = cell.copy()
        pcell.cart = True
        dat = ft_ao_gpu.ft_ao(pcell, Gv).get()
        ref = ft_ao_cpu.ft_ao(pcell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_contract_dm(self):
        nao = cell.nao
        cp.random.seed(10)
        dm = cp.random.rand(nao,nao)
        dm = dm.dot(dm.T)
        Gv = cell.get_Gv(mesh=[9,11,3])
        ft_opt = ft_ao_gpu.FTOpt(cell).build()
        rhoG = ft_opt.contract_dm(ft_opt.cell.apply_C_mat_CT(dm), Gv).get()

        ref = cp.einsum('Gpq,qp->G', ft_aopair(cell, Gv), dm).get()
        self.assertAlmostEqual(abs(ref-rhoG).max(), 0, 10)

    def test_j(self):
        cp.random.seed(10)
        nao = cell.nao
        dm = cp.random.rand(nao, nao)
        dm = dm + dm.T
        mesh = [9,11,3]
        Gv = cell.get_Gv(mesh=mesh)
        ft_opt = ft_ao_gpu.FTOpt(cell).build()
        rhoG = ft_opt.contract_dm(ft_opt.cell.apply_C_mat_CT(dm), Gv)
        coulG = rhoG * cp.array(get_coulG(cell, Gv=Gv)) / cell.vol
        vj = ft_opt.contract_rhoG(coulG, -Gv, sort_output=True)

        mydf  = AFTDF(cell).set(mesh=mesh)
        jref = mydf.get_jk(dm, with_k=False)[0]
        assert vj.ndim == 2
        assert vj.shape == jref.shape
        assert abs(vj - jref).max().get() < 1e-9

        vj_batch = ft_opt.contract_rhoG(coulG, -Gv, kpts=np.zeros((1, 3)))
        vj_single = ft_opt.contract_rhoG(coulG, -Gv, kpts=np.zeros(3))
        assert vj_batch.shape == (1,) + vj.shape
        assert vj_single.shape == vj.shape
        assert abs(vj_batch - jref).max().get() < 1e-9
        assert abs(vj_single - jref).max().get() < 1e-9

    @unittest.skip('FTOpt.contract_dm with kpts not supported')
    def test_j_kpts(self):
        cp.random.seed(10)
        kmesh = [3,2,1]
        kpts = cell.make_kpts(kmesh)
        nk = len(kpts)
        nao = cell.nao
        dm = cp.random.rand(nk, nao, nao)
        dm = dm + dm.transpose(0, 2, 1)
        mesh = [9,11,3]
        Gv = cell.get_Gv(mesh=mesh)
        ft_opt = ft_ao_gpu.FTOpt(cell)
        rhoG = ft_opt.contract_dm(ft_opt.cell.apply_C_mat_CT(dm), Gv, kpts=kpts)
        coulG = rhoG * cp.array(get_coulG(cell, Gv=Gv)) / cell.vol
        vj = ft_opt.contract_rhoG(coulG, -Gv, kpts=kpts, sort_output=True)

        mydf  = AFTDF(cell).set(mesh=mesh)
        jref = mydf.get_jk(dm, kpts=kpts, with_k=False)[0]
        assert vj.ndim == 3
        assert vj.shape == jref.shape
        assert abs(vj - jref).max().get() < 1e-9

    def test_gen_ft_kernel(self):
        kmesh = [2, 3, 2]
        kpts = cell.make_kpts(kmesh)
        kpt_iters = list(kk_adapted_iter(cell, kpts))

        opt = ft_ao_gpu.FTOpt(cell, kmesh)
        ft_kern = opt.gen_ft_kernel()

        Gv = cell.get_Gv([3,2,5])
        for kpt, ki_idx, kj_idx, self_conj in kpt_iters:
            dat = ft_kern(Gv, kpt, kpts, kj_idx)
            dat_conj = ft_kern(-Gv, -kpt, -kpts, kj_idx)
            self.assertAlmostEqual(abs(dat_conj-dat.conj()).max().get(), 0, 9)

            ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, q=kpt, kptjs=kpts)
            self.assertAlmostEqual(abs(ref-dat.get()).max(), 0, 9)

    def test_gen_ft_kernel_off_gamma(self):
        cell = pgto.M(
            atom='He .2 .3 .4; He 1.2 1.5 1.8',
            a=[[5., 0., 0.], [1., 6., 0.], [0., 1., 7.]], unit='Bohr',
            basis=[[0, [1., 1.]], [1, [.6, 1.]]],
            precision=1e-10, verbose=0)
        kpts = cell.make_kpts([2, 3, 1], with_gamma_point=False)
        bvkmesh = kpts_to_bvkmesh(cell, kpts)
        np.testing.assert_array_equal(bvkmesh, [4, 3, 1])
        opt = ft_ao_gpu.FTOpt(cell, bvkmesh)

        cached_kernel = opt.gen_ft_kernel(kpts=kpts)
        kernel = opt.gen_ft_kernel()
        Gv = cell.get_Gv([3, 3, 3])

        for q, ki_idx, kj_idx, self_conj in kk_adapted_iter(cell, kpts):
            ref = ft_ao_cpu.ft_aopair_kpts(cell, Gv, q=q, kptjs=kpts)
            dat = cached_kernel(Gv, q, kj_idx=kj_idx)
            np.testing.assert_allclose(dat.get(), ref, atol=1e-9, rtol=0)
            # Supplying k-points at evaluation time must give the same tensor.
            dat = kernel(Gv, q, kpts, kj_idx)
            np.testing.assert_allclose(dat.get(), ref, atol=1e-9, rtol=0)

    def test_ft_ao_strain_deriv(self):
        from gpu4pyscf.gto.mole import SortedGTO
        from gpu4pyscf.pbc.grad.rhf import _finite_diff_cells
        cell = pgto.M(
            verbose=5, output='/dev/null',
            atom='''N1   .19   .1      1.4
                    N2   .5   2.4      .7 ''',
            basis={'N1': [[3, [.5, 1.]], [6, [2., 1.]]], 'N2': 'ccpvdz'},
            a=np.diag([2.5, 3.9, 2.2]),
            unit='B',
            precision=1e-8)

        ft_opt = ft_ao_gpu.FTOpt(cell).build()
        nao = ft_opt.cell.nao
        cp.random.seed(6)
        dm = cp.random.rand(nao, nao) * .4
        dm = dm + dm.T
        mesh = [5, 5, 5]
        Gv = cell.get_Gv(mesh)
        GvT = cp.asarray(Gv.T, order='C')
        auxG = ft_ao_gpu.ft_ao(ft_opt.cell, Gv, sort_output=False).T
        dm_auxG = dm.dot(auxG)
        grad = cp.zeros((2, 3))
        sigma = cp.zeros((3, 3))
        aux_ft_envs = ft_ao_gpu.RysIntEnvVars.new(
            ft_opt.cell.natm, ft_opt.cell.nbas, ft_opt.cell._atm,
            ft_opt.cell._bas, ft_ao_gpu._scale_sp_ctr_coeff(ft_opt.cell),
            ft_opt.cell.ao_loc)
        err = libpbc.PBC_ft_ao_deriv(
            ctypes.cast(grad.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.cast(dm_auxG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aux_ft_envs),
            ctypes.c_int(len(Gv)),
            ctypes.c_void_p(cp.cuda.get_current_stream().ptr))
        assert err == 0

        grad = grad.get()
        sigma = sigma.get()
        assert abs(grad.sum(axis=0)).max() < 1e-10

        disp = 1e-5
        def eval_aux(c):
            Gv = c.get_Gv(mesh)
            auxG = ft_ao_gpu.ft_ao(SortedGTO.from_cell(c), Gv, sort_output=False).T
            return cp.einsum('ag,ag->', auxG.conj(), dm_auxG).real.get()

        for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 2)]:
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp=disp)
            e1 = eval_aux(cell1)
            e2 = eval_aux(cell2)
            assert abs(sigma[i, j] - (e1-e2)/2/disp) < 3e-8

        coords = cell.atom_coords()
        for i in range(cell.natm):
            for j in range(3):
                coords[i,j] += disp
                e1 = eval_aux(cell.set_geom_(coords, inplace=False))
                coords[i,j] -= 2*disp
                e2 = eval_aux(cell.set_geom_(coords, inplace=False))
                coords[i,j] += disp
                assert abs(grad[i, j] - (e1-e2)/2/disp) < 3e-8

    def test_fourier_gamma_and_cached_sampling(self):
        Gv = cell_he.get_Gv([3]*3)
        default = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv)
        single = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=np.zeros(3))
        batch = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=np.zeros((1, 3)))
        assert default.shape == single.shape == (len(Gv), cell_he.nao, cell_he.nao)
        assert batch.shape == (1,) + default.shape
        cp.testing.assert_allclose(default, single, atol=1e-12, rtol=0)
        cp.testing.assert_allclose(default, batch[0], atol=1e-12, rtol=0)
        cp.testing.assert_allclose(default, ft_ao_gpu.gen_ft_kernel(cell_he)(Gv), atol=1e-12, rtol=0)
        kpts = cell_he.make_kpts([2, 1, 1])
        kernel = ft_ao_gpu.FTOpt(cell_he, [2, 1, 1]).gen_ft_kernel(kpts=kpts)
        cached = kernel(Gv)
        assert cached.shape == (2, len(Gv), cell_he.nao, cell_he.nao)
        cp.testing.assert_allclose(cached, kernel(Gv, kpts=kpts), atol=1e-12, rtol=0)

    def test_fourier_nonzero_single_and_batch(self):
        kpts = cell_he.make_kpts([3, 1, 1])
        Gv = cell_he.get_Gv([3]*3)
        single = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=kpts[1])
        one = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=kpts[1:2])
        batch = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=kpts)
        assert single.shape == (len(Gv), cell_he.nao, cell_he.nao)
        assert one.shape == (1,) + single.shape
        assert batch.shape == (3,) + single.shape
        assert single.dtype == one.dtype == batch.dtype == np.complex128
        cp.testing.assert_allclose(single, one[0], atol=1e-10, rtol=0)
        cp.testing.assert_allclose(single, batch[1], atol=1e-10, rtol=0)

    def test_fourier_cached_gamma_shape(self):
        Gv = cell_he.get_Gv([3]*3)
        reference = ft_ao_gpu.ft_aopair(cell_he, Gv)
        opt = ft_ao_gpu.FTOpt(cell_he)

        kernel = opt.gen_ft_kernel(kpts=None)
        result = kernel(Gv)
        assert result.shape == reference.shape
        cp.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)
        assert kernel(Gv, kpts=np.zeros(3)).shape == reference.shape
        assert kernel(Gv, kpts=np.zeros((1, 3))).shape == (1,) + reference.shape

        kernel = opt.gen_ft_kernel(kpts=np.zeros((1, 3)))
        result = kernel(Gv)
        assert result.shape == (1,) + reference.shape
        cp.testing.assert_allclose(result[0], reference, atol=1e-12, rtol=0)
        assert kernel(Gv, kpts=np.zeros(3)).shape == reference.shape
        assert kernel(Gv, kpts=np.zeros((1, 3))).shape == (1,) + reference.shape

    def test_ft_aopair_kpts_with_KPoints(self):
        kpts = cell_he.make_kpts([3, 1, 1])
        sampling = KPoints(cell_he, kpts)
        Gv = cell_he.get_Gv([3]*3)
        reference = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=kpts)
        result = ft_ao_gpu.ft_aopair_kpts(cell_he, Gv, kptjs=sampling)
        cp.testing.assert_allclose(result, reference, atol=1e-12, rtol=0)
        opt = ft_ao_gpu.FTOpt(cell_he, [3, 1, 1])
        kernel = opt.gen_ft_kernel(kpts=sampling)
        cp.testing.assert_allclose(kernel(Gv), reference, atol=1e-12, rtol=0)
        cp.testing.assert_allclose(kernel(Gv, kpts=sampling), reference, atol=1e-12, rtol=0)
        rhoG = cp.ones(len(Gv), dtype=np.complex128)
        cp.testing.assert_allclose(opt.contract_rhoG(rhoG, Gv, sampling),
                                   opt.contract_rhoG(rhoG, Gv, kpts), atol=1e-12, rtol=0)


if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
