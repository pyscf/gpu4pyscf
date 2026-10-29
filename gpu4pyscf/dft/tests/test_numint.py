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
import cupy
import cupy as cp
import pyscf
from pyscf import lib, scf
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft import Grids
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.dft import xc_deriv
from gpu4pyscf.lib.cupy_helper import tag_array, libcupy_helper
try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None

HAS_GROUPED_GEMM = (hasattr(libcupy_helper, 'grouped_gemm') and
                    hasattr(libcupy_helper, 'grouped_dot'))

def setUpModule():
    global mol, grids_cpu, grids_gpu, dm, dm0, dm1, mo_occ, mo_coeff
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvdz',
        charge = 1,
        spin = 1,  # = 2S = spin_up - spin_down
        output = '/dev/null')

    np.random.seed(2)
    mf = scf.UHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1().copy()
    dm = dm1
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = (mo_coeff[0]*mo_occ[0]).dot(mo_coeff[0].T)

    grids_gpu = Grids(mol)
    grids_gpu.level = 1
    grids_gpu.build()

    grids_cpu = grids_gpu.to_cpu()
    grids_cpu.weights = cupy.asnumpy(grids_gpu.weights)
    grids_cpu.coords = cupy.asnumpy(grids_gpu.coords)

def tearDownModule():
    global mol, grids_cpu, grids_gpu
    mol.stdout.close()
    del mol, grids_cpu, grids_gpu

LDA = 'LDA_C_VWN'
GGA_PBE = 'GGA_C_PBE'
MGGA_M06 = 'MGGA_C_M06'

class KnownValues(unittest.TestCase):

    def _check_vxc(self, method, xc):
        ni = NumInt()
        fn = getattr(ni, method)
        dm = dm1
        if method == 'nr_rks':
            dm = dm0
        n, e, v = fn(mol, grids_gpu, xc, dm, hermi=1)
        v = [x.get() for x in v]

        ni_pyscf = pyscf_numint()
        fn = getattr(ni_pyscf, method)
        nref, eref, vref = fn(mol, grids_cpu, xc, dm, hermi=1)

        v = cupy.asarray(v)
        vref = cupy.asarray(vref)

        assert cupy.allclose(e, eref)
        assert cupy.allclose(n, nref)
        assert cupy.allclose(v, vref)

    def _check_rks_fxc(self, xc, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        spin = 0
        ni_pyscf = pyscf_numint()
        rho, vxc, fxc = ni_pyscf.cache_xc_kernel(mol, grids_cpu, xc, mo_coeff[0], mo_occ[0], spin)
        vref = ni_pyscf.nr_rks_fxc(
            mol, grids_cpu, xc, dm0=dm0, dms=t1, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)

        rho0 = rho.copy()
        vxc0 = vxc.copy()
        fxc0 = fxc.copy()
        ni = NumInt()
        rho, vxc, fxc = ni.cache_xc_kernel(mol, grids_gpu, xc, cupy.asarray(mo_coeff[0]), cupy.asarray(mo_occ[0]), spin)
        v = ni.nr_rks_fxc(mol, grids_gpu, xc, dms=t1, fxc=fxc, hermi=hermi)

        assert cupy.linalg.norm(rho - cupy.asarray(rho0)) < 1e-6 * cupy.linalg.norm(rho)
        assert cupy.linalg.norm(vxc - cupy.asarray(vxc0)) < 1e-6 * cupy.linalg.norm(vxc)
        assert cupy.linalg.norm(fxc - cupy.asarray(fxc0)) < 1e-6 * cupy.linalg.norm(fxc)
        assert cupy.allclose(v, vref)

    def _check_rks_fxc_st(self, xc, fpref):
        ni = NumInt()
        spin = 1
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids_gpu, xc, mo_coeff, mo_occ, spin)
        v = ni.nr_rks_fxc_st(mol, grids_gpu, xc, dms_alpha=dm, fxc=fxc)
        self.assertAlmostEqual(lib.fp(v), fpref, 12)

        ni_pyscf = pyscf_numint()
        rho, vxc, fxc = ni_pyscf.cache_xc_kernel(
            mol, grids_cpu, xc, mo_coeff, mo_occ, spin)
        vref = ni_pyscf.nr_rks_fxc_st(
            mol, grids_cpu, xc, dm0=dm0, dms_alpha=dm, rho0=rho, vxc=vxc, fxc=fxc)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 12)

    def _check_uks_fxc(self, xc, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        ni = NumInt()
        spin = 1
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids_gpu, xc, cupy.asarray(mo_coeff), cupy.asarray(mo_occ), spin)
        v = ni.nr_uks_fxc(mol, grids_gpu, xc, dms=t1, fxc=fxc, hermi=hermi)

        ni = pyscf_numint()
        dm0 = mo_coeff.dot(mo_coeff.T)
        rho_ref, vxc_ref, fxc_ref = ni.cache_xc_kernel(
            mol, grids_cpu, xc, mo_coeff, mo_occ, spin)
        v_ref = ni.nr_uks_fxc(
            mol, grids_cpu, xc, dm0=dm0, dms=t1, rho0=rho_ref, vxc=vxc_ref, fxc=fxc_ref, hermi=hermi)
        vxc_ref = np.asarray(vxc_ref)
        rho_ref = np.asarray(rho_ref)

        assert cupy.linalg.norm(rho - cupy.asarray(rho_ref)) < 1e-6 * cupy.linalg.norm(rho)
        assert cupy.linalg.norm(vxc - cupy.asarray(vxc_ref)) < 1e-6 * cupy.linalg.norm(vxc)
        assert cupy.linalg.norm(fxc - cupy.asarray(fxc_ref)) < 1e-6 * cupy.linalg.norm(fxc)
        assert cupy.linalg.norm(v - cupy.asarray(v_ref)) < 1e-6 * cupy.linalg.norm(v)

    def test_rks_lda(self):
        self._check_vxc('nr_rks', LDA)

    def test_rks_gga(self):
        self._check_vxc('nr_rks', GGA_PBE)

    def test_rks_mgga(self):
        self._check_vxc('nr_rks', MGGA_M06)

    def test_uks_lda(self):
        self._check_vxc('nr_uks', LDA)#'lda', -6.362059440515177)

    def test_uks_gga(self):
        self._check_vxc('nr_uks', GGA_PBE)#'pbe', -6.732546841646528)

    def test_uks_mgga(self):
        self._check_vxc('nr_uks', MGGA_M06)#'m06', 83.5606316500255)

    def test_rks_fxc_lda(self):
        self._check_rks_fxc(LDA, hermi=1)

    def test_rks_fxc_gga(self):
        self._check_rks_fxc(GGA_PBE, hermi=1)

    def test_rks_fxc_mgga(self):
        self._check_rks_fxc(MGGA_M06, hermi=1)

    def test_uks_fxc_lda(self):
        self._check_uks_fxc(LDA, hermi=1)

    def test_uks_fxc_gga(self):
        self._check_uks_fxc(GGA_PBE, hermi=1)

    def test_uks_fxc_mgga(self):
        self._check_uks_fxc(MGGA_M06, hermi=1)

    '''
    # Not implemented yet
    
    def test_rks_fxc_st_lda(self):
        self._check_rks_fxc_st('lda', -0.06358425564270553)

    def test_rks_fxc_st_gga(self):
        self._check_rks_fxc_st('pbe', -0.006650911990898234)

    def test_rks_fxc_st_mgga(self):
        self._check_rks_fxc_st('m06', 1.2456987899337242)
    '''

    def test_vv10(self):
        np.random.seed(10)
        rho = np.random.random((4,60))
        weight = np.random.random(60)
        coords = (np.random.random((60,3))-.5)*3
        nlc_pars = .8, .3

        ref_e, ref_v = pyscf.dft.numint._vv10nlc(rho, coords, rho, weight, coords, nlc_pars)

        rho = cupy.asarray(rho)
        coords = cupy.asarray(coords)
        rho = cupy.asarray(rho)
        weight = cupy.asarray(weight)
        coords = cupy.asarray(coords)

        e, v = dft.numint._vv10nlc(rho, coords, weight, nlc_pars)
        v = v.get()

        assert np.max(np.abs(v - ref_v)) < 1e-9

    def test_eval_rho(self):
        np.random.seed(1)
        dm = np.random.random(dm0.shape)
        ni_gpu = NumInt()
        ni_cpu = pyscf_numint()
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho(mol, ao_gpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

            rho = ni_gpu.eval_rho(mol, ao_gpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_sparse_index(self):
        mol = pyscf.M(atom='''
O   0.     0.    0.
H   5.5    0.3   2.8
H   1.7   -2.0   0.4''',
                      basis='def2-tzvpp')
        with lib.temporary_env(numint, MIN_BLK_SIZE=128**2):
            grids = gen_grid.Grids(mol).set(atom_grid=(200, 1454)).build()
            ni = NumInt()
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
            opt.l_ctr_offsets
            ao_loc = opt._sorted_mol.ao_loc
            ngrids = grids.size
            dat = grids.get_non0ao_idx(opt)
            assert lib.fp(cupy.hstack([x[1] for x in dat]).get()) == 103.60117204957997
            assert lib.fp(cupy.hstack([x[2] for x in dat]).get()) == 5.616197331343498
            assert lib.fp(cupy.hstack([x[3] for x in dat]).get()) == -22.394314323727
            assert lib.fp(cupy.hstack([x[4] for x in dat]).get()) == 351.2385939586691
            assert [i.size for x in dat for i in x[1:]] == [
                46, 18, 9, 19, 50, 20, 9, 21, 28, 12, 9, 13, 49, 19, 9, 20, 45, 17,
                9, 18, 45, 17, 9, 18, 45, 17, 9, 18, 45, 17, 9, 18, 50, 20, 9, 21,
                55, 21, 9, 22, 53, 19, 9, 20, 48, 18, 9, 19, 53, 19, 9, 20, 48, 18,
                9, 19, 43, 17, 9, 18, 40, 16, 9, 17, 48, 18, 9, 19, 48, 18, 9, 19,
                48, 18, 9, 19, 57, 21, 9, 22, 23, 11, 9, 12, 28, 12, 9, 13, 37, 15,
                9, 16, 33, 13, 9, 14, 32, 14, 9, 15, 20, 10, 9, 11, 23, 11, 9, 12,
                23, 11, 9, 12, 23, 11, 9, 12, 37, 15, 9, 16]
            assert all(x.dtype == np.int32 for x in dat[0][1:])

            if hasattr(numint, '_sparse_index'):
                for i, i0 in enumerate(range(0, ngrids, numint.MIN_BLK_SIZE)):
                    i1 = min(i0+numint.MIN_BLK_SIZE, ngrids)
                    ref = numint._sparse_index(
                        opt._sorted_mol, grids.coords[i0:i1], opt.l_ctr_offsets, ao_loc, opt)
                    assert all(r.shape == x.shape and bool((r == x).all())
                               for r, x in zip(ref[1:], dat[i][1:]))

    def test_scale_ao(self):
        ao = cupy.random.rand(1, 3, 256)
        wv = cupy.random.rand(1, 256)
        out = cupy.ones(6 * 256)
        ref = cupy.einsum('nip,np->ip', ao, wv)
        assert abs(ref - numint._scale_ao(ao, wv)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao, wv, out=out)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao+0j, wv, out=out)).max() < 1e-12
        assert abs(ref - numint._scale_ao(ao[0]+0j, wv[0], out=out)).max() < 1e-12

        ao = ao.transpose(1, 0, 2).copy(order='C').transpose(1, 0, 2)
        assert abs(ref - numint._scale_ao(ao, wv)).max() < 1e-12

        assert abs(ref - numint._scale_ao(ao[0], wv[0])).max() < 1e-12

    def test_get_rho_mo_input(self):
        mol = pyscf.M(
            atom = """
                O 0 0 0
                H 1 0 0
                H -0.5 0.5 0
            """,
            basis = '6-31g',
        )
        mf = mol.RKS(xc = 'r2scan').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (3,6)
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        assert getattr(dm, "mo_coeff", None) is not None
        assert getattr(dm, "mo_occ", None) is not None
        test_rho = mf._numint.get_rho(mf.mol, dm, mf.grids)
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_naive(mol, dm, mf.grids)

        assert test_rho.shape == ref_rho.shape
        assert np.max(np.abs(test_rho - ref_rho)) < 1e-12

    def test_get_rho_dm_input(self):
        mol = pyscf.M(
            atom = """
                O 0 0 0
                H 1 0 0
                H -0.5 0.5 0
            """,
            basis = '6-31g',
        )
        mf = mol.RKS(xc = 'PBE0').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        dm = dm.copy() # Remove attached fields
        assert getattr(dm, "mo_coeff", None) is None
        assert getattr(dm, "mo_occ", None) is None
        test_rho = mf._numint.get_rho(mf.mol, dm, mf.grids)
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_naive(mol, dm, mf.grids)

        assert test_rho.shape == ref_rho.shape
        assert np.max(np.abs(test_rho - ref_rho)) < 1e-12

    def test_get_rho_with_derivatives_restricted_mo_input(self):
        mol = pyscf.M(
            atom = """
                O 0 0 0
                H 1 0 0
                H -0.5 0.5 0
            """,
            basis = '6-31g',
        )
        mf = mol.RKS(xc = 'PBE').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        assert getattr(dm, "mo_coeff", None) is not None
        assert getattr(dm, "mo_occ", None) is not None

        test_rho = mf._numint.get_rho_with_derivatives(mf.mol, dm, mf.grids)
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_with_derivatives_naive(mol, dm, mf.grids)

        assert test_rho.shape == (1, 5, mf.grids.coords.shape[0])
        assert  ref_rho.shape == (1, 5, mf.grids.coords.shape[0])
        assert np.max(np.abs(test_rho - ref_rho)) < 2e-11

    def test_get_rho_with_derivatives_restricted_dm_input(self):
        mol = pyscf.M(
            atom = """
                O 0 0 0
                H 1 0 0
                H -0.5 0.5 0
            """,
            basis = '6-31g',
        )
        mf = mol.RKS(xc = 'r2scan').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.atom_grid = (3,6)
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        dm = dm.copy() # Remove attached fields
        assert getattr(dm, "mo_coeff", None) is None
        assert getattr(dm, "mo_occ", None) is None

        test_rho = mf._numint.get_rho_with_derivatives(mf.mol, dm, mf.grids, "PBE")
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_with_derivatives_naive(mol, dm, mf.grids, "PBE")

        assert test_rho.shape == (1, 4, mf.grids.coords.shape[0])
        assert  ref_rho.shape == (1, 4, mf.grids.coords.shape[0])
        assert np.max(np.abs(test_rho - ref_rho)) < 1e-11

    def test_get_rho_with_derivatives_unrestricted_mo_input(self):
        assert mol.spin == 1
        mf = mol.UKS(xc = 'LDA').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.level = 0
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        assert dm.shape == (2, mol.nao, mol.nao)
        assert getattr(dm, "mo_coeff", None) is not None
        assert getattr(dm, "mo_occ", None) is not None

        test_rho = mf._numint.get_rho_with_derivatives(mf.mol, dm, mf.grids)
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_with_derivatives_naive(mol, dm, mf.grids)

        assert test_rho.shape == (2, 5, mf.grids.coords.shape[0])
        assert  ref_rho.shape == (2, 5, mf.grids.coords.shape[0])
        assert np.max(np.abs(test_rho - ref_rho)) < 1e-11

    @unittest.skipIf(dftd4 is None, "dftd4 not available")
    def test_get_rho_with_derivatives_unrestricted_dm_input(self):
        assert mol.spin == 1
        mf = mol.UKS(xc = 'wB97X-d4').density_fit(auxbasis = "def2-universal-jkfit").to_gpu()
        mf.grids.level = 3
        mf.conv_tol = 1e-10
        mf.kernel()
        assert mf.converged

        dm = mf.make_rdm1()
        dm = dm.copy() # Remove attached fields
        assert dm.shape == (2, mol.nao, mol.nao)
        assert getattr(dm, "mo_coeff", None) is None
        assert getattr(dm, "mo_occ", None) is None

        test_rho = mf._numint.get_rho_with_derivatives(mf.mol, dm, mf.grids)
        test_rho = test_rho.get()

        ref_rho = numint.get_rho_with_derivatives_naive(mol, dm, mf.grids)

        assert test_rho.shape == (2, 5, mf.grids.coords.shape[0])
        assert  ref_rho.shape == (2, 5, mf.grids.coords.shape[0])
        assert np.max(np.abs(test_rho - ref_rho)) < 1e-11

    def test_ud2ts(self):
        matrix = cp.array([[0.5, 0.5],
                           [0.5, -0.5]])
        v_ud = cp.random.rand(2,4,10)
        ref = cp.einsum('ra,axg->rxg', matrix, v_ud)
        assert abs(xc_deriv.ud2ts(v_ud) - ref).max() < 1e-14

        v_ud = cp.random.rand(2,4,2,4,10)
        ref = cp.einsum('ra,tb,axbyg->rxtyg', matrix, matrix, v_ud)
        assert abs(xc_deriv.ud2ts(v_ud) - ref).max() < 1e-14

        v_ud = cp.random.rand(2,4,2,4,2,4,10)
        ref = cp.einsum('ra,tb,sc,axbyczg->rxtyszg', matrix, matrix, matrix, v_ud)
        assert abs(xc_deriv.ud2ts(v_ud) - ref).max() < 1e-14

    def test_eval_xc_eff_limited_memory(self):
        ni_cpu = pyscf_numint()
        ni_gpu = NumInt()
        ngrids = 20000
        with lib.temporary_env(numint, MEMPOOL_THRESHOLD=80000):
            cp.random.seed(2)
            rho = cp.random.rand(ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('lda', rho, deriv=1)
            ref = ni_cpu.eval_xc_eff('lda', rho.get(), deriv=1)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14

            rho = cp.random.rand(2, ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('lda', rho, deriv=2)
            ref = ni_cpu.eval_xc_eff('lda', rho.get(), deriv=2)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14
            assert abs(dat[2].get() - ref[2]).max() < 1e-14

            rho = cp.random.rand(4, ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('pbe', rho, deriv=1)
            ref = ni_cpu.eval_xc_eff('pbe', rho.get(), deriv=1)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14

            rho = cp.random.rand(2, 4, ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('pbe', rho, deriv=2)
            ref = ni_cpu.eval_xc_eff('pbe', rho.get(), deriv=2)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14
            assert abs(dat[2].get() - ref[2]).max() < 1e-14

            rho = cp.random.rand(5, ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('r2scan', rho, deriv=1)
            ref = ni_cpu.eval_xc_eff('r2scan', rho.get(), deriv=1)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14

            rho = cp.random.rand(2, 5, ngrids) * 1e-1 + .5
            dat = ni_gpu.eval_xc_eff('r2scan', rho, deriv=2)
            ref = ni_cpu.eval_xc_eff('r2scan', rho.get(), deriv=2)
            assert abs(dat[1].get() - ref[1]).max() < 1e-14
            assert abs(dat[2].get() - ref[2]).max() < 1e-14

    def test_xc_functional_classification(self):
        ni = NumInt()
        cpu = pyscf_numint()
        self.assertEqual(ni.is_hybrid_xc(None), cpu.libxc.is_hybrid_xc(None))
        for xc in ('PBE', 'PBE0', 'HSE06', 'WB97M_V'):
            with self.subTest(xc=xc):
                self.assertEqual(ni.is_hybrid_xc(xc), cpu.libxc.is_hybrid_xc(xc))
                self.assertEqual(ni.is_nlc(xc), cpu.libxc.is_nlc(xc))

    def test_eval_xc_hf(self):
        coords = np.random.default_rng(17).uniform(-2., 2., (13, 3))
        ao = pyscf_numint().eval_ao(mol, coords)
        values = ao.dot(mo_coeff[0][:, 1:4])
        density = cp.asarray(np.sum(values**2, axis=1))
        before = density.copy()
        # Exact exchange has no semilocal XC contribution, at any density.
        result = NumInt().eval_xc_eff('HF', density, deriv=3)
        reference = pyscf_numint().eval_xc_eff('HF', density.get(), deriv=3)
        self.assertEqual(len(result), len(reference))
        for actual, expected in zip(result, reference):
            np.testing.assert_array_equal(actual.get(), expected)
        np.testing.assert_array_equal(density.get(), before.get())


class GroupedIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coords = np.random.default_rng(17).uniform(-2., 2., (13, 3))
        cls.ao_cpu = mol.eval_gto('GTOval_sph_deriv1', coords)
        # Reuse converged orbitals; no additional SCF or integration grid needed.
        cls.coeff = mo_coeff[0][:, 1:7].copy()

    def _check_eval_rho_group(self, xctype):
        ao_cpu = self.ao_cpu[0] if xctype == 'LDA' else self.ao_cpu
        occ = cp.asarray([1.25, .5, 0., .25, 0., 0.])
        ao_group, coeff_group, refs = [], [], []
        ni_cpu = pyscf_numint()
        for p0, p1, indices in ((0, 5, np.arange(mol.nao)),
                                (5, 13, np.arange(1, mol.nao, 2))):
            ao_block = ao_cpu[..., p0:p1, :]
            ao_group.append(cp.asarray(ao_block[..., indices].swapaxes(-1, -2), order='C'))
            coeff_group.append(cp.asarray(self.coeff[indices], order='C'))
            # Zero padding gives an independent full-basis reference for AO screening.
            coeff_ref = np.zeros_like(self.coeff)
            coeff_ref[indices] = self.coeff[indices]
            refs.append(ni_cpu.eval_rho2(mol, ao_block, coeff_ref, occ.get(),
                                         xctype=xctype, with_lapl=False))
        inputs = ao_group + coeff_group + [occ]
        saved = [a.copy() for a in inputs]
        grouped = numint.eval_rho_group(mol, ao_group, coeff_group, occ, xctype=xctype)
        self.assertEqual(len(grouped), 2)
        ungrouped = []
        for ao, coeff, actual, ref in zip(ao_group, coeff_group, grouped, refs):
            self.assertEqual(actual.shape, ref.shape)
            np.testing.assert_allclose(actual.get(), ref, atol=1e-12, rtol=1e-12)
            ungrouped.append(numint.eval_rho2(mol, ao, coeff, occ, xctype=xctype).get())
        np.testing.assert_allclose(np.concatenate([a.get() for a in grouped], axis=-1),
                                   np.concatenate(ungrouped, axis=-1), atol=1e-12, rtol=1e-12)
        for actual, before in zip(inputs, saved):
            np.testing.assert_array_equal(actual.get(), before.get())

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_eval_rho_group_lda(self):
        self._check_eval_rho_group('LDA')

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_eval_rho_group_gga(self):
        self._check_eval_rho_group('GGA')

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_eval_rho_group_mgga(self):
        self._check_eval_rho_group('MGGA')

    def _check_nr_rks_group(self, xc):
        plain = cp.asarray(dm0)
        tagged = tag_array(plain.copy(), mo_coeff=cp.asarray(mo_coeff[0]),
                           mo_occ=cp.asarray(mo_occ[0]))
        host_tagged = lib.tag_array(dm0.copy(), mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
        batch = cp.stack([plain, .7 * plain])
        ni = NumInt()
        refs = [pyscf_numint().nr_rks(mol, grids_cpu, xc, dm0 * scale)
                for scale in (1., .7)]
        ordinary = [NumInt().nr_rks(mol, grids_gpu, xc, plain * scale)
                    for scale in (1., .7)]
        inputs = [('plain', plain), ('tagged', tagged), ('batch', batch),
                  ('host', dm0.copy()), ('host-tagged', host_tagged)]
        for name, density in inputs:
            with self.subTest(input=name):
                before = density.copy()
                actual = numint.nr_rks_group(ni, mol, grids_gpu, xc, density)
                expected = refs[0] if name != 'batch' else tuple(
                    np.stack([ref[i] for ref in refs]) for i in range(3))
                expected_gpu = ordinary[0] if name != 'batch' else tuple(
                    cp.stack([cp.asarray(ref[i]) for ref in ordinary]) for i in range(3))
                for value, ref, gpu_ref in zip(actual, expected, expected_gpu):
                    self.assertEqual(value.shape, np.shape(ref))
                    np.testing.assert_allclose(value.get(), ref, atol=1e-9, rtol=1e-10)
                    np.testing.assert_allclose(value.get(), cp.asnumpy(gpu_ref),
                                               atol=1e-10, rtol=1e-10)
                np.testing.assert_array_equal(cp.asnumpy(density), cp.asnumpy(before))
        for density in (tagged, host_tagged):
            np.testing.assert_array_equal(cp.asnumpy(density.mo_coeff), mo_coeff[0])
            np.testing.assert_array_equal(cp.asnumpy(density.mo_occ), mo_occ[0])

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_nr_rks_group_lda(self):
        self._check_nr_rks_group('LDA,VWN')

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_nr_rks_group_gga(self):
        self._check_nr_rks_group('PBE')

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_nr_rks_group_mgga(self):
        self._check_nr_rks_group('r2scan')

    def _check_nr_rks_group_nonsymmetric_dm(self, xc):
        # A real antisymmetric addition leaves the physical density unchanged.
        cross = np.outer(mo_coeff[0][:, 1], mo_coeff[0][:, 2])
        density = cp.asarray(dm0 + .05 * (cross - cross.T))
        before = density.copy()
        actual = numint.nr_rks_group(NumInt(), mol, grids_gpu, xc, density, hermi=0)
        ref = pyscf_numint().nr_rks(mol, grids_cpu, xc, density.get(), hermi=0)
        ordinary = NumInt().nr_rks(mol, grids_gpu, xc, density, hermi=0)
        for value, cpu_ref, gpu_ref in zip(actual, ref, ordinary):
            np.testing.assert_allclose(value.get(), cpu_ref, atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(value.get(), cp.asnumpy(gpu_ref),
                                       atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(density.get(), before.get())

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_nr_rks_group_nonsymmetric_dm_gga(self):
        self._check_nr_rks_group_nonsymmetric_dm('PBE')

    @unittest.skipUnless(HAS_GROUPED_GEMM, "grouped GEMM kernels not available")
    def test_nr_rks_group_nonsymmetric_dm_mgga(self):
        self._check_nr_rks_group_nonsymmetric_dm('r2scan')


class IntegrationProtocols(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.molecule = pyscf.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr',
                               basis='6-31g', verbose=0)
        eigenvalues, vectors = np.linalg.eigh(cls.molecule.intor_symmetric('int1e_ovlp'))
        coeff = vectors / np.sqrt(eigenvalues)
        cls.coeff = coeff[:, np.r_[len(eigenvalues)-1, np.arange(len(eigenvalues)-1)]]
        coords = np.random.default_rng(59).uniform(-.8, .8, (29, 3))
        cls.grid = gen_grid.Grids(cls.molecule)
        cls.grid.coords = cp.asarray(coords)
        cls.grid.weights = cp.asarray(np.random.default_rng(60).uniform(.02, .08, len(coords)))
        cls.grid_cpu = cls.grid.to_cpu()
        # Conversion resets derived quadrature arrays; keep the reference on
        # these exact custom points rather than rebuilding a default grid.
        cls.grid_cpu.coords = coords.copy()
        cls.grid_cpu.weights = cls.grid.weights.get()

    def test_xc_workspace_gga_restricted(self):
        density = np.zeros((4, 137))
        density[0] = np.linspace(.7, 1.1, 137)
        density[1:4] = np.array([.03, -.02, .01])[:, None]
        # Include a genuine noncontiguous density view.
        storage = cp.zeros(density.shape[:-1] + (274,))
        storage[..., ::2] = cp.asarray(density)
        gpu_density = storage[..., ::2]
        reference = pyscf_numint().eval_xc_eff('PBE', density, deriv=2, spin=0)
        for columns in (31, 64):
            with self.subTest(workspace_columns=columns):
                work = cp.full((1, columns), 7.25)
                result = NumInt().eval_xc_eff('PBE', gpu_density, deriv=2,
                                             spin=0, work=work)
                for actual, expected in zip(result[:3], reference[:3]):
                    np.testing.assert_allclose(actual.get(), expected,
                                               atol=2e-11, rtol=2e-11)
                np.testing.assert_array_equal(gpu_density.get(), density)
                if columns < 64:
                    np.testing.assert_array_equal(work.get(), 7.25)

    def _check_response_orbitals(self, xc, unrestricted):
        molecule, grid, grid_cpu = self.molecule, self.grid, self.grid_cpu
        c = self.coeff
        occ = np.zeros(c.shape[1])
        occ[0] = 2
        mo1 = np.stack([.02*c[:, 1:2], -.03*c[:, 2:3]])
        if unrestricted:
            beta = c.copy()
            beta[:, 0] = np.cos(.15)*c[:, 0] + np.sin(.15)*c[:, 1]
            beta[:, 1] = -np.sin(.15)*c[:, 0] + np.cos(.15)*c[:, 1]
            coeffs = np.stack([c, beta])
            occupations = np.stack([occ*.5, occ*.5])
            occupied = (c[:, :1], beta[:, :1])
            first_order = (mo1, np.stack([-.025*beta[:, 1:2], .015*beta[:, 2:3]]))
        else:
            coeffs, occupations = c, occ
            occupied, first_order = c[:, :1], mo1
        cpu = pyscf_numint()
        rho, vxc, fxc = cpu.cache_xc_kernel(molecule, grid_cpu, xc, coeffs,
                                           occupations, spin=int(unrestricted))
        reference_dm = np.einsum('...pi,...i,...qi->...pq', coeffs, occupations, coeffs)
        method = 'nr_uks_fxc' if unrestricted else 'nr_rks_fxc'
        for hermi in (0, 1):
            with self.subTest(xc=xc, unrestricted=unrestricted, hermi=hermi):
                if unrestricted:
                    transition = np.stack([np.einsum('pi,nqi->npq', a, b)
                                           for a, b in zip(occupied, first_order)])
                else:
                    transition = 2*np.einsum('pi,nqi->npq', occupied, first_order)
                if hermi:
                    transition += transition.swapaxes(-1, -2)
                tagged = tag_array(cp.asarray(transition),
                                   mo1=tuple(cp.asarray(x) for x in first_order) if unrestricted else cp.asarray(first_order),
                                   occ_coeff=tuple(cp.asarray(x) for x in occupied) if unrestricted else cp.asarray(occupied))
                expected = getattr(cpu, method)(molecule, grid_cpu, xc, dm0=reference_dm,
                                                dms=transition, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)
                # A fresh NumInt must initialize its AO optimizer even though
                # the caller provides an independently computed kernel.
                ni = NumInt()
                actual = getattr(ni, method)(molecule, grid, xc, dms=tagged,
                                             fxc=cp.asarray(fxc), hermi=hermi)
                with lib.temporary_env(numint, FREE_CUPY_CACHE=True):
                    ordinary = getattr(ni, method)(molecule, grid, xc, dms=cp.asarray(transition),
                                                   fxc=cp.asarray(fxc), hermi=hermi)
                np.testing.assert_allclose(actual.get(), expected, atol=2e-11, rtol=2e-10)
                np.testing.assert_allclose(actual.get(), ordinary.get(), atol=2e-11, rtol=2e-10)
                np.testing.assert_array_equal(tagged.get(), transition)
                if not unrestricted and hermi and xc == 'PBE':
                    # Differentiate potentials along actual orbital variations,
                    # keeping both displaced density matrices positive.
                    step = 1e-3
                    matrices = []
                    for sign in (1, -1):
                        perturbed = occupied + sign*step*first_order[0]
                        matrices.append(cpu.nr_rks(molecule, grid_cpu, xc,
                                                   2*perturbed@perturbed.T)[2])
                    np.testing.assert_allclose((matrices[0]-matrices[1])/(2*step),
                                               actual[0].get(), atol=2e-9, rtol=2e-7)

    def test_rks_first_order_orbital_response_lda(self):
        self._check_response_orbitals('LDA,VWN', False)

    def test_uks_first_order_orbital_response_gga(self):
        self._check_response_orbitals('PBE', True)

    def test_response_requires_initialized_kernel(self):
        ni = NumInt()
        for method in ('nr_rks_fxc', 'nr_uks_fxc', 'nr_rks_fxc_st'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(RuntimeError, 'fxc was not initialized'):
                    getattr(ni, method)(self.molecule, self.grid, 'PBE')

    def test_get_rho_with_derivatives_without_build(self):
        density = 2*self.coeff[:, :1]@self.coeff[:, :1].T
        for xc, xctype in [('LDA,VWN', 'LDA'), ('PBE', 'GGA'), ('r2scan', 'MGGA')]:
            with self.subTest(xc=xc):
                ao = pyscf_numint().eval_ao(self.molecule, self.grid_cpu.coords,
                                           deriv=0 if xctype == 'LDA' else 1)
                expected = pyscf_numint().eval_rho(self.molecule, ao, density,
                                                  xctype=xctype, hermi=1, with_lapl=False)
                ni = NumInt()
                actual = ni.get_rho_with_derivatives(self.molecule, cp.asarray(density),
                                                     self.grid, xc=xc)
                np.testing.assert_allclose(actual[0].get(), expected.reshape(-1, len(self.grid.weights)),
                                           atol=2e-12, rtol=2e-12)

    def test_grouped_blocks_build_real_quadrature_and_optimizer(self):
        grids = gen_grid.Grids(self.molecule)
        grids.atom_grid = (10, 14)
        batches = list(numint._grouped_block_loop(NumInt(), self.molecule, grids, deriv=0))
        blocks = [tuple(parts) for batch in batches for parts in zip(*batch)]
        self.assertEqual(sum(len(x[2]) for x in blocks), grids.size)
        for ao, indices, weights, coords in blocks:
            # Sorted basis functions are related to original AOs by the genuine
            # optimizer coefficient matrix, independently of grouped evaluation.
            opt = numint._GDFTOpt.from_mol(self.molecule)
            expected = self.molecule.eval_gto('GTOval_sph', coords.get()).T
            expected = opt.sort_orbitals(cp.asarray(expected), axis=[0])[indices]
            np.testing.assert_allclose(ao.get(), expected.get(), atol=2e-13, rtol=2e-12)
            self.assertEqual(ao.shape[1], len(weights))

    def test_density_contractions_accept_strided_ao_views(self):
        ao = NumInt().eval_ao(self.molecule, self.grid.coords, deriv=1, transpose=False)
        bra, ket = ao[0, :, ::2], ao[1, :, ::2]
        expected = np.einsum('ig,ig->g', bra.get(), ket.get())
        np.testing.assert_allclose(numint._contract_rho(bra, ket).get(), expected, atol=2e-13)
        contiguous_bra, contiguous_ket = cp.ascontiguousarray(bra), cp.ascontiguousarray(ket)
        np.testing.assert_allclose(numint._contract_rho1(contiguous_bra, contiguous_ket).get(),
                                   expected[None], atol=2e-13)


if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
