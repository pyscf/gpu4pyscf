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
from importlib.metadata import version
from packaging.version import Version
from pyscf.dft import xc_deriv
import numpy as np
import pyscf
import cupy
from pyscf import lib, scf
from pyscf.dft.numint2c import NumInt2C as pyscf_numint2c
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft import Grids
from gpu4pyscf.dft import numint2c
from gpu4pyscf.dft import numint
from pyscf.dft import numint2c as pyscf_numint2c_file
from gpu4pyscf.dft.numint2c import NumInt2C
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid
try:
    import mcfun
except ImportError:
    mcfun = None

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
        output = '/dev/null'
        )

    np.random.seed(2)
    mf = scf.GHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1().copy()
    dm = dm1
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)

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

    def test_eval_rho(self):
        np.random.seed(1)
        dm = np.random.random(dm0.shape) + np.random.random(dm0.shape)*1.0j
        dm = dm + dm.conj().T
        ni_gpu = NumInt2C()
        ni_cpu = pyscf_numint2c()
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

    def test_eval_rho2(self):
        np.random.seed(1)
        mo_coeff_test = np.random.random(mo_coeff.shape) + np.random.random(mo_coeff.shape)*1.0j
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        ni_cpu = pyscf_numint2c()
        ni_cpu.collinear='m'
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho2(mol, ao_gpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            ref = ni_cpu.eval_rho2(mol, ao_cpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_get_rho(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'

        np.random.seed(1)
        ni_gpu_1c = NumInt()
        dm_test = np.random.random(dm0.shape) + np.random.random(dm0.shape)*1.0j
        dm_test = dm_test + dm_test.T.conj()

        n2c = dm_test.shape[0]
        nao = n2c//2
        dm_1c_test = dm_test[:nao,:nao] + dm_test[nao:,nao:]
        rho_gpu = ni_gpu.get_rho(mol, dm_test, grids_gpu)
        rho_1c_gpu = ni_gpu_1c.get_rho(mol, dm_1c_test.real, grids_gpu)
        self.assertAlmostEqual(abs(rho_gpu - rho_1c_gpu).max().item(), 0, 10)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_eval_xc_eff(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        ni_cpu = pyscf_numint2c()
        ni_cpu.collinear='m'
        np.random.seed(1)
        dm = dm0*1.0 + dm0 * 0.1j
        dm = dm + dm.T.conj()
        for xc_code in (LDA, GGA_PBE, MGGA_M06):
            n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, xc_code, dm)
            n_cpu, exc_cpu, vmat_cpu = ni_cpu.nr_vxc(mol, grids_cpu, xc_code, dm)
            self.assertAlmostEqual(abs(n_gpu.get() - n_cpu).max(), 0, 10)
            self.assertAlmostEqual(abs(exc_gpu.get() - exc_cpu).max(), 0, 10)
            self.assertAlmostEqual(abs(vmat_gpu.get() - vmat_cpu).max(), 0, 10)

    def test_eval_xc_eff_fp(self):
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        np.random.seed(1)
        dm = dm0*1.0 + dm0 * 0.1j
        dm = dm + dm.T.conj()

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, LDA, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -1.310501342423071).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.20448306536588537-6.75460139752253e-21j)).max(), 0, 10)

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, GGA_PBE, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -0.7237150857425112).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.05446425800187435-4.486282070082083e-21j)).max(), 0, 10)

        n_gpu, exc_gpu, vmat_gpu = ni_gpu.nr_vxc(mol, grids_gpu, MGGA_M06, dm)
        self.assertAlmostEqual(abs(n_gpu.get() - 17.9999262659497).max(), 0, 10)
        self.assertAlmostEqual(abs(exc_gpu.get() - -0.7703982586705045).max(), 0, 10)
        self.assertAlmostEqual(abs(lib.fp(vmat_gpu.get()) 
            - (-0.18688247306409317+7.50400133342109e-20j)).max(), 0, 10)

    def test_mcol_lda_vxc_mat(self):
        xc_code = 'lda,'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001 + np.random.rand(n2c, n2c) * .0001j
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=0, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='LDA', hermi=1, with_lapl=False)
        
        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='LDA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_lda_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_lda_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) - 
            (-9.596802359283691+2.969010922568367e-05j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) - 
            (-9.596802359283691+2.969010922568367e-05j)).max(), 0, 13)
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu - v1_gpu).max().item(), 0, 13)

        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)
            
            v0_cpu = pyscf_numint2c_file._mcol_lda_vxc_mat(mol, ao.transpose(1,0).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)

    def test_mcol_gga_vxc_mat(self):
        xc_code = 'pbe,'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001 + np.random.rand(n2c, n2c) * .001j
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='GGA', hermi=1, with_lapl=False)

        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='GGA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_gga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_gga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test)
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) - 
            (-9.624260408900755+0.0003122100947141664j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) - 
            (-9.624260408900755+0.0003122100947141664j)).max(), 0, 13)
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu - v1_gpu).max().item(), 0, 13)

        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)

            v0_cpu = pyscf_numint2c_file._mcol_gga_vxc_mat(mol, ao.transpose(0,2,1).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)

    def test_mcol_mgga_vxc_mat(self):
        xc_code = 'tpss'

        nao = mol.nao
        n2c = nao * 2
        ao_loc = mol.ao_loc
        np.random.seed(12)
        dm = np.random.rand(n2c, n2c) * .001
        dm += np.eye(n2c)
        dm = dm + dm.T.conj()
        ngrids = 8
        coords = np.random.rand(ngrids,3)
        weight = np.random.rand(ngrids)

        ao = numint.eval_ao(mol, coords, deriv=1, transpose=False)
        rho = numint2c.eval_rho(mol, ao, dm, xctype='MGGA', hermi=1, with_lapl=False)

        # --- GPU Calculations (Always Run) ---
        ni_GPU = NumInt2C()
        ni_GPU.collinear = 'mcol'
        eval_xc_GPU = ni_GPU.mcfun_eval_xc_adapter(xc_code)
        vxc_gpu = eval_xc_GPU(xc_code, rho, deriv=1, xctype='MGGA')[1]
        mask_gpu = cupy.arange(mol.nbas)
        shls_slice = (0, mol.nbas)
        
        v0_gpu = numint2c._mcol_mgga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 0, assemble_spin_components=True)
        v1_gpu = numint2c._mcol_mgga_vxc_mat(mol, ao, cupy.asarray(weight), rho, vxc_gpu.copy(), 
            mask_gpu, shls_slice, ao_loc, 1, assemble_spin_components=True)
        v1_gpu = v1_gpu + v1_gpu.conj().T

        # --- Assertions (Always Run) ---
        # Fingerprint checks (from _fp test) <-- UPDATED
        self.assertAlmostEqual(abs(lib.fp(v1_gpu.get()) 
            - (-11.359687631112195+6.13828986953566e-22j)).max(), 0, 13)
        self.assertAlmostEqual(abs(lib.fp(v0_gpu.get()) 
            - (-11.359687631112195+6.13828986953566e-22j)).max(), 0, 13)
        
        # Internal consistency check
        self.assertAlmostEqual(abs(v0_gpu - v1_gpu).max().item(), 0, 13)
        
        # --- CPU Comparison (Conditional) ---
        if mcfun is not None:
            ni_CPU = pyscf_numint2c()
            ni_CPU.collinear = 'mcol'
            eval_xc_cpu = ni_CPU.mcfun_eval_xc_adapter(xc_code)
            vxc_cpu = eval_xc_cpu(xc_code, rho.get(), deriv=1)[1]
            mask = np.ones((8, mol.nbas), dtype=np.uint8)

            v0_cpu = pyscf_numint2c_file._mcol_mgga_vxc_mat(mol, ao.transpose(0,2,1).get(), weight, 
                rho.get(), vxc_cpu.copy(), mask, shls_slice, ao_loc, 0)
            
            # CPU vs GPU check
            self.assertAlmostEqual(abs(v0_gpu.get() - v0_cpu).max(), 0, 13)


@unittest.skipIf(mcfun is None or Version(version('mcfun')) < Version('0.2.6'),
                 'requires mcfun >= 0.2.6 for the corrected CPU reference')
class MultiCollinearResponse(unittest.TestCase):
    direction = np.array([2., -1., 2.]) / 3

    @staticmethod
    def _adapter(xc, cpu=False, threshold=.99):
        ni = pyscf_numint2c() if cpu else NumInt2C()
        ni.collinear = 'mcol'
        ni.spin_samples = 86
        ni.collinear_samples = 80
        ni.collinear_thrd = threshold
        return ni.mcfun_eval_xc_adapter(xc)

    @classmethod
    def _density(cls, xc, strong=False):
        nvar = 1 if xc == 'LDA,VWN' else 4
        total = np.array([[1.1, .8], [.08, .04], [-.05, -.03], [.02, -.01]])[:nvar]
        spin = np.array([[1.1, .8], [.04, .02], [-.03, -.01], [.01, -.006]])[:nvar]
        spin[0] *= .995 if strong else .3
        rho = np.concatenate([total[None], cls.direction[:, None, None] * spin[None]])
        if not strong and nvar > 1:
            rho[1:, 1:] += np.random.default_rng(17).normal(scale=.005, size=(3, 3, 2))
        return rho

    @staticmethod
    def _evaluate(adapter, xc, rho, deriv):
        return [cupy.asnumpy(a) for a in adapter(xc, cupy.asarray(rho), deriv=deriv)]

    def _check_weak_spin_derivatives_match_mcfun(self, xc):
        rho = self._density(xc)
        gpu, cpu = self._adapter(xc), self._adapter(xc, cpu=True)
        for deriv in range(3):
            with self.subTest(xc=xc, deriv=deriv):
                actual = self._evaluate(gpu, xc, rho, deriv)
                expected = cpu(xc, rho, deriv=deriv)
                self.assertEqual(len(actual), deriv + 1)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)

    def test_weak_spin_derivatives_match_mcfun_lda(self):
        self._check_weak_spin_derivatives_match_mcfun('LDA,VWN')

    def test_weak_spin_derivatives_match_mcfun_gga(self):
        self._check_weak_spin_derivatives_match_mcfun('PBE')

    def test_signed_collinear_pbe_matches_libxc_and_kernel_reciprocity(self):
        xc = 'PBE'
        rho = self._density(xc, strong=True)
        total = rho[0]
        spin = np.einsum('a,axg->xg', self.direction, rho[1:])
        direct = pyscf_numint().eval_xc_eff(
            xc, np.stack([.5*(total+spin), .5*(total-spin)]), deriv=2)
        transform = np.zeros((4, 2))
        transform[0, 0] = 1
        transform[1:, 1] = self.direction
        adapter = self._adapter(xc)
        for deriv in range(3):
            with self.subTest(deriv=deriv):
                actual = self._evaluate(adapter, xc, rho, deriv)
                np.testing.assert_allclose(actual[0], direct[0], atol=2e-13, rtol=2e-13)
                if deriv > 0:
                    projected = np.einsum('as,axg->sxg', transform, actual[1])
                    np.testing.assert_allclose(projected, xc_deriv.ud2ts(direct[1]),
                                               atol=2e-12, rtol=2e-12)
                if deriv > 1:
                    projected = np.einsum('as,axbyg,bt->sxtyg', transform, actual[2], transform)
                    np.testing.assert_allclose(projected, xc_deriv.ud2ts(direct[2]),
                                               atol=2e-10, rtol=2e-11)
                    np.testing.assert_allclose(actual[2], actual[2].transpose(2, 3, 0, 1, 4),
                                               atol=2e-11, rtol=2e-11)
                    # The MCfun collinear-limit transverse kernel is an axis
                    # integral, not the Hessian of just E(t, |m|, grad|m|).
                    z, weights = np.polynomial.legendre.leggauss(80)
                    z, weights = .5*(z+1), .5*weights
                    sampled = np.stack([.5*(total[..., None]+spin[..., None]*z),
                                        .5*(total[..., None]-spin[..., None]*z)])
                    f = pyscf_numint().eval_xc_eff(xc, sampled.reshape(2, 4, -1),
                                                  deriv=2)[2]
                    f = xc_deriv.ud2ts(f)[1, :, 1].reshape(4, 4, rho.shape[-1], 80)
                    reference = np.einsum('xygn,n->xyg', f, weights)
                    perpendicular = np.array([1., 0., -1.]) / np.sqrt(2)
                    projected = np.einsum('a,axbyg,b->xyg', perpendicular,
                                           actual[2][1:, :, 1:], perpendicular)
                    np.testing.assert_allclose(projected, reference, atol=2e-10, rtol=2e-11)

    def _check_mixed_threshold_points_match_individual_routes(self, xc):
        rho = self._density(xc)
        rho[..., 1:] = self._density(xc, strong=True)[..., 1:]
        before = rho.copy()
        adapter = self._adapter(xc)
        spherical = self._adapter(xc, threshold=None)
        actual = self._evaluate(adapter, xc, rho, 2)
        weak = self._evaluate(spherical, xc, rho[..., :1], 2)
        strong = self._evaluate(adapter, xc, rho[..., 1:], 2)
        for result, a, b in zip(actual, weak, strong):
            np.testing.assert_allclose(result, np.concatenate([a, b], axis=-1),
                                       atol=2e-11, rtol=2e-11)
        np.testing.assert_array_equal(rho, before)

    def test_mixed_threshold_points_match_individual_routes_lda(self):
        self._check_mixed_threshold_points_match_individual_routes('LDA,VWN')

    def test_mixed_threshold_points_match_individual_routes_gga(self):
        self._check_mixed_threshold_points_match_individual_routes('PBE')

    def _check_spin_rotation_covariance(self, xc, strong):
        axis = np.array([1., 2., -1.]) / np.sqrt(6)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(.37)*skew + (1-np.cos(.37))*(skew @ skew)
        transform = np.eye(4)
        transform[1:, 1:] = rotation
        adapter = self._adapter(xc)
        rho = self._density(xc, strong=strong)
        values = self._evaluate(adapter, xc, rho, 2)
        rotated = np.einsum('ab,bxg->axg', transform, rho)
        actual = self._evaluate(adapter, xc, rotated, 2)
        expected = [values[0], np.einsum('ab,bxg->axg', transform, values[1]),
                    np.einsum('ab,bxcyg,dc->axdyg', transform, values[2], transform)]
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)

    def test_spin_rotation_covariance_lda_weak_spin(self):
        self._check_spin_rotation_covariance('LDA,VWN', False)

    def test_spin_rotation_covariance_lda_strong_spin(self):
        self._check_spin_rotation_covariance('LDA,VWN', True)

    def test_spin_rotation_covariance_gga_weak_spin(self):
        self._check_spin_rotation_covariance('PBE', False)

    def test_spin_rotation_covariance_gga_strong_spin(self):
        self._check_spin_rotation_covariance('PBE', True)


if __name__ == "__main__":
    print("Full Tests for dft numint2c")
    unittest.main()

