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
import pyscf
import cupy
from pyscf import lib, scf
from pyscf.dft import Grids
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft

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

    grids_cpu = Grids(mol)
    grids_cpu.level = 1
    grids_cpu.build()

    grids_gpu = Grids(mol)
    grids_gpu.level = 1
    grids_gpu.build()

    grids_gpu.weights = cupy.asarray(grids_gpu.weights)
    grids_gpu.coords = cupy.asarray(grids_gpu.coords)

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
        rho = np.random.random((4,20))
        coords = (np.random.random((20,3))-.5)*3
        vvrho = np.random.random((4,60))
        vvweight = np.random.random(60)
        vvcoords = (np.random.random((60,3))-.5)*3
        nlc_pars = .8, .3

        rho = cupy.asarray(rho)
        coords = cupy.asarray(coords)
        vvrho = cupy.asarray(vvrho)
        vvweight = cupy.asarray(vvweight)
        vvcoords = cupy.asarray(vvcoords)

        v = dft.numint._vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars)
        self.assertAlmostEqual(lib.fp(v[0].get()), 0.15894647203764295, 8)
        self.assertAlmostEqual(lib.fp(v[1].get()), 0.20500922537924576, 8)

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
            self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 10)

            rho = ni_gpu.eval_rho(mol, ao_gpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            self.assertAlmostEqual(abs(rho.get() - ref).max(), 0, 10)

if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
