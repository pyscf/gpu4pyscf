# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import pyscf
import cupy
from pyscf import lib
from pyscf.dft import Grids
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft

mol = pyscf.M(
    atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
''',
    basis='ccpvtz',
    spin=None,
)
mol.verbose=1
np.random.seed(2)
nao = mol.nao
dm = np.random.random((2,nao,nao))
dm1 = dm + dm.transpose(0,2,1)
mo_coeff = np.random.rand(nao, nao)
mo_occ = (np.random.rand(nao) > .5).astype(np.double)
dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)

grids_cpu = Grids(mol)
grids_cpu.level = 5
grids_cpu.build()

grids_gpu = Grids(mol)
grids_gpu.level = 5
grids_gpu.build()

grids_gpu.weights = cupy.asarray(grids_gpu.weights)
grids_gpu.coords = cupy.asarray(grids_gpu.coords)

LDA = 'LDA_C_VWN'
GGA_PBE = 'GGA_C_PBE'
MGGA_M06 = 'MGGA_C_M06'

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):

    def _check_vxc(self, method, xc):
        ni = NumInt(xc=xc)
        fn = getattr(ni, method)
        n, e, v = fn(mol, grids_gpu, xc, dm0, hermi=1)
        v = [i.get() for i in v]

        ni_pyscf = pyscf_numint()
        fn = getattr(ni_pyscf, method)
        nref, eref, vref = fn(mol, grids_cpu, xc, dm0, hermi=1)

        cupy.allclose(e, eref)
        cupy.allclose(n, nref)
        cupy.allclose(v, vref)

    def _check_rks_fxc(self, xc, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        spin = 0
        ni_pyscf = pyscf_numint()
        rho, vxc, fxc = ni_pyscf.cache_xc_kernel(mol, grids_cpu, xc, mo_coeff, mo_occ, spin)
        vref = ni_pyscf.nr_rks_fxc(
            mol, grids_cpu, xc, dm0=dm0, dms=t1, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)

        rho0 = rho.copy()
        vxc0 = vxc.copy()
        fxc0 = fxc.copy()
        ni = NumInt()
        rho, vxc, fxc = ni.cache_xc_kernel(mol, grids_gpu, xc, cupy.asarray(mo_coeff), cupy.asarray(mo_occ), spin)
        v = ni.nr_rks_fxc(mol, grids_gpu, xc, dms=t1, fxc=fxc, hermi=hermi)
        if xc == MGGA_M06:
            assert cupy.allclose(rho[[0,1,2,3,5]], rho0[[0,1,2,3,5]])
        else:
            assert cupy.allclose(rho, rho0)

        assert cupy.linalg.norm(vxc - cupy.asarray(vxc0)) < 1e-6 * cupy.linalg.norm(vxc)
        assert cupy.linalg.norm(fxc - cupy.asarray(fxc0)) < 1e-6 * cupy.linalg.norm(fxc)
        assert cupy.allclose(v, vref)

    def _check_rks_fxc_st(self, xc, fpref):
        ni = NumInt()
        spin = 1
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids_gpu, xc, [mo_coeff]*2, [mo_occ*.5]*2, spin)
        v = ni.nr_rks_fxc_st(mol, grids_gpu, xc, dms_alpha=dm, fxc=fxc)
        self.assertAlmostEqual(lib.fp(v), fpref, 12)

        ni_pyscf = pyscf_numint()
        rho, vxc, fxc = ni_pyscf.cache_xc_kernel(
            mol, grids_cpu, xc, [mo_coeff]*2, [mo_occ*.5]*2, spin)
        vref = ni_pyscf.nr_rks_fxc_st(
            mol, grids_cpu, xc, dm0=dm0, dms_alpha=dm, rho0=rho, vxc=vxc, fxc=fxc)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 12)

    def _check_uks_fxc(self, xc, fpref, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        ni = NumInt()
        spin = 1
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids_gpu, xc, [mo_coeff]*2, [mo_occ, 1-mo_occ], spin)
        v = ni.nr_uks_fxc(mol, grids_gpu, xc, dms=t1, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(lib.fp(v), fpref, 12)

        ni = ni.to_cpu()
        dm0 = mo_coeff.dot(mo_coeff.T)
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids_cpu, xc, [mo_coeff]*2, [mo_occ, 1-mo_occ], spin)
        vref = ni.nr_uks_fxc(
            mol, grids_cpu, xc, dm0=dm0, dms=t1, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 12)

    def test_rks_lda(self):
        self._check_vxc('nr_rks', LDA)

    def test_rks_gga(self):
        self._check_vxc('nr_rks', GGA_PBE)

    def test_rks_mgga(self):
        self._check_vxc('nr_rks', MGGA_M06)

    # Not implemented yet
    '''
    def test_uks_lda(self):
        self._check_vxc('nr_uks', 'lda', -6.362059440515177)

    def test_uks_gga(self):
        self._check_vxc('nr_uks', 'pbe', -6.732546841646528)

    def test_uks_mgga(self):
        self._check_vxc('nr_uks', 'm06', 83.5606316500255)
    '''

    def test_rks_fxc_lda(self):
        self._check_rks_fxc(LDA, hermi=1)

    def test_rks_fxc_gga(self):
        self._check_rks_fxc(GGA_PBE, hermi=1)

    def test_rks_fxc_mgga(self):
        self._check_rks_fxc(MGGA_M06, hermi=1)

    '''
    def test_uks_fxc_lda(self):
        self._check_uks_fxc('lda', -0.1125902447294953, hermi=1)
        self._check_uks_fxc('lda', -0.05629512236474782, hermi=0)

    def test_uks_fxc_gga(self):
        self._check_uks_fxc('pbe', -0.0752237471800169, hermi=1)
        self._check_uks_fxc('pbe', -0.03761187359000853, hermi=0)

    def test_uks_fxc_mgga(self):
        self._check_uks_fxc('m06', 0.7005336565753997, hermi=1)
        self._check_uks_fxc('m06', 0.35026682828770006, hermi=0)

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
        return

if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
