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
from pyscf import lib
from pyscf.dft import Grids
from gpu4pyscf.dft.numint import NumInt, _GDFTOpt

mol = pyscf.M(
    atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
''',
    basis='ccpvtz',
    spin=None,
)

np.random.seed(2)
nao = mol.nao
dm = np.random.random((2,nao,nao))
dm1 = dm + dm.transpose(0,2,1)
mo_coeff = np.random.rand(nao, nao)
mo_occ = (np.random.rand(nao) > .5).astype(np.double)
dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)
grids = Grids(mol).build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def _check_vxc(self, method, xc, fpref):
        ni = NumInt()
        fn = getattr(ni, method)
        ni.device = 'gpu'
        e, n, v = fn(mol, grids, xc, dm1, hermi=1)
        self.assertAlmostEqual(lib.fp(v), fpref, 9)
        ni.device = 'cpu'
        eref, nref, vref = fn(mol, grids, xc, dm1, hermi=1)
        self.assertAlmostEqual(abs(e - eref).max(), 0, 10)
        self.assertAlmostEqual(abs(n - nref).max(), 0, 10)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 9)

    def _check_rks_fxc(self, xc, fpref, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        ni = NumInt()
        spin = 0
        ni.device = 'gpu'
        rho, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo_coeff, mo_occ, spin)
        v = ni.nr_rks_fxc(mol, grids, xc, dms=t1, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(lib.fp(v), fpref, 9)
        ni.device = 'cpu'
        rho, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo_coeff, mo_occ, spin)
        vref = ni.nr_rks_fxc(
            mol, grids, xc, dm0=dm0, dms=t1, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 9)

    def _check_rks_fxc_st(self, xc, fpref):
        ni = NumInt()
        spin = 1
        ni.device = 'gpu'
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, xc, [mo_coeff]*2, [mo_occ*.5]*2, spin)
        v = ni.nr_rks_fxc_st(mol, grids, xc, dms_alpha=dm, fxc=fxc)
        self.assertAlmostEqual(lib.fp(v), fpref, 9)
        ni.device = 'cpu'
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, xc, [mo_coeff]*2, [mo_occ*.5]*2, spin)
        vref = ni.nr_rks_fxc_st(
            mol, grids, xc, dm0=dm0, dms_alpha=dm, rho0=rho, vxc=vxc, fxc=fxc)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 9)

    def _check_uks_fxc(self, xc, fpref, hermi=1):
        if hermi == 1:
            t1 = dm1
        else:
            t1 = dm
        ni = NumInt()
        spin = 1
        ni.device = 'gpu'
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, xc, [mo_coeff]*2, [mo_occ, 1-mo_occ], spin)
        v = ni.nr_uks_fxc(mol, grids, xc, dms=t1, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(lib.fp(v), fpref, 8)
        ni.device = 'cpu'
        dm0 = mo_coeff.dot(mo_coeff.T)
        rho, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, xc, [mo_coeff]*2, [mo_occ, 1-mo_occ], spin)
        vref = ni.nr_uks_fxc(
            mol, grids, xc, dm0=dm0, dms=t1, rho0=rho, vxc=vxc, fxc=fxc, hermi=hermi)
        self.assertAlmostEqual(abs(v - vref).max(), 0, 8)

    def test_rks_lda(self):
        self._check_vxc('nr_rks', 'lda,', -5.592159200616021)

    def test_rks_gga(self):
        self._check_vxc('nr_rks', 'pbe,', -5.962921669162233)

    def test_rks_mgga(self):
        self._check_vxc('nr_rks', 'm06,', 117.50826508660953)

    def test_uks_lda(self):
        self._check_vxc('nr_uks', 'lda,', -6.362059440515177)

    def test_uks_gga(self):
        self._check_vxc('nr_uks', 'pbe,', -6.732546841646528)

    def test_uks_mgga(self):
        self._check_vxc('nr_uks', 'm06,', 83.5606316500255)

    def test_rks_fxc_lda(self):
        self._check_rks_fxc('lda,', -0.06358425564270617, hermi=1)
        self._check_rks_fxc('lda,', -0.03179212782135285, hermi=0)

    def test_rks_fxc_gga(self):
        self._check_rks_fxc('pbe,', -0.006650911990898234, hermi=1)
        self._check_rks_fxc('pbe,', -0.0033254559954490615, hermi=0)

    def test_rks_fxc_mgga(self):
        self._check_rks_fxc('m06,', 1.2456987899337146, hermi=1)
        self._check_rks_fxc('m06,', 0.6228493949668574, hermi=0)

    def test_uks_fxc_lda(self):
        self._check_uks_fxc('lda,', -0.1125902447294953, hermi=1)
        self._check_uks_fxc('lda,', -0.05629512236474782, hermi=0)

    def test_uks_fxc_gga(self):
        self._check_uks_fxc('pbe,', -0.0752237471800169, hermi=1)
        self._check_uks_fxc('pbe,', -0.03761187359000853, hermi=0)

    def test_uks_fxc_mgga(self):
        self._check_uks_fxc('m06,', 0.7005336565753997, hermi=1)
        self._check_uks_fxc('m06,', 0.35026682828770006, hermi=0)

    def test_rks_fxc_st_lda(self):
        self._check_rks_fxc_st('lda,', -0.06358425564270553)

    def test_rks_fxc_st_gga(self):
        self._check_rks_fxc_st('pbe,', -0.006650911990898234)

    def test_rks_fxc_st_mgga(self):
        self._check_rks_fxc_st('m06,', 1.2456987899337242)

    def test_gdftopt(self):
        mol = pyscf.M(
            atom='He',
            basis=[
                [0, [1, 1]],
                [0, [2, 1]],
                [0, [.5, 1]],
                [0, [3, 1], [4, 1]],
                [1, [1, 1]],
                [1, [2, 1]],
                [1, [3, 1]],
                [2, [1, 1]],
            ])
        opt = _GDFTOpt.from_mol(mol)
        self.assertEqual(opt.coeff.shape, (40, 18))
        self.assertTrue(all(opt.coeff.max(axis=1)[13:16] == 0))
        self.assertTrue(all(opt.coeff.max(axis=1)[22:] == 0))
        self.assertEqual(opt.mol.nbas, 12)
        self.assertTrue(all(opt.mol._bas[::4, 1] == [0, 1, 2]))
        self.assertTrue(all(opt.l_bas_offsets == [0, 4, 8, 12]))
        self.assertTrue(all(opt.l_ctr_offsets == [0, 3, 4, 8, 12]))


if __name__ == "__main__":
    print("Full Tests for dft numint")
    unittest.main()
