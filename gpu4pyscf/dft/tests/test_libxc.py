# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
from gpu4pyscf.dft.numint import NumInt as numint_gpu
from pyscf.dft.numint import NumInt as numint_cpu
import cupy

def setUpModule():
    global mol, dm1, dm0
    mol = pyscf.M(
        atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
''',
        basis='ccpvtz',
        spin=None,
        output = '/dev/null'
    )
    np.random.seed(2)
    nao = mol.nao
    dm = np.random.random((2,nao,nao))
    dm1 = dm + dm.transpose(0,2,1)
    np.random.seed(1)
    mo_coeff = np.random.rand(nao, nao)
    mo_occ = (np.random.rand(nao) > .5).astype(np.double)
    dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _diff(dat, ref):
    d = dat - ref
    return np.min((abs(d/(ref+1e-300)), abs(d)), axis=0)

class KnownValues(unittest.TestCase):
    def _check_xc(self, xc, spin=0, fxc_tol=1e-10, kxc_tol=1e-10):
        ni_cpu = numint_cpu()
        ni_gpu = numint_gpu()
        xctype = ni_cpu._xc_type(xc)

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1
        grids = Grids(mol).build()
        ao = ni_cpu.eval_ao(mol, grids.coords, ao_deriv)
        rho = ni_cpu.eval_rho(mol, ao, dm0, xctype=xctype)
        if spin != 0:
            rho = (rho, rho)

        exc_cpu, vxc_cpu, fxc_cpu, kxc_cpu = ni_cpu.eval_xc_eff(xc, rho, deriv=2, xctype=xctype)
        exc_gpu, vxc_gpu, fxc_gpu, kxc_gpu = ni_gpu.eval_xc_eff(xc, cupy.array(rho), deriv=2, xctype=xctype)

        assert _diff(exc_gpu[:,0].get(), exc_cpu).max() < 1e-10
        assert _diff(vxc_gpu.get(), vxc_cpu).max() < 1e-10
        if fxc_gpu is not None:
            assert _diff(fxc_gpu.get(), fxc_cpu).max() < fxc_tol
        if kxc_gpu is not None:
            assert _diff(kxc_gpu.get(), kxc_cpu).max() < kxc_tol

    def test_LDA(self):
        self._check_xc('LDA_C_VWN')

    def test_GGA(self):
        self._check_xc('HYB_GGA_XC_B3LYP')
        self._check_xc('GGA_X_B88', fxc_tol=1e-10)
        self._check_xc('GGA_C_PBE', fxc_tol=1e-5)

    def test_mGGA(self):
        self._check_xc('MGGA_C_M06', fxc_tol=1e-5)

    def test_u_LDA(self):
        self._check_xc('LDA_C_VWN', spin=1)

    def test_u_GGA(self):
        # large errors found in B88 for the spin polarized case
        self._check_xc('HYB_GGA_XC_B3LYP', spin=1, fxc_tol=1e-3)
        self._check_xc('GGA_X_B88', spin=1, fxc_tol=1e-1)
        self._check_xc('GGA_C_PBE', spin=1, fxc_tol=1e-5)

    def test_u_mGGA(self):
        self._check_xc('MGGA_C_M06', spin=1, fxc_tol=1e-5)

if __name__ == "__main__":
    print("Full Tests for xc fun")
    unittest.main()
