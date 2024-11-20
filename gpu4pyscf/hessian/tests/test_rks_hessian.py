# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
import numpy
from pyscf import gto, scf, lib, dft
from pyscf import grad, hessian
from pyscf.hessian import rks as rks_cpu
from gpu4pyscf.hessian import rks as rks_gpu

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 6
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_vxc(method, xc='LDA'):
    mf = dft.rks.RKS(mol, xc=xc)
    mf.conv_tol = 1e-14
    mf.grids.level = 1
    mf.kernel()
    dm = mf.make_rdm1()
    hobj = mf.Hessian()
    fn = getattr(rks_cpu, method)
    vxc_cpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    mf = mf.to_gpu()
    mf.kernel()
    hobj = mf.Hessian()
    fn = getattr(rks_gpu, method)
    vxc_gpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    # outputs of _get_vxc_deriv2 and _get_vxc_deriv1 in GPU4PYSCF are different from PySCF for memory efficiency
    if method == '_get_vxc_deriv2':
        vxc_cpu = numpy.einsum('nxyij,ij->nxyi', vxc_cpu, dm)

    if method == '_get_vxc_deriv1':
        mo_occ = mf.mo_occ.get()
        mo_coeff = mf.mo_coeff.get()
        mocc = mo_coeff[:,mo_occ>0]
        vxc_cpu = numpy.einsum('nxij,jq->nxiq', vxc_cpu, mocc)
        vxc_cpu = numpy.einsum('nxiq,ip->nxpq', vxc_cpu, mo_coeff)

    #print(f'testing {method} for {xc}')
    assert numpy.linalg.norm(vxc_cpu - vxc_gpu.get()) < 1e-6

def _vs_cpu(mf, tol=1e-7):
    mf.conv_tol_cpscf = 1e-8
    ref = mf.Hessian().kernel()
    hessobj = mf.Hessian().to_gpu()
    hessobj.base.cphf_grids = hessobj.base.grids
    e2_gpu = hessobj.kernel()
    assert abs(ref - e2_gpu).max() < tol

class KnownValues(unittest.TestCase):
    def test_vxc_diag(self):
        _check_vxc('_get_vxc_diag', xc='LDA')
        _check_vxc('_get_vxc_diag', xc='PBE')
        _check_vxc('_get_vxc_diag', xc='TPSS')

    def test_vxc_deriv1(self):
        _check_vxc('_get_vxc_deriv1', xc='LDA')
        _check_vxc('_get_vxc_deriv1', xc='PBE')
        _check_vxc('_get_vxc_deriv1', xc='TPSS')

    def test_vxc_deriv2(self):
        _check_vxc('_get_vxc_deriv2', xc='LDA')
        _check_vxc('_get_vxc_deriv2', xc='PBE')
        _check_vxc('_get_vxc_deriv2', xc='TPSS')

    def test_hessian_lda(self, disp=None):
        print('-----testing LDA Hessian----')
        mf = mol.RKS(xc='LDA').run()
        _vs_cpu(mf)

    def test_hessian_gga(self):
        print('-----testing PBE Hessian----')
        mf = mol.RKS(xc='PBE').run()
        _vs_cpu(mf)

    def test_hessian_hybrid(self):
        print('-----testing B3LYP Hessian----')
        mf = mol.RKS(xc='b3lyp').run()
        _vs_cpu(mf)

    def test_hessian_mgga(self):
        print('-----testing M06 Hessian----')
        mf = mol.RKS(xc='m06').run()
        _vs_cpu(mf, tol=1e-6)

    def test_hessian_rsh(self):
        print('-----testing wb97 Hessian----')
        mf = mol.RKS(xc='wb97').run()
        _vs_cpu(mf)

if __name__ == "__main__":
    print("Full Tests for RKS Hessian")
    unittest.main()
