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
from pyscf.hessian import uks as uks_cpu
from gpu4pyscf.hessian import uks as uks_gpu

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_vxc(method, xc='LDA'):
    mf = dft.uks.UKS(mol, xc=xc)
    mf.conv_tol = 1e-14
    mf.grids_level = 1
    mf.kernel()
    dm = mf.make_rdm1()
    hobj = mf.Hessian()
    fn = getattr(uks_cpu, method)
    vxc_cpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    mf = mf.to_gpu()
    mf.kernel()
    hobj = mf.Hessian()
    fn = getattr(uks_gpu, method)
    vxc_gpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    # outputs of _get_vxc_deriv2 and _get_vxc_deriv1 in GPU4PYSCF are different from PySCF for memory efficiency
    if method == '_get_vxc_deriv2':
        vxc0 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[0], dm[0])
        vxc1 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[1], dm[1])
        vxc_cpu = (vxc0, vxc1)

    if method == '_get_vxc_deriv1':
        mo_occ = mf.mo_occ.get()
        mo_coeff = mf.mo_coeff.get()
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        vxc0 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[0], mocca)
        vxc0 = numpy.einsum('nxiq,ip->nxpq', vxc0, mo_coeff[0])
        vxc1 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[1], moccb)
        vxc1 = numpy.einsum('nxiq,ip->nxpq', vxc1, mo_coeff[1])
        vxc_cpu = (vxc0, vxc1)

    #print(f'testing {method} for {xc}')
    #print(numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()))
    #print(numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()))
    assert numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()) < 1e-6
    assert numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()) < 1e-6

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

if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()