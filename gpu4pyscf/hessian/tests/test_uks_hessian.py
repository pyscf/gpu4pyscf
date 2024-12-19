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
import numpy
from pyscf import gto, scf, lib, dft
from pyscf import grad, hessian
from pyscf.hessian import uks as uks_cpu
from gpu4pyscf.hessian import uks as uks_gpu

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
    mol.spin = 1
    mol.charge = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_vxc(method, xc='LDA'):
    mf = dft.uks.UKS(mol, xc=xc)
    mf.conv_tol = 1e-14
    mf.grids.level = 1
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
        mf = mol.UKS(xc='LDA').run()
        _vs_cpu(mf)

    def test_hessian_gga(self):
        print('-----testing PBE Hessian----')
        mf = mol.UKS(xc='PBE').run()
        _vs_cpu(mf, tol=1e-6)

    def test_hessian_hybrid(self):
        print('-----testing B3LYP Hessian----')
        mf = mol.UKS(xc='b3lyp').run()
        _vs_cpu(mf, tol=1e-6)

    def test_hessian_mgga(self):
        print('-----testing M06 Hessian----')
        mf = mol.UKS(xc='m06').run()
        _vs_cpu(mf)

    def test_hessian_rsh(self):
        print('-----testing wb97 Hessian----')
        mf = mol.UKS(xc='wb97').run()
        _vs_cpu(mf)

if __name__ == "__main__":
    print("Full Tests for UKS Hessian")
    unittest.main()
