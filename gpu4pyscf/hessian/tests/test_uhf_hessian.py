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
import cupy
import pyscf
from pyscf import gto, lib
from pyscf import grad, hessian
from pyscf.hessian import uhf as uhf_cpu
from gpu4pyscf import scf
from gpu4pyscf.hessian import uhf as uhf_gpu
from gpu4pyscf.hessian import jk

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.spin = 1
    mol.charge = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_hessian_uhf(self):
        mf = mol.UHF().run()
        mf.conv_tol_cpscf = 1e-8
        hobj = mf.Hessian()
        ref = hobj.kernel()
        e2_gpu = hobj.to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-6

    def test_partial_hess_elec(self):
        mf = pyscf.scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = uhf_cpu._partial_hess_ejk(hobj)
        e2_cpu = ej_cpu - ek_cpu

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, e2_gpu = uhf_gpu._partial_hess_ejk(hobj)

        assert numpy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-7
        assert numpy.linalg.norm(e2_cpu - e2_gpu.get()) < 1e-7

    def test_hessian_uhf_D3(self):
        print('----- testing UHF with D3BJ ------')
        mf = mol.UHF()
        mf.disp = 'd3bj'
        mf.run()
        mf.conv_tol_cpscf = 1e-8
        ref = mf.Hessian().kernel()
        e2_gpu = mf.Hessian().to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-6

    def test_jk_mix(self):
        mol1 = pyscf.M(
            atom='''
        C  -1.20806619, -0.34108413, -0.00755148
        C   1.28636081, -0.34128013, -0.00668648
        H   2.53407081,  1.81906387, -0.00736748
        H   1.28693681,  3.97963587, -0.00925948
        ''',
            basis='''unc
        #BASIS SET:
        H    S
            1.815041   1
            0.591063   1
        H    P
            2.305000   1
        #BASIS SET:
        C    S
            8.383976   1
            3.577015   1
            1.547118   1
        H    P
            2.305000   1
            1.098827   1
            0.806750   1
            0.282362   1
        H    D
            1.81900    1
            0.72760    1
            0.29104    1
        H    F
            0.970109   1
        C    G
            0.625000   1
        C    H
            0.4        1
            ''',
            output = '/dev/null'
        )
        nao = mol1.nao
        mo_coeff = cupy.random.rand(2, nao, nao)
        mocca = mo_coeff[0,:,:3]
        moccb = mo_coeff[1,:,:2]
        mo_occ = cupy.zeros([2,nao])
        mo_occ[0,:3] = 1
        mo_occ[1,:2] = 1
        dm = cupy.empty([2,nao,nao])
        dm[0] = mocca.dot(mocca.T)
        dm[1] = moccb.dot(moccb.T)
        vj_mo, vk_mo = jk.get_jk(mol1, dm, mo_coeff, mo_occ, hermi=1)
        
        mf = scf.UHF(mol1)
        vj, vk = mf.get_jk(mol1, dm, hermi=1)
        vj2 = cupy.empty([5*nao])
        vk2 = cupy.empty([5*nao])
        vj = vj[0] + vj[1]
        vj2[:3*nao] = (mo_coeff[0].T @ vj @ mocca).reshape(1,-1)
        vj2[3*nao:] = (mo_coeff[1].T @ vj @ moccb).reshape(1,-1)
        vk2[:3*nao] = (mo_coeff[0].T @ vk[0] @ mocca).reshape(1,-1)
        vk2[3*nao:] = (mo_coeff[1].T @ vk[1] @ moccb).reshape(1,-1)
        assert cupy.linalg.norm(vj2 - vj_mo) < 1e-5
        assert cupy.linalg.norm(vk2 - vk_mo) < 1e-5
        mol1.stdout.close()
        
if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()
