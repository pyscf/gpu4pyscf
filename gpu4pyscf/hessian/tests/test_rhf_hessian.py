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
import pyscf
from pyscf import gto, lib
from pyscf.hessian import rhf as rhf_cpu
from gpu4pyscf import scf, hessian
from gpu4pyscf.hessian import rhf as rhf_gpu
from gpu4pyscf.hessian import jk

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_hessian_rhf(self):
        mf = mol.RHF().run()
        mf.conv_tol_cpscf = 1e-8
        hobj = mf.Hessian()
        ref = hobj.kernel()
        e2_gpu = hobj.to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-6

    def test_partial_hess_elec(self):
        mf = pyscf.scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = rhf_cpu._partial_hess_ejk(hobj)
        e2_cpu = ej_cpu - ek_cpu

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, e2_gpu = rhf_gpu._partial_hess_ejk(hobj)

        assert abs(e1_cpu - e1_gpu.get()).max() < 1e-7
        assert abs(e2_cpu - e2_gpu.get()).max() < 1e-7

    def test_ejk_ip2(self):
        mol = gto.M(
            atom = '''
            O       0.0000000000    -0.0000000000     0.1174000000
                C 1. 1. 0
                H1 3.1 0.12 4.35
                H2 2.1 1.31 6
            ''',
            basis='6-31g**', unit='B', output = '/dev/null')
        np.random.seed(9)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao) - .5
        dm = mo_coeff.dot(mo_coeff.T) * 2
        mo_occ = np.ones(nao) * 2
        mo_energy = np.random.rand(nao)

        ejk = rhf_gpu._partial_ejk_ip2(mol, dm)
        assert abs(lib.fp(ejk.get()) - 1116.6336092900506) < 1e-8
        mf = mol.RHF()
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        mf.mo_energy = mo_energy
        h = rhf_cpu.Hessian(mf)
        e1, refj, refk = rhf_cpu._partial_hess_ejk(h, mo_energy, mo_coeff, mo_occ)
        e2_ref = refj - refk
        assert abs(ejk.get() - e2_ref).max() < 1e-8
        mol.stdout.close()
        
    def test_get_jk(self):
        mol = gto.M(
            atom = '''
            O       0.0000000000    -0.0000000000     0.1174000000
            H      -0.7570000000     4.0000000000    -0.4696000000
            H       0.7570000000     4.0000000000    -0.4696000000
                C 1. 1. 0
                H1 3.1 0.12 4.35
                H2 2.1 1.31 6
            ''',
            basis='def2-tzvpp', unit='B', output = '/dev/null')
        np.random.seed(9)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao)
        dm = mo_coeff.dot(mo_coeff.T) * 2
        vj, vk = rhf_gpu._get_jk_ip1(mol, dm)
        assert abs(lib.fp(vj.get()) -  87674.69061160382) < 1e-7
        assert abs(lib.fp(vk.get()) - -9.317650662101629) < 1e-7

        h1ao = [None] * mol.natm
        aoslices = mol.aoslice_by_atom()
        for ia in range(mol.natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, mol.nbas)*3
            vj1, vj2, vk1, vk2 = rhf_cpu._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['ji->s2kl', -dm[:,p0:p1],  # vj1
                                          'lk->s1ij', -dm         ,  # vj2
                                          'li->s1kj', -dm[:,p0:p1],  # vk1
                                          'jk->s1il', -dm         ], # vk2
                                         shls_slice=shls_slice)
            vj1[:,p0:p1] += vj2
            vk1[:,p0:p1] += vk2
            vj1 = vj1 + vj1.transpose(0,2,1)
            vk1 = vk1 + vk1.transpose(0,2,1)
            h1ao[ia] = vj1, vk1
        h1ao = np.array(h1ao)
        refj = h1ao[:,0]
        refk = h1ao[:,1]
        mol.stdout.close()
        assert abs(vj.get() - refj).max() < 1e-8
        assert abs(vk.get() - refk).max() < 1e-8

    def test_hessian_rhf_D3(self):
        print('----- testing RHF with D3BJ ------')
        mf = mol.RHF()
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
        mo_coeff = cupy.random.rand(nao, nao)
        mo_occ = cupy.zeros([nao])
        mo_occ[:3] = 2
        mocc = mo_coeff[:,:3]
        dm = mocc.dot(mocc.T) * 2
        vj_mo, vk_mo = jk.get_jk(mol1, dm, mo_coeff, mo_occ, hermi=1)
        
        mf = scf.RHF(mol1)
        vj, vk = mf.get_jk(mol1, dm, hermi=1)
        vj_cpu = (mo_coeff.T @ vj @ mocc).reshape(1,-1)
        vk_cpu = (mo_coeff.T @ vk @ mocc).reshape(1,-1)
        assert cupy.linalg.norm(vj_cpu - vj_mo) < 1e-5
        assert cupy.linalg.norm(vk_cpu - vk_mo) < 1e-5
        mol1.stdout.close()

    def test_ecp_hess(self):
        mol = gto.M(atom='H 0 0 1.5; Cu 0 0 0', basis='lanl2dz',
                    ecp={'Cu':'lanl2dz'}, 
                    verbose=0,
                    output = '/dev/null')
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.20927804440983355, 6)

        mfs = mf.nuc_grad_method().as_scanner()
        e1 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0  0.001'))[1]
        e2 = mfs(mol.set_geom_('H 0 0 1.5; Cu 0 0 -0.001'))[1]
        mol.stdout.close()
        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/0.002*lib.param.BOHR).max(), 0, 5)

if __name__ == "__main__":
    print("Full Tests for RHF Hessian")
    unittest.main()
