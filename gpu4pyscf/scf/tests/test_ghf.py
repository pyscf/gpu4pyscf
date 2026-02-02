# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import pyscf
import numpy
import cupy as cp


def setUpModule():
    global mol, mol1
    mol = pyscf.gto.Mole()
    mol.atom = '''
        O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''
    mol.spin = None
    mol.basis = 'cc-pvdz'
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.build()

    mol1 = pyscf.gto.M(
        verbose = 0,
        atom = '''
            O    0    0    0
            H    0.   -0.757   0.587
            H    0.   0.757    0.587''',
        charge = 1,
        spin = 1,
        basis = 'cc-pvdz',
        output = '/dev/null')


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_ghf_scf(self):
        mf = mol.GHF().to_gpu()
        assert mf.device == 'gpu'
        e_tot = mf.kernel()
        e_ref = mf.to_cpu().kernel()
        assert abs(e_tot - e_ref) < 1e-5

    def test_ghf_scf_complex_dm(self):
        mf = mol.GHF().to_gpu()
        assert mf.device == 'gpu'
        e_tot = mf.kernel()
        dm1 = mf.make_rdm1()
        numpy.random.seed(1)
        n2c = mol.nao_nr() * 2
        dm_perturb = numpy.random.random((n2c,n2c)) + 1j*numpy.random.random((n2c,n2c))
        dm_perturb = dm_perturb + dm_perturb.T.conj()
        dm_perturb = cp.asarray(dm_perturb)
        dm_1 = dm1 + dm_perturb*0.001
        e_ref = mf.to_cpu().kernel(dm_1.get())
        e_tot = mf.kernel(dm_1)
        assert abs(e_tot - e_ref) < 1e-5

    def test_get_jk_complex(self):
        mf = mol.GHF().to_gpu()
        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao)) + 1j*numpy.random.random((nao,nao))
        d = d1 + d1.T.conj()
        d_real = d.real
        vj_gpu = mf.get_j(mol, d_real)
        vk_gpu = mf.get_k(mol, d)

        mf_cpu = mf.to_cpu()
        vj_cpu = mf_cpu.get_j(mol, d_real)
        vk_cpu = mf_cpu.get_k(mol, d)

        assert numpy.allclose(vj_gpu, vj_cpu)
        assert numpy.allclose(vk_gpu, vk_cpu)

    def test_get_jk_real(self):
        mf = mol.GHF().to_gpu()

        nao = mol.nao_nr()*2
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d = d1 + d1.T.conj()
        d_real = d.real
        vj_gpu = mf.get_j(mol, d_real)
        vk_gpu = mf.get_k(mol, d)

        mf_cpu = mf.to_cpu()
        vj_cpu = mf_cpu.get_j(mol, d_real)
        vk_cpu = mf_cpu.get_k(mol, d)

        assert numpy.allclose(vj_gpu, vj_cpu)
        assert numpy.allclose(vk_gpu, vk_cpu)

    def test_to_cpu(self):
        mf = mol.GHF().to_gpu()
        assert mf.device == 'gpu'
        e_tot = mf.kernel()
        mf = mf.to_cpu()
        assert getattr(mf, 'device', None) is None
        e_ref = mf.kernel()
        assert abs(e_tot - e_ref) < 1e-5


if __name__ == "__main__":
    print("Full Tests for ghf")
    unittest.main()
