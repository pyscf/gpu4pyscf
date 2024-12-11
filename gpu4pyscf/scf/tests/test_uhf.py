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
import tempfile
import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf import scf

mol = pyscf.M(
    atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
C   1.43439081,  1.81898387, -0.00800148
C   0.73673681,  3.02749287, -0.00920048
''',
    basis='ccpvtz',
    charge=1,
    spin=1,
    output = '/dev/null'
)

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

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_get_jk(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol)
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -1782.4478082102428, 7)
        self.assertAlmostEqual(lib.fp(vk), -280.36548013781095, 7)
        mf1 = mf.to_cpu()
        refj, refk = mf1.get_jk(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)
        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((2, nao,nao))
            dm = dm + dm.transpose(0,2,1)
            mf = scf.UHF(mol)
            vj, vk = mf.get_jk(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -1790.0063863999496, 7)
            self.assertAlmostEqual(lib.fp(vk), -8.969890703683895 , 7)

            mf1 = mf.to_cpu()
            refj, refk = mf1.get_jk(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_j(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -1782.4478082102423 , 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((2,nao,nao))
            dm = dm + dm.transpose(0,2,1)
            mf = scf.UHF(mol)
            vj = mf.get_j(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -1790.0063863999503, 7)

            mf1 = mf.to_cpu()
            refj = mf1.get_j(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(lib.fp(vk), -280.36548013781083, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((2,nao,nao))
            dm = dm + dm.transpose(0,2,1)
            mf = scf.UHF(mol)
            vk = mf.get_k(mol, dm)
            self.assertAlmostEqual(lib.fp(vk), -8.969890703691519 , 7)

            mf1 = mf.to_cpu()
            refk = mf1.get_k(mol, dm)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_jk1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol1)
        vj, vk = mf.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555374763, 7)
        self.assertAlmostEqual(lib.fp(vk), -34.851182918653606, 7)

        mf1 = mf.to_cpu()
        refj, refk = mf1.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 8)

    @unittest.skip('hermi=0')
    def test_get_jk1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.UHF(mol1)
        vj, vk = mf.get_jk(mol1, cupy.asarray(dm), hermi=0)
        self.assertAlmostEqual(lib.fp(vj.get()), 89.57263277687345 , 7)
        self.assertAlmostEqual(lib.fp(vk.get()),-26.369697697245883, 7)

        mf1 = mf.to_cpu()
        refj, refk = mf1.get_jk(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vj.get() - refj).max(), 0, 8)
        self.assertAlmostEqual(abs(vk.get() - refk).max(), 0, 8)

    def test_get_j1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol1)
        vj = mf.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555374712, 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    @unittest.skip('hermi=0')
    def test_get_j1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.UHF(mol1)
        vj = mf.get_j(mol1, dm, hermi=0).get()
        self.assertAlmostEqual(lib.fp(vj), 89.5726327768736, 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.UHF(mol1)
        vk = mf.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vk), -34.85118291865315, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    @unittest.skip('hermi=0')
    def test_get_k1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.UHF(mol1)
        vk = mf.get_k(mol1, dm, hermi=0).get()
        self.assertAlmostEqual(lib.fp(vk),-26.369697697246007, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    # end to end test
    def test_uhf_scf(self):
        e_tot = scf.UHF(mol).kernel()
        e_ref = -150.76441654065087
        print('--------- testing UHF -----------')
        print('pyscf - qchem ', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_uhf_d3bj(self):
        mf = scf.UHF(mol)
        mf.disp = 'd3bj'
        e_tot = mf.kernel()
        e_ref = -150.7949833081
        print('--------- testing UHF with D3BJ ---')
        print('pyscf - qchem ', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    '''
    def test_uhf_d4(self):
        mf = scf.UHF(mol)
        mf.disp = 'd4'
        e_tot = mf.kernel()
        e_ref = -150.7604264160
        print('--------- testing UHF with D4 ----')
        print('pyscf - qchem ', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5
    '''

    def test_chkfile(self):
        ftmp = tempfile.NamedTemporaryFile(dir = pyscf.lib.param.TMPDIR)
        mf = scf.UHF(mol)
        mf.chkfile = ftmp.name
        mf.kernel()
        dma_stored, dmb_stored = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        dma_stored, dmb_stored = cupy.asnumpy(dma_stored), cupy.asnumpy(dmb_stored)

        mf_copy = scf.UHF(mol)
        mf_copy.chkfile = ftmp.name
        dma_loaded, dmb_loaded = mf_copy.init_guess_by_chkfile()
        assert np.allclose(dma_stored, dma_loaded, atol = 1e-14) # Since we reload the MO coefficients, the density matrix should be identical up to numerical noise.
        assert np.allclose(dmb_stored, dmb_loaded, atol = 1e-14)
        assert not np.allclose(dma_stored, dmb_loaded, atol = 1e-1) # Just to make sure alpha and beta electron are different in the test system

    # TODO:
    #test analyze
    #test mulliken_pop
    #test mulliken_spin_pop
    #test mulliken_meta
    #test mulliken_meta_spin
    #test stability
    #test newton
    #test x2c
    #test dipole
    #test canonicalize
    #test det_ovlp

if __name__ == "__main__":
    print("Full Tests for UHF")
    unittest.main()
