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
    spin=None,
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
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = scf.RHF(mol)
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -498.6834601181653 , 7)
        self.assertAlmostEqual(lib.fp(vk), -13.552287262014744, 7)
        mf1 = mf.to_cpu()
        refj, refk = mf1.get_jk(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)
        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = scf.RHF(mol)
            vj, vk = mf.get_jk(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -3530.1507509846288, 7)
            self.assertAlmostEqual(lib.fp(vk), -845.7403732632113 , 7)

            mf1 = mf.to_cpu()
            refj, refk = mf1.get_jk(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_j(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = scf.RHF(mol)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -498.6834601181653 , 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = scf.RHF(mol)
            vj = mf.get_j(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -3530.1507509846288, 7)

            mf1 = mf.to_cpu()
            refj = mf1.get_j(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = scf.RHF(mol)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(lib.fp(vk), -13.552287262014744, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = scf.RHF(mol)
            vk = mf.get_k(mol, dm)
            self.assertAlmostEqual(lib.fp(vk), -845.7403732632113 , 7)

            mf1 = mf.to_cpu()
            refk = mf1.get_k(mol, dm)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_jk1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.RHF(mol1)
        vj, vk = mf.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555375858, 7)
        self.assertAlmostEqual(lib.fp(vk), -34.851182918643005, 7)

        mf1 = mf.to_cpu()
        refj, refk = mf1.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 8)

    @unittest.skip('hermi=0')
    def test_get_jk1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.RHF(mol1)
        vj, vk = mf.get_jk(mol1, cupy.asarray(dm), hermi=0)
        self.assertAlmostEqual(lib.fp(vj.get()), 89.57263277687994, 7)
        self.assertAlmostEqual(lib.fp(vk.get()),-26.36969769724246, 7)

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
        mf = scf.RHF(mol1)
        vj = mf.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555375858, 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    @unittest.skip('hermi=0')
    def test_get_j1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.RHF(mol1)
        vj = mf.get_j(mol1, dm, hermi=0).get()
        self.assertAlmostEqual(lib.fp(vj), 89.57263277687994, 7)

        mf1 = mf.to_cpu()
        refj = mf1.get_j(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = scf.RHF(mol1)
        vk = mf.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vk), -34.851182918643005, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    @unittest.skip('hermi=0')
    def test_get_k1_hermi0(self):
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        mf = scf.RHF(mol1)
        vk = mf.get_k(mol1, dm, hermi=0).get()
        self.assertAlmostEqual(lib.fp(vk),-26.36969769724246, 7)

        mf1 = mf.to_cpu()
        refk = mf1.get_k(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)
    '''
    # end to end test
    def test_rhf_scf(self):
        e_tot = scf.RHF(mol).kernel()
        self.assertAlmostEqual(e_tot, -151.08447712520285)
    '''
if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()
