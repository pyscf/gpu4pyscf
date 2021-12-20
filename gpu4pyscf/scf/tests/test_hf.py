import unittest
import numpy as np
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
)

def tearDownModule():
    global mol, mol1
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_get_jk(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = mol.RHF()
        mf.device = 'gpu'
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -498.6834601181653 , 7)
        self.assertAlmostEqual(lib.fp(vk), -13.552287262014744, 7)

        mf.device = 'cpu'
        refj, refk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = mol.RHF()
            mf.device = 'gpu'
            vj, vk = mf.get_jk(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -3530.1507509846288, 7)
            self.assertAlmostEqual(lib.fp(vk), -845.7403732632113 , 7)

            mf.device = 'cpu'
            refj, refk = mf.get_jk(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_j(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = mol.RHF()
        mf.device = 'gpu'
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), -498.6834601181653 , 7)

        mf.device = 'cpu'
        refj = mf.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = mol.RHF()
            mf.device = 'gpu'
            vj = mf.get_j(mol, dm)
            self.assertAlmostEqual(lib.fp(vj), -3530.1507509846288, 7)

            mf.device = 'cpu'
            refj = mf.get_j(mol, dm)
            self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k(self):
        np.random.seed(1)
        nao = mol.nao
        dm = np.random.random((nao,nao))
        dm = dm + dm.T
        mf = mol.RHF()
        mf.device = 'gpu'
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(lib.fp(vk), -13.552287262014744, 7)

        mf.device = 'cpu'
        refk = mf.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

        with lib.temporary_env(mol, cart=True):
            np.random.seed(1)
            nao = mol.nao
            dm = np.random.random((nao,nao))
            dm = dm + dm.T
            mf = mol.RHF()
            mf.device = 'gpu'
            vk = mf.get_k(mol, dm)
            self.assertAlmostEqual(lib.fp(vk), -845.7403732632113 , 7)

            mf.device = 'cpu'
            refk = mf.get_k(mol, dm)
            self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_get_jk1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = mol1.RHF()
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vj, vk = mf.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555375858, 7)
        self.assertAlmostEqual(lib.fp(vk), -34.851182918643005, 7)

        mf.device = 'cpu'
        refj, refk = mf.get_jk(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 8)

        np.random.seed(1)
        dm = np.random.random((2,nao,nao))
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vj, vk = mf.get_jk(mol1, dm, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), 89.57263277687994, 7)
        self.assertAlmostEqual(lib.fp(vk),-26.36969769724246, 7)

        mf.device = 'cpu'
        refj, refk = mf.get_jk(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 8)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 8)

    def test_get_j1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = mol1.RHF()
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vj = mf.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vj), 179.14526555375858, 7)

        mf.device = 'cpu'
        refj = mf.get_j(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

        np.random.seed(1)
        dm = np.random.random((2,nao,nao))
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vj = mf.get_j(mol1, dm, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), 89.57263277687994, 7)

        mf.device = 'cpu'
        refj = mf.get_j(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 7)

    def test_get_k1(self):
        # test l >= 4
        np.random.seed(1)
        nao = mol1.nao
        dm = np.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        mf = mol1.RHF()
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vk = mf.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(lib.fp(vk), -34.851182918643005, 7)

        mf.device = 'cpu'
        refk = mf.get_k(mol1, dm, hermi=1)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

        np.random.seed(1)
        dm = np.random.random((2,nao,nao))
        with lib.temporary_env(scf.hf, BLKSIZE_BY_L=[8, 24, 24, 40, 120]):
            mf.device = 'gpu'
            vk = mf.get_k(mol1, dm, hermi=0)
        self.assertAlmostEqual(lib.fp(vk),-26.36969769724246, 7)

        mf.device = 'cpu'
        refk = mf.get_k(mol1, dm, hermi=0)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 7)

    def test_group_by_l(self):
        l_slices, g_shls, h_shls = scf.hf._group_shells_to_slices(mol1)
        self.assertEqual(list(l_slices), [0, 10, 20, 26, 28, 30])
        self.assertEqual(list(g_shls), [28, 30])
        self.assertEqual(list(h_shls), [30, 32])


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()
