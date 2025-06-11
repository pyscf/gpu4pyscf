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

    # end to end test
    def test_rhf_scf(self):
        e_tot = scf.RHF(mol).kernel()
        e_ref = -151.08447712520285
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rhf_d3(self):
        mf = scf.RHF(mol)
        mf.disp = 'd3bj'
        e_tot = mf.kernel()
        e_ref = -151.1150439066
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rhf_d4(self):
        mf = scf.RHF(mol)
        mf.disp = 'd4'
        e_tot = mf.kernel()
        e_ref = -151.09634038447925
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_chkfile(self):
        ftmp = tempfile.NamedTemporaryFile(dir = pyscf.lib.param.TMPDIR)
        mf = scf.RHF(mol)
        mf.chkfile = ftmp.name
        mf.kernel()
        dm_stored = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        dm_stored = cupy.asnumpy(dm_stored)

        mf_copy = scf.RHF(mol)
        mf_copy.chkfile = ftmp.name
        dm_loaded = mf_copy.init_guess_by_chkfile()
        # Since we reload the MO coefficients, the density matrix should be identical up to numerical noise.
        assert np.allclose(dm_stored, dm_loaded, atol = 1e-14) 
    
    def test_init_guess(self):
        atom = [
            ('X-O', (0.000000, 0.000000, 0.000000)),
            ('H', (0.000000, 0.757160, 0.586260)),
            ('H', (0.000000, -0.757160, 0.586260))
        ]
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = scf.RHF(mol)
        e_tot = mf.kernel()
        e_ref = mf.to_cpu().kernel()
        assert np.abs(e_tot - e_ref) < 1e-7

        mol = pyscf.M(atom=' H 0 0 1.5; Cu 0 0 0', basis='lanl2dz',
                    ecp='lanl2dz', verbose=0)
        mf = scf.RHF(mol)
        e_tot = mf.kernel()
        e_ref = mf.to_cpu().kernel()
        assert np.abs(e_tot - e_ref) < 1e-7

    # TODO:
    #test analyze
    #test mulliken_pop
    #test mulliken_meta
    #test stability
    #test newton
    #test x2c
    #test dipole
    #test canonicalize

if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()
