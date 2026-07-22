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
import cupy as cp
import pyscf
from pyscf import df, lib
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.df import df_jk
from gpu4pyscf.df.df import DF
from gpu4pyscf.lib.cupy_helper import tag_array

atom='''
Ti 0.0 0.0 0.0
Cl 0.0 0.0 2.0
Cl 0.0 2.0 -1.0
Cl 1.73 -1.0 -1.0
Cl -1.73 -1.0 -1.0''',

bas='def2-tzvpp'

def setUpModule():
    global mol, mol_sph, auxmol, auxmol_sph
    mol = pyscf.M(atom=atom, basis=bas, output='/dev/null', cart=True, verbose=1)
    auxmol = df.addons.make_auxmol(mol, auxbasis='sto3g')

    mol_sph = pyscf.M(atom=atom, basis=bas, output='/dev/null', cart=False, verbose=1)
    auxmol_sph = df.addons.make_auxmol(mol_sph, auxbasis='sto3g')

def tearDownModule():
    global mol, mol_sph, auxmol, auxmol_sph
    mol.stdout.close()
    mol_sph.stdout.close()
    auxmol.stdout.close()
    auxmol_sph.stdout.close()
    del mol, auxmol, mol_sph, auxmol_sph

class KnownValues(unittest.TestCase):
    def test_get_j(self):
        cupy.random.seed(np.asarray(1, dtype=np.uint64))
        nao = mol.nao
        mf = gpu_scf.RHF(mol)
        mf = mf.density_fit(auxbasis='sto3g')
        mf.with_df.build()

        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T
        vj = df_jk.get_j(mf.with_df, dm)
        ref, _ = mf.get_jk(dm=dm, hermi=1)
        assert abs(vj - ref).max() < 1e-11

        dm = cupy.random.rand(2, nao, nao)
        vj = df_jk.get_j(mf.with_df, dm, hermi=0)
        ref, _ = mf.get_jk(dm=dm, hermi=0)
        assert abs(vj - ref).max() < 1e-12

        dm = cupy.random.rand(15, nao, nao)
        vj = df_jk.get_j(mf.with_df, dm, hermi=0)
        ref, _ = mf.get_jk(dm=dm, hermi=0)
        assert abs(vj - ref).max() < 1e-11

    def test_jk_hermi0(self):
        dfobj = DF(mol, 'sto3g').build()
        np.random.seed(3)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        refj, refk = dfobj.to_cpu().get_jk(dm, hermi=0)
        vj, vk = dfobj.get_jk(dm, hermi=0)
        assert abs(vj - refj).max() < 1e-9
        assert abs(vk - refk).max() < 1e-9
        assert abs(lib.fp(vj) - 455.864593801164).max() < 1e-9
        assert abs(lib.fp(vk) - 37.7022369618297).max() < 1e-9

    def test_jk_mo(self):
        dfobj = DF(mol, 'sto3g').build()
        np.random.seed(3)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao)
        mo_occ = np.zeros([nao])
        mo_occ[:3] = 2
        dm = 2.0*mo_coeff[:,mo_occ>1].dot(mo_coeff[:,mo_occ>1].T)
        refj, refk = dfobj.to_cpu().get_jk(dm)
        dm = cupy.asarray(dm)
        dm = tag_array(dm, mo_coeff=cupy.asarray(mo_coeff), mo_occ=cupy.asarray(mo_occ))
        vj, vk = dfobj.get_jk(dm)
        vj = vj.get()
        vk = vk.get()
        assert abs(vj - refj).max() < 1e-9
        assert abs(vk - refk).max() < 1e-9

    def test_jk_cpu(self):
        dfobj = DF(mol, 'sto3g').build()
        dfobj.use_gpu_memory = False
        np.random.seed(3)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao)
        mo_occ = np.zeros([nao])
        mo_occ[:3] = 2
        dm = 2.0*mo_coeff[:,mo_occ>1].dot(mo_coeff[:,mo_occ>1].T)
        refj, refk = dfobj.to_cpu().get_jk(dm)
        dm = cupy.asarray(dm)
        dm = tag_array(dm, mo_coeff=cupy.asarray(mo_coeff), mo_occ=cupy.asarray(mo_occ))
        vj, vk = dfobj.get_jk(dm)
        vj = vj.get()
        vk = vk.get()
        assert abs(vj - refj).max() < 1e-9
        assert abs(vk - refk).max() < 1e-9

    def test_limited_mem(self):
        from gpu4pyscf.df import df
        mol = pyscf.M(atom='''
O       0.873    5.017    1.816
H       1.128    5.038    2.848
H       0.173    4.317    1.960
O       3.665    1.316    1.319
H       3.904    2.233    1.002
H       4.224    0.640    0.837
''', basis='def2-tzvp')
        mf = mol.RHF().to_gpu()
        mf = mf.density_fit()
        mf.with_df.use_gpu_memory = False
        mf.run()
        assert isinstance(mf.with_df._cderi[0], np.ndarray)
        assert abs(mf.e_tot - -152.09455538734778) < 1e-8

        with lib.temporary_env(df, get_avail_mem=lambda *args, **kw: 3000000):
            mf = mol.RHF().to_gpu()
            mf = mf.density_fit().run()
            assert isinstance(mf.with_df._cderi[0], np.ndarray)
            assert abs(mf.e_tot - -152.09455538734778) < 1e-8

    def test_ghf_get_jk_real(self):
        mol = pyscf.M(
            atom = '''
            O    0    0    0
            H    0.   -0.757   0.587
            H    0.   0.757    0.587''',
            basis = 'def2-svp')
        mf = mol.GHF().to_gpu().density_fit(auxbasis='weigend')
        dm = mf.get_init_guess(key='hcore')

        mf_cpu = mf.to_cpu()
        ref = mf_cpu.get_jk(mol, dm.get())

        vj, vk = mf.get_jk(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-11
        assert abs(ref[1] - vk.get()).max() < 1e-12

        vj = mf.get_j(mol, dm)
        vk = mf.get_k(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-12
        assert abs(ref[1] - vk.get()).max() < 1e-12

        vk_ref = mol.GHF().get_k(mol, dm.get())
        mf.only_dfj = True
        vj, vk = mf.get_jk(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-12
        assert abs(vk_ref - vk.get()).max() < 1e-12

        vj = mf.get_j(mol, dm)
        vk = mf.get_k(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-12
        assert abs(vk_ref - vk.get()).max() < 1e-12

    def test_ghf_get_jk_complex(self):
        mol = pyscf.M(
            atom = '''
            O    0    0    0
            H    0.   -0.757   0.587
            H    0.   0.757    0.587''',
            basis = 'def2-svp')
        mf = mol.GHF().to_gpu().density_fit(auxbasis='weigend')
        cp.random.seed(1)
        n2c = mol.nao * 2
        dm = cp.random.rand(n2c, n2c) + 1j * cp.random.rand(n2c, n2c)
        dm = dm + dm.conj().T

        mf_cpu = mf.to_cpu()
        ref = mf_cpu.get_jk(mol, dm.get())

        vj, vk = mf.get_jk(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-11
        assert abs(ref[1] - vk.get()).max() < 1e-11

        vj = mf.get_j(mol, dm)
        vk = mf.get_k(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-11
        assert abs(ref[1] - vk.get()).max() < 1e-11

        vk_ref = mol.GHF().get_k(mol, dm.get())
        mf.only_dfj = True
        vj, vk = mf.get_jk(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-11
        assert abs(vk_ref - vk.get()).max() < 1e-11

        vj = mf.get_j(mol, dm)
        vk = mf.get_k(mol, dm)
        assert abs(ref[0] - vj.get()).max() < 1e-11
        assert abs(vk_ref - vk.get()).max() < 1e-11

if __name__ == "__main__":
    print("Full Tests for DF JK")
    unittest.main()
