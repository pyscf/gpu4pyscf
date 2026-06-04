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
from pyscf import df, lib
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.df import int3c2e, df_jk
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

    def test_vj_incore(self):
        int3c_gpu = int3c2e.get_int3c2e(mol, auxmol, aosym=True, direct_scf_tol=1e-14)
        intopt = int3c2e.VHFOpt(mol, auxmol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True)
        cupy.random.seed(np.asarray(1, dtype=np.uint64))
        nao = intopt.mol.nao
        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T

        # pass 1
        rhoj_outcore = cupy.einsum('ijL,ij->L', int3c_gpu, dm)
        rhoj_incore = 2.0*int3c2e.get_j_int3c2e_pass1(intopt, dm)
        assert cupy.linalg.norm(rhoj_outcore - rhoj_incore) < 1e-8

        # pass 2
        vj_outcore = cupy.einsum('ijL,L->ij', int3c_gpu, rhoj_outcore)
        vj_incore = int3c2e.get_j_int3c2e_pass2(intopt, rhoj_incore)
        assert cupy.linalg.norm(vj_outcore - vj_incore) < 1e-5
    
    def test_vj_sph_incore(self):
        int3c_gpu = int3c2e.get_int3c2e(mol_sph, auxmol, aosym=True, direct_scf_tol=1e-14)
        intopt = int3c2e.VHFOpt(mol_sph, auxmol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=False, aosym=True)
        cupy.random.seed(np.asarray(1, dtype=np.uint64))
        nao = intopt.mol.nao
        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T
        
        # pass 1
        rhoj_outcore = cupy.einsum('ijL,ij->L', int3c_gpu, dm)
        rhoj_incore = 2.0*int3c2e.get_j_int3c2e_pass1(intopt, dm)
        assert cupy.linalg.norm(rhoj_outcore - rhoj_incore) < 1e-8

        # pass 2
        vj_outcore = cupy.einsum('ijL,L->ij', int3c_gpu, rhoj_outcore)
        vj_incore = int3c2e.get_j_int3c2e_pass2(intopt, rhoj_incore)
        assert cupy.linalg.norm(vj_outcore - vj_incore) < 1e-5

    def test_j_outcore(self):
        cupy.random.seed(np.asarray(1, dtype=np.uint64))
        nao = mol.nao
        dm = cupy.random.rand(nao, nao)
        dm = dm + dm.T
        mf = gpu_scf.RHF(mol).density_fit()
        mf.kernel()
        vj0, _ = mf.get_jk(dm=dm, with_j=True, with_k=False, hermi=1)
        vj = df_jk.get_j(mf.with_df, dm)
        assert cupy.linalg.norm(vj - vj0) < 1e-4
    
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

if __name__ == "__main__":
    print("Full Tests for DF JK")
    unittest.main()
