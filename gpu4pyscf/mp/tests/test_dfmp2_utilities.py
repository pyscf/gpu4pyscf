# Copyright 2026 The PySCF Developers. All Rights Reserved.
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


"""
Unittest for DF-MP2 utility functions.

This will only test intermediate results, not the final MP2 energy and the full process correctness.

Some additional assumptions:
- no cartesian
- restricted reference
"""

import pytest
import unittest
import cupy as cp
import numpy as np
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import pyscf.mp.dfmp2

from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.RHF(mol).density_fit().run()
    mp = pyscf.mp.dfmp2.DFMP2(mf)
    mp.with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri')
    mp.run()
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class Intermediates(unittest.TestCase):
    def test_balanced_split(self):
        self.assertEqual(dfmp2_addons.balanced_split(9, 3), [3, 3, 3])
        self.assertEqual(dfmp2_addons.balanced_split(10, 3), [4, 3, 3])
        self.assertEqual(dfmp2_addons.balanced_split(5, 10), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_get_j2c_decomp_cpu(self):
        j2c = aux.intor('int2c2e')

        # test usual case of cholesky decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'cd', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'tag'})
        self.assertTrue(j2c_decomp['tag'] == 'cd')
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(np.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(np.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition with lower threshold
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-3)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertFalse(np.allclose(j2c, j2c_rebuild))
        self.assertTrue(np.allclose(j2c, j2c_rebuild, atol=1e-4, rtol=1e-3))

    def test_get_j2c_decomp_gpu(self):
        j2c = cp.asarray(aux.intor('int2c2e'))

        # test usual case of cholesky decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'cd', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'tag'})
        self.assertTrue(j2c_decomp['tag'] == 'cd')
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(cp.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(cp.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition with lower threshold
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-3)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertFalse(cp.allclose(j2c, j2c_rebuild))
        self.assertTrue(cp.allclose(j2c, j2c_rebuild, atol=1e-4, rtol=1e-3))

    def test_j2c(self):
        j2c_cpu = aux.intor('int2c2e')
        j2c_gpu = dfmp2_addons.get_j2c_bdiv(intopt)
        self.assertTrue(cp.allclose(j2c_gpu, j2c_cpu))

    def test_j3c_ovl_bdiv(self):
        nocc = mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        naux_cart = aux.nao_cart()
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]

        j3c_cpu = pyscf.df.incore.aux_e2(mol, aux)
        j3c_ovl_cpu = pyscf.lib.einsum('uvP, ui, va -> iaP', j3c_cpu, occ_coeff, vir_coeff, optimize=True)

        # j3c_ovl on GPU
        j3c_ovl_cart = cp.empty([nocc, nvir, naux_cart])
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on CPU
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart])
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on GPU, FP32
        j3c_ovl_cart = cp.empty([nocc, nvir, naux_cart], dtype=cp.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(j3c_ovl_set[0].dtype == cp.float32)
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))

        # j3c_ovl on CPU, FP32
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart], dtype=np.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(j3c_ovl_set[0].dtype == np.float32)
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))

    def test_get_j3c_by_aux_id_gpu(self):
        # this only tests function works
        vhfopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, aux, 'int2e')
        vhfopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=32)
        for idx_k in range(len(vhfopt.aux_log_qs)):
            j3c_batched = dfmp2_addons.get_j3c_by_aux_id_gpu(vhfopt, idx_k)
            print(j3c_batched.shape, j3c_batched.strides)

    def test_dfmp2_kernel_one_gpu(self):
        nocc = mol.nelectron // 2
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]
        occ_energy = mf.mo_energy[:nocc]
        vir_energy = mf.mo_energy[nocc:]

        result = dfmp2_drivers.dfmp2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.2132034596360335, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.06542902835968734, 7)

        result = dfmp2_drivers.dfmp2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='vhfopt')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.2132034596360335, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.06542902835968734, 7)
