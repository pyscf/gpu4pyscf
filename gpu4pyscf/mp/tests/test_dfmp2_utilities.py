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

from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers


def setUpModule():
    global mol, aux, mf, with_df, intopt
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
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, intopt


class Intermediates(unittest.TestCase):
    def test_j2c(self):
        j2c_cpu = aux.intor('int2c2e')
        j2c_gpu = gpu4pyscf.df.int3c2e_bdiv.int2c2e(aux)
        self.assertTrue(cp.allclose(j2c_gpu, j2c_cpu))

    def test_j3c_ovl(self):
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
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart], 64)
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on CPU
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart])
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart], 64)
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on GPU, FP32
        j3c_ovl_cart = cp.empty([nocc, nvir, naux_cart], dtype=cp.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart], 64)
        self.assertTrue(j3c_ovl_set[0].dtype == cp.float32)
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))
        
        # j3c_ovl on CPU, FP32
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart], dtype=np.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart], 64)
        self.assertTrue(j3c_ovl_set[0].dtype == np.float32)
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))
