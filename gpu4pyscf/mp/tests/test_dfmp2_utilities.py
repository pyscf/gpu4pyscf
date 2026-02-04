"""
Unittest for DF-MP2 utility functions.

This will only test intermediate results, not the final MP2 energy and the full process correctness.

Some additional assumptions:
- no cartesian
- restricted reference
"""

import unittest
import cupy as cp
import pytest
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv

from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers


def setUpModule():
    global mol, aux, mf, with_df, intopt
    atom_token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=atom_token, basis='def2-TZVPP', max_memory=32000, cart=False).build()
    aux = pyscf.gto.Mole(atom=atom_token, basis='def2-TZVPP-ri', max_memory=32000, cart=False).build()
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
        import gpu4pyscf.df.int3c2e

        nocc = mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        naux = aux.nao
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]

        j3c_cpu = pyscf.df.incore.aux_e2(mol, aux)
        j3c_ovl_cpu = pyscf.lib.einsum('uvP, ui, va -> iaP', j3c_cpu, occ_coeff, vir_coeff, optimize=True)

        vhfopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, aux, 'int2e')
        vhfopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=64)

        j3c_ovl_gpu = cp.empty([nocc, nvir, naux])
        dfmp2_addons.get_j3c_ovl_gpu(mol, vhfopt, [occ_coeff], [vir_coeff], [j3c_ovl_gpu])
        j3c_ovl_gpu = vhfopt.unsort_orbitals(j3c_ovl_gpu, aux_axis=[2])
        self.assertTrue(cp.allclose(j3c_ovl_gpu, j3c_ovl_cpu))
