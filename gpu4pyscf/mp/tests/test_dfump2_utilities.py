import pytest
import unittest
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv

from gpu4pyscf.mp import dfmp2_drivers


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.UHF(mol).density_fit().run()
    mp = pyscf.mp.dfump2.DFUMP2(mf)
    mp.with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri')
    mp.run()
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class Intermediates(unittest.TestCase):
    def test_dfump2_kernel_one_gpu(self):
        spins = [0, 1]
        nocc = mol.nelec
        occ_coeff = [mf.mo_coeff[s][:, : nocc[s]] for s in spins]
        vir_coeff = [mf.mo_coeff[s][:, nocc[s] :] for s in spins]
        occ_energy = [mf.mo_energy[s][: nocc[s]] for s in spins]
        vir_energy = [mf.mo_energy[s][nocc[s] :] for s in spins]

        result = dfmp2_drivers.dfump2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.17417642616453902, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.05344288908582173, 7)

        result = dfmp2_drivers.dfump2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='vhfopt')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.17417642616453902, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.05344288908582173, 7)
