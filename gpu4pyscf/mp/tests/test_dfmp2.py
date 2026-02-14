
import pytest
import unittest
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import gpu4pyscf.mp.dfmp2


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
    
    with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri').build()
    mf._eri = with_df.get_ao_eri()
    mp = pyscf.mp.mp2.MP2(mf).run(with_t2=True)
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class KnownValues(unittest.TestCase):
    def test_dfmp2(self):
        mp_gpu = gpu4pyscf.mp.dfmp2.DFMP2(mf, auxbasis='def2-TZVPP-ri').run()
        self.assertAlmostEqual(mp_gpu.e_corr, mp.e_corr, 7)
