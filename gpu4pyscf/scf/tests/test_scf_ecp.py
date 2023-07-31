import unittest
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf import scf

lib.num_threads(8)
atom = '''
I 0 0 0 
I 1 0 0
'''
bas='def2-svp'
mol = pyscf.M(atom=atom, basis=bas, ecp=bas)
mol.verbose = 4

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf(self):
        mf = scf.RHF(mol)
        mf.max_cycle = 10
        mf.conv_tol = 1e-9
        e_tot = mf.kernel()
        assert np.allclose(e_tot, -578.9674228876)

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()