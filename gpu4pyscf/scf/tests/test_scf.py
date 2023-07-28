import unittest
import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf import scf
from gpu4pyscf.dft import rks

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-qzvpp'
mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
mol.verbose = 4

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rhf(self):
        mf = scf.RHF(mol)
        mf.max_cycle = 10
        e_tot = mf.kernel(conv_tol=1e-9)
        assert np.allclose(e_tot, -76.0667232412)

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()