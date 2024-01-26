# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import pyscf
from pyscf import lib
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf import scf
from gpu4pyscf.df import df_jk

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_uhf(self):
        print('------- HF -----------------')
        mf = scf.UHF(mol).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_tot = mf.kernel()
        e_pyscf = -75.6599919479438
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_to_cpu(self):
        mf = scf.UHF(mol).density_fit().to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)
        # grids are still not df._key
        #assert 'gpu' not in mf.grids.__module__
        
        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)
        #assert 'gpu' in mf.grids.__module__

if __name__ == "__main__":
    print("Full Tests for unrestricted Hartree-Fock")
    unittest.main()
