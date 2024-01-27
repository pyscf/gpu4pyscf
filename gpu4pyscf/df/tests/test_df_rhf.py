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

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rhf(self):
        print('------- HF -----------------')
        mf = scf.RHF(mol).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_tot = mf.kernel()
        e_qchem = -76.0624582299
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_to_cpu(self):
        mf = scf.RHF(mol).density_fit().to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)

        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)
        #assert 'gpu' in mf.grids.__module__

if __name__ == "__main__":
    print("Full Tests for restricted Hartree-Fock")
    unittest.main()
