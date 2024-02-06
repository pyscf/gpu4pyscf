# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import rks


atom = '''
I 0 0 0
I 1 0 0
'''
bas='def2-tzvpp'
grids_level = 7

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, ecp=bas)
    mol.output='/dev/null'
    mol.verbose = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.grids.prune = None
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        assert np.allclose(e_tot, -582.7625143308, rtol=1e-8)

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()
