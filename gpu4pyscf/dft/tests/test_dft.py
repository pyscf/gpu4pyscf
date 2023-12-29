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
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-qzvpp'
grids_level = 3
nlcgrids_level = 1

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose = 1,
        output = '/dev/null'
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.nlcgrids.level = nlcgrids_level
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft("LDA, vwn5")
        assert np.allclose(e_tot, -75.9117117360)

    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        assert np.allclose(e_tot, -76.3866453049)

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP')
        assert np.allclose(e_tot, -76.4728129216)

    def test_rks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06")
        assert np.allclose(e_tot, -76.4321318125)

    def test_rks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97")
        assert np.allclose(e_tot, -76.4543067064)

    def test_rks_vv10(self):
        print("------- wB97m-v -------------")
        e_tot = run_dft('HYB_MGGA_XC_WB97M_V')
        assert np.allclose(e_tot, -76.4391208632)

    #TODO: add test cases for D3/D4 and gradient

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()
