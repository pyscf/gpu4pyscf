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
from gpu4pyscf.dft import uks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-tzvpp'
grids_level = 3
nlcgrids_level = 1

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose = 1,
        spin = 1,
        charge = 1,
        output = '/dev/null'
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = uks.UKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.nlcgrids.level = nlcgrids_level
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by pyscf,           # def2-qzvpp
    '''
    def test_uks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft("LDA, vwn5")
        assert np.allclose(e_tot, -75.4231504131) #-75.42821982483972)

    def test_uks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        assert np.allclose(e_tot, -75.9128621398)# -75.91732813416843)

    def test_uks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP')
        assert np.allclose(e_tot, -75.9987351592) #-76.00306439862237)

    def test_uks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06")
        assert np.allclose(e_tot, -75.9609384616) #-75.96551006522827)

    def test_uks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97")
        assert np.allclose(e_tot, -75.9833214499) #-75.987601337562)

    def test_uks_vv10(self):
        print("------- wB97m-v -------------")
        e_tot = run_dft('HYB_MGGA_XC_WB97M_V')
        assert np.allclose(e_tot, -75.9697577968)# -75.97363094678428)

    #TODO: add test cases for D3/D4 and gradient

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()

