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
from gpu4pyscf.dft import uks

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'
grids_level = 5

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

def run_dft(xc):
    mf = uks.UKS(mol, xc=xc).density_fit(auxbasis='def2-tzvpp-jkfit')
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_uks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft("LDA,VWN5")
        e_pyscf = -75.42319302444447
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_uks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        e_pyscf = -75.91291185761159
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_uks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP')
        e_pyscf = -75.9987750880688
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_uks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06")
        e_pyscf = -75.96097588711966
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_uks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97")
        e_pyscf = -75.98337641724999
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_uks_wb97m_v(self):
        print('-------- wB97m-v --------------')
        e_tot = run_dft("HYB_MGGA_XC_WB97M_V")
        e_pyscf = -75.96980058343685
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.allclose(e_tot, e_pyscf)

    def test_to_cpu(self):
        mf = uks.UKS(mol).density_fit().to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        # grids are still not df._key
        #assert 'gpu' not in mf.grids.__module__

        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)
        #assert 'gpu' in mf.grids.__module__

if __name__ == "__main__":
    print("Full Tests for unrestricted Kohn-Sham")
    unittest.main()
