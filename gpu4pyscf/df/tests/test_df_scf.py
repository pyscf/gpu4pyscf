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
from gpu4pyscf.dft import rks

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
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-tzvpp-jkfit')
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    e_dft = mf.kernel()
    return e_dft

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

    def test_rks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft("LDA,VWN5")
        e_qchem = -75.9046407209
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE')
        e_qchem = -76.3800181250
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP')
        e_qchem = -76.4666493796
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_rks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06")
        e_qchem = -76.4265841359
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_rks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97")
        e_qchem = -76.4486277053
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_rks_wb97m_v(self):
        print('-------- wB97m-v --------------')
        e_tot = run_dft("HYB_MGGA_XC_WB97M_V")
        e_qchem = -76.4334567297
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.allclose(e_tot, e_qchem)

    def test_to_cpu(self):
        mf = scf.RHF(mol).density_fit().to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)

        mf = rks.RKS(mol).density_fit().to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        # grids are still not df._key
        #assert 'gpu' not in mf.grids.__module__

        # TODO: coming soon
        #mf = mf.to_gpu()
        #assert isinstance(mf, df_jk._DFHF)
        #assert 'gpu' in mf.grids.__module__

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()
