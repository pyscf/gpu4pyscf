# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import rks

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-tzvpp'
grids_level = 5
nlcgrids_level = 2

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        output='/dev/null'
    )
    mol_cart = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        cart=1,
        output = '/dev/null'
    )

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def run_dft(xc, mol, disp=None):
    mf = rks.RKS(mol, xc=xc)
    mf.disp = disp
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
        mf = mol_sph.RKS(xc='LDA,vwn5').to_gpu()
        mf.grids.level = grids_level
        mf.nlcgrids.level = nlcgrids_level
        e_tot = mf.kernel()
        e_ref = -75.9046410402
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

        # test serialization
        mf1 = pickle.loads(pickle.dumps(mf))
        assert mf1.e_tot == e_tot

    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE', mol_sph)
        e_ref = -76.3800182418
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP', mol_sph)
        e_ref = -76.4666495594
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06", mol_sph)
        e_ref = -76.4265870634
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97", mol_sph)
        e_ref = -76.4486274326
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_vv10(self):
        print("------- wB97m-v -------------")
        e_tot = run_dft('HYB_MGGA_XC_WB97M_V', mol_sph)
        e_ref = -76.4334218842
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_cart(self):
        print("-------- cart ---------------")
        e_tot = run_dft('b3lyp', mol_cart)
        e_ref = -76.4672144985
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_d3bj(self):
        print('-------- B3LYP with d3bj -------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d3bj')
        e_ref = -76.4672233969
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_d4(self):
        print('-------- B3LYP with d4 -------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d4')
        e_ref = -76.4669590803
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_b3lyp_d3bj(self):
        print('-------- B3LYP-d3bj -------------')
        e_tot = run_dft('B3LYP-d3bj', mol_sph)
        e_ref = -76.4672233969
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_wb97x_d3bj(self):
        print('-------- wb97x-d3bj -------------')
        e_tot = run_dft('wb97x-d3bj', mol_sph)
        e_ref = -76.47761276450566
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_wb97m_d3bj(self):
        print('-------- wb97m-d3bj -------------')
        e_tot = run_dft('wb97m-d3bj', mol_sph)
        e_ref = -76.47675948061112
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_b3lyp_d4(self):
        print('-------- B3LYP with d4 -------------')
        e_tot = run_dft('B3LYP-d4', mol_sph)
        e_ref = -76.4669590803
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()
