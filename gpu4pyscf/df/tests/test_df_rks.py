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

import unittest
import numpy as np
import pyscf
from pyscf.df import df_jk as cpu_df_jk
from pyscf.dft import rks as cpu_rks
from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.df import df_jk as gpu_df_jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas, max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def run_dft(xc, mol, disp=None):
    mf = gpu_rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-tzvpp-jkfit')
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    mf.disp = disp
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''

    def test_rks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft("LDA,VWN5", mol_sph)
        e_qchem = -75.9046768207
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE', mol_sph)
        e_qchem = -76.3800605005
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP', mol_sph)
        e_qchem = -76.4666819950
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06", mol_sph)
        e_qchem = -76.4266137244
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97", mol_sph)
        e_qchem = -76.4486707747
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_wb97m_v(self):
        print('-------- wB97m-v --------------')
        e_tot = run_dft("HYB_MGGA_XC_WB97M_V", mol_sph)
        e_qchem = -76.4334564629
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_b3lyp_d3(self):
        print('-------- B3LYP with D3(BJ) -------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d3bj')
        e_qchem = -76.4672558312 # w/o D3(BJ) -76.4666819950
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rks_b3lyp_d4(self):
        print('-------- B3LYP with D4 ---------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d4')
        e_qchem = -76.4669915146 # w/o D3(BJ) -76.4666819950
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_to_cpu(self):
        mf = gpu_rks.RKS(mol_sph).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_to_gpu(self):
        mf = cpu_rks.RKS(mol_sph).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        assert isinstance(mf, gpu_df_jk._DFHF)
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_rks_cart(self):
        print('-------- B3LYP (CART) -------------')
        e_tot = run_dft('B3LYP', mol_cart)
        e_ref = -76.46723795965626 # data from PySCF
        print(f'diff from PySCF {e_tot - e_ref}')
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_wb97m_d3bj(self):
        print('-------- wB97m-d3bj -------------')
        e_tot = run_dft('wb97m-d3bj', mol_sph)
        e_ref = -76.47679432135077
        print(f'diff from PySCF {e_tot - e_ref}')
        assert np.abs(e_tot - e_ref) < 1e-5

if __name__ == "__main__":
    print("Full Tests for restricted Kohn-Sham")
    unittest.main()
