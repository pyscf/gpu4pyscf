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
from pyscf.dft import uks as cpu_uks
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf.dft import uks as gpu_uks
from gpu4pyscf.df import df_jk as gpu_df_jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'
grids_level = 5

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000)
    mol_sph.output = '/dev/null'
    mol_sph.build()
    mol_sph.verbose = 1

    mol_cart = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, cart=1, max_memory=32000)
    mol_cart.output = '/dev/null'
    mol_cart.build()
    mol_cart.verbose = 1

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def run_dft(mol, xc, disp=None):
    mf = gpu_uks.UKS(mol, xc=xc).density_fit(auxbasis='def2-tzvpp-jkfit')
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    mf.disp = disp
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_uks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft(mol_sph, "LDA,VWN5")
        e_pyscf = -75.42319302444447
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft(mol_sph, 'PBE')
        e_pyscf = -75.91291185761159
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft(mol_sph, 'B3LYP')
        e_pyscf = -75.9987750880688
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft(mol_sph, "M06")
        e_pyscf = -75.96097588711966
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft(mol_sph, "HYB_GGA_XC_WB97")
        e_pyscf = -75.98337641724999
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_wb97m_v(self):
        print('-------- wB97m-v --------------')
        e_tot = run_dft(mol_sph, "HYB_MGGA_XC_WB97M_V")
        e_pyscf = -75.96980058343685
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_b3lyp_d3(self):
        print('-------- B3LYP D3(BJ) -------------')
        e_tot = run_dft(mol_sph, 'B3LYP', disp='d3bj')
        e_pyscf = -75.9993489428 #-75.9987750880688
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_b3lyp_d4(self):
        print('-------- B3LYP D4 -------------')
        e_tot = run_dft(mol_sph, 'B3LYP', disp='d4')
        e_pyscf =  -75.9989312099 #-75.9987750880688
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_to_cpu(self):
        mf = gpu_uks.UKS(mol_sph).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_to_gpu(self):
        mf = cpu_uks.UKS(mol_sph).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_gpu()
        assert isinstance(mf, gpu_df_jk._DFHF)
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_uks_cart(self):
        print('-------- B3LYP cart-------------')
        mf = gpu_uks.UKS(mol_cart, xc='B3LYP').density_fit()
        e_tot = mf.kernel()
        e_pyscf = mf.to_cpu().kernel()
        print(f'diff from pyscf {e_tot - e_pyscf}')
        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_uks_wb97m_d3bj(self):
        print('-------- wB97m-d3bj -------------')
        e_tot = run_dft(mol_sph, 'wb97m-d3bj')
        e_ref = -76.00969751095374
        print(f'diff from pyscf {e_tot - e_ref}')
        assert np.abs(e_tot - e_ref) < 1e-5

if __name__ == "__main__":
    print("Full Tests for unrestricted Kohn-Sham")
    unittest.main()
