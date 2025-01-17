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
import pickle
import numpy as np
import pyscf
from pyscf import scf as cpu_scf
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf.df import df_jk as gpu_df_jk
from gpu4pyscf import scf as gpu_scf
try:
    import cloudpickle
except ImportError:
    cloudpickle = None

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas, cart=0,
                      symmetry=True, output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rhf(self):
        print('------- RHF -----------------')
        mf = mol_sph.RHF().density_fit(auxbasis='def2-tzvpp-jkfit').to_gpu()
        e_tot = mf.kernel()
        e_qchem = -76.0624582299
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

        # test serialization
        if cloudpickle is not None:
            mf1 = pickle.loads(cloudpickle.dumps(mf))
            assert mf1.e_tot == e_tot

    def test_cart(self):
        print('------- RHF Cart -----------------')
        mf = gpu_scf.RHF(mol_cart).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_gpu = mf.kernel()
        e_cpu = mf.to_cpu().kernel()
        print(f'diff from pyscf {e_gpu - e_cpu}')
        assert np.abs(e_gpu - e_cpu) < 1e-5

    def test_rhf_d3(self):
        print('--------- RHF with D3(BJ) ---------')
        mf = gpu_scf.RHF(mol_sph).density_fit(auxbasis='def2-tzvpp-jkfit')
        mf.disp = 'd3bj'
        e_tot = mf.kernel()
        e_qchem = -76.0669672259
        print(f'diff from qchem {e_tot - e_qchem}')
        assert np.abs(e_tot - e_qchem) < 1e-5

    def test_rhf_d4(self):
        print('-------- RHF with D4 ----------')
        mf = gpu_scf.RHF(mol_sph).density_fit(auxbasis='def2-tzvpp-jkfit')
        mf.disp = 'd4'
        e_tot = mf.kernel()
        e_ref = -76.06343941431916 #-76.0669672259
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_to_cpu(self):
        mf = gpu_scf.RHF(mol_sph).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        assert isinstance(mf, cpu_df_jk._DFHF)
        e_cpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_to_gpu(self):
        mf = cpu_scf.RHF(mol_sph).density_fit()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        assert isinstance(mf, gpu_df_jk._DFHF)
        e_gpu = mf.kernel()
        assert np.abs(e_cpu - e_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for restricted Hartree-Fock")
    unittest.main()
