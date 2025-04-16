# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import rks_lowmem

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-tzvp'

def setUpModule():
    global mol_sph
    mol_sph = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
    )

def tearDownModule():
    global mol_sph
    del mol_sph

def run_dft(xc, mol, disp=None):
    mf = rks_lowmem.RKS(mol, xc=xc)
    mf.disp = disp
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rks_lda(self):
        print('------- LDA ----------------')
        e_tot = run_dft('SVWN', mol_sph)
        e_ref = mol_sph.RKS(xc='SVWN').kernel()
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-8

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP', mol_sph)
        e_ref = mol_sph.RKS(xc='B3LYP').kernel()
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-8

    def test_rks_wb97x(self):
        print('-------- wb97 -------------')
        e_tot = run_dft('wb97', mol_sph)
        e_ref = mol_sph.RKS(xc='wb97').kernel()
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-8

    def test_rks_wb97m(self):
        print('-------- wb97mv -------------')
        e_tot = run_dft('wb97mv', mol_sph)
        e_ref = mol_sph.RKS(xc='wb97mv').kernel()
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-7

if __name__ == "__main__":
    print("Full Tests for rks_lowmem")
    unittest.main()
