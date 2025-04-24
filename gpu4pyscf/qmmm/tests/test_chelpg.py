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
from pyscf import lib
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.qmmm import chelpg

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

    global molu
    molu = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000)
    molu.output = '/dev/null'
    molu.build()
    molu.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    global molu
    molu.stdout.close()
    del mol

def run_dft_chelpg(xc, deltaR):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    q = chelpg.eval_chelpg_layer_gpu(mf, deltaR=deltaR)
    return e_dft, q.get()

def run_udft_chelpg(xc, deltaR):
    mf = uks.UKS(molu, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    q = chelpg.eval_chelpg_layer_gpu(mf, deltaR=deltaR)
    return e_dft, q.get()

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    $rem
    JOBTYP  SP
    METHOD  b3lyp
    BASIS   def2-tzvpp
    XC_GRID 000099000590
    CHELPG_DX 2
    CHELPG        TRUE
    SCF_CONVERGENCE 10
    $end

        Ground-State ChElPG Net Atomic Charges

     Atom                 Charge (a.u.)
  ----------------------------------------
      1 O                    -0.712558
      2 H                     0.356292
      3 H                     0.356266
  ----------------------------------------

       Atom                 Charge (a.u.)
  ----------------------------------------
      1 O                     0.046042
      2 H                     0.476984
      3 H                     0.476974
  ----------------------------------------
    '''
    def test_rks_b3lyp(self):
        print('-------- RKS B3LYP -------------')
        e_tot, q = run_dft_chelpg('B3LYP', 0.1)
        assert abs(e_tot - -76.4666495181) < 1e-6
        assert abs(q - np.array([-0.712558, 0.356292, 0.356266])).max() < 1e-5

    def test_uks_b3lyp(self):
        print('-------- UKS B3LYP -------------')
        e_tot, q = run_udft_chelpg('B3LYP', 0.1)
        assert abs(e_tot - -75.9987351018) < 1e-6
        assert abs(q - np.array([0.046042, 0.476984, 0.476974])).max() < 1e-5


if __name__ == "__main__":
    print("Full Tests for CHELPG")
    unittest.main()
