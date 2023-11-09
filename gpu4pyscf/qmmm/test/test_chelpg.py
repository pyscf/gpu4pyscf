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
from gpu4pyscf.dft import rks
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

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft_chelpg(xc, deltaR):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    q = chelpg.eval_chelpg_layer_gpu(mf, deltaR=deltaR)
    return e_dft, q
    

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
    '''
    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot, q = run_dft_chelpg('B3LYP', 0.1)
        assert np.allclose(e_tot, -76.4666495181)
        assert np.allclose(q, np.array([-0.712558, 0.356292, 0.356266]))
        

if __name__ == "__main__":
    print("Full Tests for SCF")
    unittest.main()