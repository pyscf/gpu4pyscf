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
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.properties import ir

lib.num_threads(8)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='def2tzvpp'
grids_level = 6

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

def run_dft_df_if(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = grids_level
    mf.conv_tol_cpscf = 1e-3
    e_dft = mf.kernel()
    h = mf.Hessian()
    h.auxbasis_response = 2 
    freq, intensity = ir.eval_ir_freq_intensity(mf, h)
    return e_dft, freq, intensity


class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem

    $rem
    MEM_STATIC    2000
    JOBTYPE       freq
    METHOD        b3lyp
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     12
    XC_GRID 000099000590
    SYMMETRY FALSE
    SYM_IGNORE = TRUE
    ri_j        True
    ri_k        True
    aux_basis     RIJK-def2-tzvpp
    THRESH        14
    $end

     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   Mode:                 1                      2                      3
 Frequency:      1630.86                3850.08                3949.35
 Force Cnst:      1.6970                 9.1261                 9.9412
 Red. Mass:       1.0829                 1.0450                 1.0818
 IR Active:          YES                    YES                    YES
 IR Intens:       69.334                  4.675                 47.943
 Raman Active:       YES                    YES                    YES
    
    '''

    def test_rks_b3lyp_df(self):
        print('-------- RKS density fitting B3LYP -------------')
        e_tot, freq, intensity = run_dft_df_if('B3LYP')
        assert np.allclose(e_tot, -76.4666819537)
        qchem_freq = np.array([1630.86, 3850.08, 3949.35])
        qchem_intensity = np.array([69.334, 4.675, 47.943])
        print(freq, intensity)
        assert np.allclose(freq, qchem_freq, rtol=1e-03)
        assert np.allclose(intensity, qchem_intensity, rtol=1e-02)



if __name__ == "__main__":
    print("Full Tests for ir intensity")
    unittest.main()