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
from pyscf.dft import rks as rks_cpu
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.properties import ir

try:
    from pyscf.prop import infrared
except Exception:
    infrared = None

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

def _vs_cpu(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.conv_tol = 1e-13
    mf.conv_tol_cpscf = 1e-7
    e_gpu = mf.kernel()
    h = mf.Hessian()
    freq, intensity_gpu = ir.eval_ir_freq_intensity(mf, h)
    
    mf_cpu = rks_cpu.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf_cpu.conv_tol = 1e-13
    mf_cpu.conv_tol_cpscf = 1e-7
    e_cpu = mf_cpu.kernel()
    mf_ir = infrared.rks.Infrared(mf_cpu).run()
    assert np.abs(e_gpu - e_cpu) < 1e-5
    assert np.linalg.norm(mf_ir.ir_inten - intensity_gpu.get()) < 1e-2


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
        assert np.allclose(freq, qchem_freq, rtol=1e-03)
        assert np.allclose(intensity, qchem_intensity, rtol=1e-02)

    @unittest.skipIf(infrared is None, "Skipping test if pyscf.properties is not installed")
    def test_cpu(self):
        _vs_cpu('b3lyp')

if __name__ == "__main__":
    print("Full Tests for ir intensity")
    unittest.main()
