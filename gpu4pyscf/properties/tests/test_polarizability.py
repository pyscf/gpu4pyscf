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
from gpu4pyscf.properties import polarizability

try:
    from pyscf.prop import polarizability as polar
except Exception:
    polar = None

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

def run_dft_polarizability(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    polar = polarizability.eval_polarizability(mf)
    return e_dft, polar

def run_dft_df_polarizability(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    polar = polarizability.eval_polarizability(mf)
    return e_dft, polar

def _vs_cpu(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_gpu = mf.kernel()
    polar_gpu = polarizability.eval_polarizability(mf)
    
    mf_cpu = rks_cpu.RKS(mol, xc=xc)
    mf_cpu.conv_tol = 1e-12
    e_cpu = mf_cpu.kernel()
    polar_cpu = polar.rhf.Polarizability(mf_cpu).polarizability()

    assert np.abs(e_gpu - e_cpu) < 1e-5
    assert np.linalg.norm(polar_cpu - polar_gpu) < 1e-3

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem

    $rem
    MEM_STATIC    2000
    JOBTYPE       POLARIZABILITY
    METHOD        b3lyp
    RESPONSE_POLAR 0
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     12
    XC_GRID 000099000590
    SYMMETRY FALSE
    SYM_IGNORE = TRUE
    $end

     -----------------------------------------------------------------------------
    Polarizability tensor      [a.u.]
      8.5899463     -0.0000000     -0.0000000
     -0.0000000      6.0162267     -0.0000000
     -0.0000000     -0.0000000      7.5683123

    $rem
    MEM_STATIC    2000
    JOBTYPE       POLARIZABILITY
    METHOD        b3lyp
    RESPONSE_POLAR -1
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
    Polarizability tensor      [a.u.]
        8.5902540     -0.0000000     -0.0000000
        0.0000000      6.0167648      0.0000000
       -0.0000000     -0.0000000      7.5688173
    
    '''
    def test_rks_b3lyp(self):
        print('-------- RKS B3LYP -------------')
        e_tot, polar = run_dft_polarizability('B3LYP')
        assert np.allclose(e_tot, -76.4666494276)
        qchem_polar = np.array([ [ 8.5899463,    -0.0000000,     -0.0000000],
                                 [-0.0000000,     6.0162267,     -0.0000000],
                                 [-0.0000000,    -0.0000000,      7.5683123]])
        assert np.allclose(polar, qchem_polar)

    def test_rks_b3lyp_df(self):
        print('-------- RKS density fitting B3LYP -------------')
        e_tot, polar = run_dft_df_polarizability('B3LYP')
        assert np.allclose(e_tot, -76.4666819553)
        qchem_polar = np.array([ [  8.5902540,    -0.0000000,     -0.0000000],
                                 [  0.0000000,     6.0167648,     -0.0000000],
                                 [ -0.0000000,    -0.0000000,      7.5688173]])
        assert np.allclose(polar, qchem_polar)

    @unittest.skipIf(polar is None, "Skipping test if pyscf.properties is not installed")
    def test_cpu(self):
        _vs_cpu('b3lyp')

if __name__ == "__main__":
    print("Full Tests for polarizabillity")
    unittest.main()
