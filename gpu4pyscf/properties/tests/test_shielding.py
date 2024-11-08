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
from gpu4pyscf.properties import shielding

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

def run_dft_nmr_shielding(xc):
    mf = rks.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    tensor = shielding.eval_shielding(mf)
    return e_dft, tensor

def run_dft_df_nmr_shielding(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = grids_level
    e_dft = mf.kernel()
    tensor = shielding.eval_shielding(mf)
    return e_dft, tensor


class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem

    $rem
    METHOD             b3lyp
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     12
    XC_GRID 000099000590
    SYMMETRY FALSE
    SYM_IGNORE  TRUE
    MOPROP               1
    MOPROP_PERTNUM       0  ! do all perturbations at once
    MOPROP_CONV_1ST      10  ! sets the CPSCF convergence threshold
    MOPROP_DIIS_DIM_SS   4  ! no. of DIIS subspace vectors
    MOPROP_MAXITER_1ST 100  ! max iterations
    MOPROP_DIIS          5  ! turns on DIIS (=0 to turn off)
    MOPROP_DIIS_THRESH   1
    MOPROP_DIIS_SAVE     0
    $end

       ATOM           ISOTROPIC        ANISOTROPIC       REL. SHIFTS
------------------------------------------------------------------------
   Atom O     1     332.07586444       45.66898820
   Atom H     2      31.39150070       19.36432602
   Atom H     3      31.39060707       19.36602790

    $rem
    METHOD             b3lyp
    BASIS         def2-tzvpp
    SCF_CONVERGENCE     10
    XC_GRID 000099000590
    ri_j        True
    ri_k        True
    aux_basis     RIJK-def2-tzvpp
    THRESH        14
    SYMMETRY FALSE
    SYM_IGNORE  TRUE
    MOPROP               1
    MOPROP_PERTNUM       0  ! do all perturbations at once
    MOPROP_CONV_1ST      10  ! sets the CPSCF convergence threshold
    MOPROP_DIIS_DIM_SS   4  ! no. of DIIS subspace vectors
    MOPROP_MAXITER_1ST 100  ! max iterations
    MOPROP_DIIS          5  ! turns on DIIS (=0 to turn off)
    MOPROP_DIIS_THRESH   1
    MOPROP_DIIS_SAVE     0
    $end

    
       ATOM           ISOTROPIC        ANISOTROPIC       REL. SHIFTS
------------------------------------------------------------------------
   Atom O     1     332.07961083       45.66777197
   Atom H     2      31.39250594       19.36477246
   Atom H     3      31.39160966       19.36647972
    
    '''
    def test_rks_b3lyp(self):
        print('-------- RKS B3LYP -------------')
        e_tot, tensor = run_dft_nmr_shielding('B3LYP')
        nmr_total = tensor[0].get()+tensor[1].get()
        isotropic_pyscf = np.array([nmr_total[0].trace()/3, nmr_total[1].trace()/3, nmr_total[2].trace()/3])
        isotropic_qchem = np.array([332.07586444, 31.39150070, 31.39060707])
        assert np.allclose(e_tot, -76.4666494276)
        assert np.allclose(isotropic_pyscf, isotropic_qchem, rtol=1.0E-4)

    def test_rks_b3lyp_df(self):
        print('-------- RKS density fitting B3LYP -------------')
        e_tot, tensor = run_dft_df_nmr_shielding('B3LYP')
        nmr_total = tensor[0].get()+tensor[1].get()
        isotropic_pyscf = np.array([nmr_total[0].trace()/3, nmr_total[1].trace()/3, nmr_total[2].trace()/3])
        isotropic_qchem = np.array([332.07961083, 31.39250594, 31.39160966])
        assert np.allclose(e_tot, -76.4666819553)
        assert np.allclose(isotropic_pyscf, isotropic_qchem, rtol=1.0E-4)


if __name__ == "__main__":
    print("Full Tests for nmr shielding constants")
    unittest.main()
