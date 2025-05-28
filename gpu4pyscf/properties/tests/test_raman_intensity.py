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
import pytest
import numpy as np
import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf as rhf
from gpu4pyscf.properties.raman import eval_raman_intensity

def setUpModule():
    global mol

    atom = '''
    C      0.00000    0.00000    0.00000
    H      0.00000    0.00000    1.08900
    H      1.02672    0.00000   -0.36300
    H     -0.51336   -0.88916   -0.36300
    H     -0.51336    0.88916   -0.36300
    '''
    basis = 'def2-svp'

    mol = pyscf.M(atom=atom, basis=basis, max_memory=32000,
                  output='/dev/null', verbose=1)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def make_mf(mol, xc = None, if_density_fitting = False):
    if xc is not None:
        mf = rks.RKS(mol, xc = xc)
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
    else:
        mf = rhf.RHF(mol)
    mf.conv_tol = 1e-15
    mf.conv_tol_cpscf = 1e-10
    mf.direct_scf_tol = 1e-16
    mf.verbose = 0
    if if_density_fitting:
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
    mf.kernel()
    return mf

class KnownValues(unittest.TestCase):
    @pytest.mark.skip("Too slow, and probably nobody wants to use direct SCF to compute the Raman spectra of a big molecule")
    def test_raman_wb97mv(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       FREQ
        # METHOD        wB97m-v
        # BASIS         def2-svp
        # THRESH 14
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # DORAMAN TRUE
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 8000
        # MEM_TOTAL  80000
        # $end
        reference_frequencies = np.array([1282.51, 1282.55, 1282.74, 1516.80, 1516.84, 3114.63, 3263.94, 3264.04, 3264.12])
        reference_raman_intensities = np.array([1.695,  1.695,   1.695, 26.572, 26.571, 141.201, 59.844, 59.846,  59.843])
        reference_depolarization_ratio = np.array([0.750, 0.750, 0.750, 0.750, 0.750, 0.000, 0.750, 0.750, 0.750])

        mf = make_mf(mol, xc = "wb97m-v")

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 2.0
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.1
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

    def test_raman_wb97mv_densityfitting(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       FREQ
        # METHOD        wB97m-v
        # BASIS         def2-svp
        # RI_J TRUE
        # RI_K TRUE
        # AUX_BASIS RIJK-def2-qzvpp
        # THRESH 14
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # DORAMAN TRUE
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 8000
        # MEM_TOTAL  80000
        # $end
        reference_frequencies = np.array([1282.42, 1282.46, 1282.65, 1516.86, 1516.90, 3113.54, 3263.18, 3263.27, 3263.36])
        reference_raman_intensities = np.array([1.695,  1.696,   1.695, 26.563, 26.562, 141.275, 59.858, 59.860,  59.857])
        reference_depolarization_ratio = np.array([0.750, 0.750, 0.750, 0.750, 0.750, 0.000, 0.750, 0.750, 0.750])

        mf = make_mf(mol, xc = "wb97m-v", if_density_fitting = True)

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 2.0
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.1
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

    def test_raman_hf(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       FREQ
        # METHOD        hf
        # BASIS         def2-svp
        # THRESH 14
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # DORAMAN TRUE
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 8000
        # MEM_TOTAL  80000
        reference_frequencies = np.array([1439.50, 1439.50, 1439.51, 1648.55, 1648.55, 3166.20, 3301.71, 3301.73, 3301.75])
        reference_raman_intensities = np.array([1.361,  1.361,   1.361, 25.708, 25.709, 158.467, 71.770, 71.768,  71.768])
        reference_depolarization_ratio = np.array([0.750, 0.750, 0.750, 0.750, 0.750, 0.000, 0.750, 0.750, 0.750])

        mf = make_mf(mol)

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 2.0
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.1
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

    def test_raman_hf_densityfitting(self):
        # Reference the same as above, because the error introducted by density fitting is much smaller than the error
        # from hessian and polarizability derivative calculations.
        reference_frequencies = np.array([1439.50, 1439.50, 1439.51, 1648.55, 1648.55, 3166.20, 3301.71, 3301.73, 3301.75])
        reference_raman_intensities = np.array([1.361,  1.361,   1.361, 25.708, 25.709, 158.467, 71.770, 71.768,  71.768])
        reference_depolarization_ratio = np.array([0.750, 0.750, 0.750, 0.750, 0.750, 0.000, 0.750, 0.750, 0.750])

        mf = make_mf(mol, if_density_fitting = True)

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 2.0
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.1
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

if __name__ == "__main__":
    print("Full Tests for Raman intensity")
    unittest.main()
