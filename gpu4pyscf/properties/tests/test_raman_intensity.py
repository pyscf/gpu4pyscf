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
import cupy as cp
import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf as rhf
from gpu4pyscf.properties.raman import eval_raman_intensity, \
    polarizability_derivative_numerical_dx, polarizability_derivative_numerical_dEdE

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

def make_mf(mol, xc = None, if_density_fitting = False, pcm = None):
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
    if pcm is not None:
        mf = mf.PCM()
        mf.with_solvent.method = pcm
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

    def test_polarizability_derivative_pcm_with_response(self):
        mf = make_mf(mol, xc = "pbe", if_density_fitting = True, pcm = "IEF-PCM")
        mf.with_solvent.equilibrium_solvation = True

        # reference_dalpha_dx = polarizability_derivative_numerical_dx(mf)
        # print(repr(reference_dalpha_dx))
        reference_dalpha_dx = cp.array([[[[-5.85154755e+00,  1.61904722e-12,  4.14031609e+00],
         [ 1.61904722e-12,  5.85922257e+00,  2.76859473e-11],
         [ 4.14031609e+00,  2.76859473e-11, -1.69249870e-03]],

        [[ 2.53613504e-05,  5.85695360e+00, -5.33201933e-06],
         [ 5.85695360e+00,  1.02842170e-05,  4.14704964e+00],
         [-5.33201933e-06,  4.14704964e+00,  1.78506552e-05]],

        [[ 4.13840898e+00, -6.33052391e-06,  3.75018198e-03],
         [-6.33052391e-06,  4.14964657e+00, -4.47051833e-06],
         [ 3.75018198e-03, -4.47051833e-06, -8.26404851e+00]]],


       [[[ 9.58595658e-01, -7.01008840e-11,  2.67198752e+00],
         [-7.01008840e-11, -9.58744588e-01, -1.60125967e-12],
         [ 2.67198752e+00, -1.60125967e-12, -2.07134114e-05]],

        [[ 1.93758360e-05, -9.58467670e-01, -6.45860809e-07],
         [-9.58467670e-01,  1.72479186e-05,  2.67190787e+00],
         [-6.45860809e-07,  2.67190787e+00,  1.83121021e-05]],

        [[ 3.11962800e-01, -5.86895840e-06,  2.25307751e-04],
         [-5.86895840e-06,  3.11870646e-01, -2.50136009e-06],
         [ 2.25307751e-04, -2.50136009e-06,  1.22581997e+01]]],


       [[[ 1.09083361e+01, -1.24283694e-11, -2.74816040e+00],
         [-1.24283694e-11, -2.45820330e-02, -1.24194071e-11],
         [-2.74816040e+00, -1.24194071e-11,  1.27117129e+00]],

        [[-1.42096974e-05,  2.20279590e+00, -3.49855589e-07],
         [ 2.20279590e+00, -1.51766715e-05, -1.79639060e+00],
         [-3.49855589e-07, -1.79639060e+00, -1.55562230e-05]],

        [[-1.96312222e+00, -1.74807468e-06,  3.49734774e+00],
         [-1.74807468e-06, -1.00928446e+00, -1.23249438e-06],
         [ 3.49734774e+00, -1.23249438e-06, -1.32647533e+00]]],


       [[[-3.00763992e+00, -3.32261695e+00, -2.03206228e+00],
         [-3.32261695e+00, -2.43786597e+00, -4.11211454e-01],
         [-2.03206228e+00, -4.11211454e-01, -6.34680096e-01]],

        [[-1.39333696e+00, -3.55060211e+00, -4.12578537e-01],
         [-3.55060211e+00, -8.03607314e+00, -2.51123606e+00],
         [-4.12578537e-01, -2.51123606e+00, -1.10204979e+00]],

        [[-1.24358791e+00, -4.12294374e-01, -1.75066234e+00],
         [-4.12294374e-01, -1.72608323e+00, -3.03363728e+00],
         [-1.75066234e+00, -3.03363728e+00, -1.33387197e+00]]],


       [[[-3.00768507e+00,  3.32263881e+00, -2.03206618e+00],
         [ 3.32263881e+00, -2.43792543e+00,  4.11228022e-01],
         [-2.03206618e+00,  4.11228022e-01, -6.34708232e-01]],

        [[ 1.39332732e+00, -3.55059997e+00,  4.12573280e-01],
         [-3.55059997e+00,  8.03605857e+00, -2.51123647e+00],
         [ 4.12573280e-01, -2.51123647e+00,  1.10203586e+00]],

        [[-1.24357111e+00,  4.12291025e-01, -1.75066085e+00],
         [ 4.12291025e-01, -1.72606135e+00,  3.03363549e+00],
         [-1.75066085e+00,  3.03363549e+00, -1.33385687e+00]]]])

        test_dalpha_dx = polarizability_derivative_numerical_dEdE(mf, dE = 1e-3)

        assert np.linalg.norm(test_dalpha_dx - reference_dalpha_dx) < 3e-3

    def test_polarizability_derivative_pcm_without_response(self):
        mf = make_mf(mol, xc = "pbe0", if_density_fitting = True, pcm = "IEF-PCM")
        assert mf.with_solvent.equilibrium_solvation is False

        # reference_dalpha_dx = polarizability_derivative_numerical_dx(mf)
        # print(repr(reference_dalpha_dx))
        reference_dalpha_dx = cp.array([[[[-4.20497968e+00, -8.41795791e-12,  2.97334791e+00],
         [-8.41795791e-12,  4.20483842e+00,  1.48424529e-11],
         [ 2.97334791e+00,  1.48424529e-11, -3.19048115e-04]],

        [[ 2.94336200e-06,  4.20492755e+00, -1.20366322e-06],
         [ 4.20492755e+00, -4.58817873e-07,  2.97343284e+00],
         [-1.20366322e-06,  2.97343284e+00,  1.27436639e-06]],

        [[ 2.97261596e+00, -2.56039742e-07,  1.88633188e-04],
         [-2.56039742e-07,  2.97326106e+00, -1.81140794e-07],
         [ 1.88633188e-04, -1.81140794e-07, -5.94658271e+00]]],


       [[[ 6.51791924e-01, -1.32289741e-11,  1.91600403e+00],
         [-1.32289741e-11, -6.51799118e-01, -1.40903253e-11],
         [ 1.91600403e+00, -1.40903253e-11, -4.34067005e-06]],

        [[-2.30362396e-06, -6.51792239e-01,  2.19188279e-07],
         [-6.51792239e-01, -1.73000814e-06,  1.91598677e+00],
         [ 2.19188279e-07,  1.91598677e+00, -2.01672634e-06]],

        [[ 2.41190094e-01,  7.83213859e-07,  8.23630214e-06],
         [ 7.83213859e-07,  2.41186648e-01,  3.22834293e-07],
         [ 8.23630214e-06,  3.22834293e-07,  8.91905387e+00]]],


       [[[ 7.92569013e+00,  5.96714697e-11, -2.00631090e+00],
         [ 5.96714697e-11,  9.99092754e-03,  4.71026096e-11],
         [-2.00631090e+00,  4.71026096e-11,  9.28001674e-01]],

        [[-7.36018961e-05,  1.58924976e+00,  3.35767561e-05],
         [ 1.58924976e+00,  2.17540723e-05, -1.25316110e+00],
         [ 3.35767561e-05, -1.25316110e+00,  8.14465650e-06]],

        [[-1.44781488e+00,  4.14390360e-06,  2.50719495e+00],
         [ 4.14390360e-06, -6.94875358e-01,  2.38367120e-06],
         [ 2.50719495e+00,  2.38367120e-06, -9.91035000e-01]]],


       [[[-2.18628141e+00, -2.40209072e+00, -1.44150319e+00],
         [-2.40209072e+00, -1.78149555e+00, -3.26169747e-01],
         [-1.44150319e+00, -3.26169747e-01, -4.63838222e-01]],

        [[-1.03420376e+00, -2.57116044e+00, -3.26161301e-01],
         [-2.57116044e+00, -5.83811154e+00, -1.81810751e+00],
         [-3.26161301e-01, -1.81810751e+00, -8.03576978e-01]],

        [[-8.82942445e-01, -3.26261291e-01, -1.25371011e+00],
         [-3.26261291e-01, -1.25977527e+00, -2.17154667e+00],
         [-1.25371011e+00, -2.17154667e+00, -9.90784219e-01]]],


       [[[-2.18628599e+00,  2.40208438e+00, -1.44149276e+00],
         [ 2.40208438e+00, -1.78147114e+00,  3.26165438e-01],
         [-1.44149276e+00,  3.26165438e-01, -4.63849437e-01]],

        [[ 1.03420590e+00, -2.57117731e+00,  3.26170425e-01],
         [-2.57117731e+00,  5.83814087e+00, -1.81811934e+00],
         [ 3.26170425e-01, -1.81811934e+00,  8.03576825e-01]],

        [[-8.82928937e-01,  3.26270834e-01, -1.25371248e+00],
         [ 3.26270834e-01, -1.25976770e+00,  2.17155401e+00],
         [-1.25371248e+00,  2.17155401e+00, -9.90793721e-01]]]])

        test_dalpha_dx = polarizability_derivative_numerical_dEdE(mf)

        assert np.linalg.norm(test_dalpha_dx - reference_dalpha_dx) < 3e-3

    def test_raman_pbe0_densityfitting_pcm_with_response(self):
        ### Q-Chem input
        # $rem
        # JOBTYPE       FREQ
        # METHOD        PBE0
        # BASIS         def2-svp
        # RI_J TRUE
        # RI_K TRUE
        # AUX_BASIS RIJK-def2-qzvpp
        # SOLVENT_METHOD PCM
        # THRESH 14
        # XC_GRID 000099000590
        # NL_GRID 000050000194
        # DORAMAN TRUE
        # SYMMETRY FALSE
        # SYM_IGNORE TRUE
        # MEM_STATIC 8000
        # MEM_TOTAL  80000
        # $end

        # $PCM
        # theory IEFPCM
        # Radii BONDI
        # HeavyPoints 302
        # HPoints 302
        # vdwScale 1.2
        # $end

        # $solvent
        # dielectric 78.3553
        # $end
        reference_frequencies = np.array([1268.56, 1268.80, 1269.18, 1501.37, 1501.74, 3120.07, 3280.46, 3280.48, 3280.51])
        reference_raman_intensities = np.array([2.109,   2.110,   2.130,  47.484,  47.585, 266.057, 115.212, 115.024, 115.119])
        reference_depolarization_ratio = np.array([0.750, 0.750, 0.750, 0.750, 0.750, 0.000, 0.750, 0.750, 0.750])

        mf = make_mf(mol, xc = "pbe0", if_density_fitting = True, pcm = "IEF-PCM")
        mf.with_solvent.equilibrium_solvation = True # Q-Chem has PCM response turned on for polarizability calculation,
                                                     # and, of course, its numerical differentiation.

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 3.0
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.5
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

    def test_raman_pbe0_densityfitting_pcm_without_response(self):
        # This is a consistent test, because Henry cannot find external reference for Raman + PCM without electric field response.
        reference_frequencies = np.array(
            [1268.31466976, 1268.53875777, 1268.89123638, 1501.31720611, 1501.68181867,
             3118.86671417, 3279.47203975, 3279.47851944, 3279.50991252]
        )
        reference_raman_intensities = np.array(
            [  1.29990936,   1.29947435,   1.30029895,  24.61515597,  24.61708687,
             137.5133924 ,  58.44938597,  58.46380702,  58.4575389 ]
        )
        reference_depolarization_ratio = np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.0, 0.75, 0.75, 0.75])

        mf = make_mf(mol, xc = "pbe0", if_density_fitting = True, pcm = "IEF-PCM")
        assert mf.with_solvent.equilibrium_solvation is False

        test_frequencies, test_raman_intensities, test_depolarization_ratio = eval_raman_intensity(mf)

        assert np.linalg.norm(test_frequencies - reference_frequencies) < 0.1
        assert np.linalg.norm(test_raman_intensities - reference_raman_intensities) < 0.01
        assert np.linalg.norm(test_depolarization_ratio - reference_depolarization_ratio) <= 0.001

if __name__ == "__main__":
    print("Full Tests for Raman intensity")
    unittest.main()
