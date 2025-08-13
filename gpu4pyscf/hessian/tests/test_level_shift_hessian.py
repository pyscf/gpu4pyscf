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

import pyscf
import numpy as np
import unittest
import pytest
from gpu4pyscf.dft import RKS, UKS
from gpu4pyscf.scf import HF, UHF
from pyscf.hessian import thermo

def setUpModule():
    global mol_close, mol_open

    mol_close = pyscf.M(
        atom = '''
            O     -1.168500    0.182500    0.000000
            O      1.114600    0.210300    0.000000
            C      0.053800   -0.392700    0.000000
            H     -0.328661   -1.494191   -0.538879
            H     -1.582685    0.639818    1.199294
        ''',
        basis = '6-31g',
        charge = 0,
        spin = 0,
        output='/dev/null',
        verbose = 0,
    )

    mol_open = pyscf.M(
        atom = '''
            O     -1.168500    0.182500    0.000000
            O      1.114600    0.210300    0.000000
            C      0.053800   -0.392700    0.000000
            H     -0.328661   -1.494191   -0.538879
        ''',
        basis = '6-31g',
        charge = 0,
        spin = 1,
        output='/dev/null',
        verbose = 0,
    )

def tearDownModule():
    global mol_close, mol_open
    mol_close.stdout.close()
    mol_open.stdout.close()
    del mol_close, mol_open

class KnownValues(unittest.TestCase):
    # All reference results from the same calculation with mf.level_shift = 0

    def test_level_shift_hessian_rks(self):
        mf = RKS(mol_close, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_close, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -189.52569283262818
        ref_frequency = np.array([-295.34653089,  653.58625595,  879.10396141, 1150.01342468,
            1208.68664975, 1340.42573432, 1349.27967164, 1761.31892282,
            1792.17470571])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-2

    def test_level_shift_hessian_uks(self):
        mf = UKS(mol_open, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_open, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.92925230031926
        ref_frequency = np.array([ 292.72066163,  877.69576251, 1082.27549661, 1277.57877335,
            1693.48392029, 1760.06314071])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-2

    def test_level_shift_hessian_rhf(self):
        mf = HF(mol_close)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_close, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.53825152772055
        ref_frequency = np.array([-286.23234581,  703.64600407,  988.93397474, 1234.5357708 ,
            1246.90357193, 1391.30937505, 1443.06463755, 1774.96137283,
            1818.52710482])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-2

    def test_level_shift_hessian_uhf(self):
        mf = UHF(mol_open)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_open, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.00032587095123
        ref_frequency = np.array([ 624.64398354, 1030.20793768, 1183.88711991, 1362.44325456,
            1694.04574717, 1798.01003435])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-2

if __name__ == "__main__":
    print("Tests for HF and KS hessian with level shift")
    unittest.main()
