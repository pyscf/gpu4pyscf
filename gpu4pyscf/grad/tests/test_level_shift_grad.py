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
from gpu4pyscf.scf.hf_lowmem import RHF as HF_LOWMEM
from gpu4pyscf.dft.rks_lowmem import RKS as RKS_LOWMEM

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

    def test_level_shift_gradient_rks(self):
        mf = RKS(mol_close, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -189.52569283262818
        ref_gradient = np.array([
            [ 0.07222875, -0.05127681, -0.11885916],
            [ 0.00376853,  0.02164838, -0.01800676],
            [ 0.01062449,  0.04490839,  0.0718105 ],
            [-0.04677538, -0.04503511, -0.03373301],
            [-0.03984876,  0.02975217,  0.0987993 ],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_level_shift_gradient_uks(self):
        mf = UKS(mol_open, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -188.92925230031926
        ref_gradient = np.array([
            [ 0.00146016,  0.00819138, -0.01503009],
            [-0.00426437,  0.02476811, -0.01476034],
            [ 0.03320805,  0.00757202,  0.06555009],
            [-0.03041575, -0.04053383, -0.03575653],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_level_shift_gradient_rhf(self):
        mf = HF(mol_close)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -188.53825152772055
        ref_gradient = np.array([
            [ 0.06201409, -0.05796198, -0.1389516 ],
            [ 0.03321063,  0.03785769, -0.01981921],
            [ 0.00279401,  0.03633387,  0.08622744],
            [-0.0547028 , -0.05290103, -0.04192148],
            [-0.04331594,  0.03667145,  0.11446484],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_level_shift_gradient_uhf(self):
        mf = UHF(mol_open)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -188.00032587095123
        ref_gradient = np.array([
            [-9.48177927e-05, -6.48561043e-03, -2.05869163e-02],
            [ 2.91312662e-02,  3.60939518e-02, -1.78901925e-02],
            [ 1.47109479e-02,  2.13332766e-02,  8.39433701e-02],
            [-4.37473963e-02, -5.09416180e-02, -4.54662613e-02],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    # Lowmem

    def test_level_shift_gradient_rks_lowmem(self):
        mf = RKS_LOWMEM(mol_close, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-11

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -189.525657478194
        ref_gradient = np.array([
            [ 0.0722262 , -0.05127889, -0.11886791],
            [ 0.00377738,  0.02165158, -0.01800976],
            [ 0.01062201,  0.0449061 ,  0.07181856],
            [-0.04677732, -0.0450351 , -0.03373522],
            [-0.03985063,  0.02975333,  0.09880519],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_level_shift_gradient_rhf_lowmem(self):
        mf = HF_LOWMEM(mol_close)
        mf.conv_tol = 1e-11

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        ref_energy = -188.538338996066
        ref_gradient = np.array([
            [ 0.06199652, -0.05796828, -0.13895198],
            [ 0.03324866,  0.03785366, -0.0198227 ],
            [ 0.00276811,  0.03634416,  0.08623013],
            [-0.05469687, -0.05289908, -0.04192038],
            [-0.04331642,  0.03666955,  0.11446493],
        ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

if __name__ == "__main__":
    print("Tests for HF and KS gradient with level shift")
    unittest.main()
