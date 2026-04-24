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
from pyscf import gto
from gpu4pyscf.tdscf import uhf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = """
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587"""
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_tda_vs_qchem(self):
        '''
        $rem
        METHOD HF
        BASIS cc-pvdz
        SPIN_FLIP true
        UNRESTRICTED true
        CIS_N_ROOTS 4
        CIS_TRIPLETS false
        CIS_DER_NUMSTATE 2
        CALC_NAC true
        SYMMETRY false
        SYM_IGNORE true
        BASIS_LIN_DEP_THRESH 12
        CIS_CONVERGENCE 8
        $end

        $derivative_coupling
        comment
        1 3
        $end

        ---------------------------------------------------
        SF-CIS derivative coupling without ETF
        Atom         X              Y              Z
        ---------------------------------------------------
        1       0.009187       0.000000      -0.000000
        2       0.104593      -0.000000       0.000000
        3       0.104593      -0.000000      -0.000000
        ---------------------------------------------------
        ---------------------------------------------------
        CIS Force Matrix Element
        Atom         X              Y              Z
        ---------------------------------------------------
        1      -0.093882       0.000000      -0.000000
        2       0.046941      -0.000000       0.000000
        3       0.046941      -0.000000      -0.000000
        ---------------------------------------------------
        ---------------------------------------------------
        SF-CIS derivative coupling with ETF
        Atom         X              Y              Z 
        ---------------------------------------------------
        1      -0.403747       0.000000      -0.000000
        2       0.201873      -0.000000       0.000000
        3       0.201873      -0.000000      -0.000000
        ---------------------------------------------------
        '''
        atom = '''
        O       0.0000000000     0.0000000000     0.0000000000
        H       0.0000000000    -0.7570000000     0.5870000000
        H       0.0000000000     0.7570000000     0.5870000000
        '''
        mol = gto.M(atom=atom, charge=0, spin=2, basis='cc-pvdz', verbose=0)
        mf = mol.UKS(xc='HF').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))

        ref_scaled = np.array([
            [ 0.009187,  0.000000, -0.000000],
            [ 0.104593, -0.000000,  0.000000],
            [ 0.104593, -0.000000, -0.000000]
        ])
        ref_etf = np.array([
            [-0.093882,  0.000000, -0.000000],
            [ 0.046941, -0.000000,  0.000000],
            [ 0.046941, -0.000000,  0.000000]
        ])
        ref_etf_scaled = np.array([
            [-0.403747,  0.000000, -0.000000],
            [ 0.201873, -0.000000,  0.000000],
            [ 0.201873, -0.000000, -0.000000]
        ])

        self.assertAlmostEqual(np.max(np.abs(np.abs(tdnac.de_scaled) - np.abs(ref_scaled))), 0, 4)
        self.assertAlmostEqual(np.max(np.abs(np.abs(tdnac.de_etf) - np.abs(ref_etf))), 0, 4)
        self.assertAlmostEqual(np.max(np.abs(np.abs(tdnac.de_etf_scaled) - np.abs(ref_etf_scaled))), 0, 4)

    def test_b3lyp_tddft_vs_cpu(self):
        atom = '''
        C      0.000000    0.000000    0.000000
        O      0.000000    0.000000    1.205000
        H     -0.937704    0.000000   -0.513544
        H      0.937704    0.100000   -0.513544
        '''
        mol = gto.M(atom=atom, charge=0, spin=2, basis='cc-pvdz', verbose=0)
        mf = mol.UKS(xc='B3LYP').to_gpu().run()
        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))

        B3LYP_REF_ETF = np.array([
            [ 2.50608626e-03,  3.33283037e-04,  3.17791471e-05],
            [-9.06831663e-04, -1.32591849e-04, -1.33038220e-05],
            [ 1.11121960e-03, -3.60580919e-02, -3.53482487e-04],
            [-2.71053372e-03,  3.58574323e-02,  3.35164345e-04],
        ])

        B3LYP_REF_FULL = np.array([
            [-2.98860503e-03, -3.83780059e-04, -2.82220009e-05],
            [ 7.40415810e-04,  1.27679743e-04,  1.47507459e-05],
            [-1.53440277e-03,  4.54561852e-02,  6.15100183e-04],
            [ 3.28538114e-03, -4.52414485e-02, -5.96649158e-04],
        ])
        
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(B3LYP_REF_FULL)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(B3LYP_REF_ETF)).max(), 0, 5)

    def test_mcol_lda(self):
        mf = self.mol.UKS(xc='SVWN').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[-0.016524445, -0., 0.], [0.0314787174, 0., -0.], [0.0314787174, 0., 0.]])
        ref_de_etf = np.array([[-0.1597263295, -0., 0.], [0.0798628981, 0., -0.], [0.0798628981, 0., 0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[0.0165439706, 0., 0.], [-0.0314291432, -0., 0.], [-0.0314291432, -0., -0.]])
        ref_de_etf = np.array([[0.1595859482, 0., -0.], [-0.0797927058, -0., 0.], [-0.0797927058, -0., -0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

    def test_mcol_tpss(self):
        mf = self.mol.UKS(xc='TPSS').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[-0.0217746301, 0., -0.], [0.0310642837, -0., 0.], [0.0310642837, -0., -0.]])
        ref_de_etf = np.array([[-0.1499947871, 0., -0.], [0.0750031808, -0., 0.], [0.0750031808, -0., -0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[-0.0219158391, 0., -0.], [0.0310488458, -0., 0.], [0.0310488458, -0., -0.]])
        ref_de_etf = np.array([[-0.149851143, 0., -0.], [0.0749313759, -0., 0.], [0.0749313759, -0., -0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

    def test_mcol_cam(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').to_gpu().run()
        td = uhf.SpinFlipTDA(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[-0.0183340396, -0., 0.], [0.0310193662, -0., -0.], [0.0310193662, 0., -0.]])
        ref_de_etf = np.array([[-0.1511503886, -0., -0.], [0.0755699221, -0., 0.], [0.0755699221, 0., 0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

        td = uhf.SpinFlipTDHF(mf).set(extype=1, collinear='mcol', collinear_samples=20).run()
        tdnac = td.NAC()
        tdnac.kernel(states=(1, 3))
        ref_de = np.array([[0.0183795234, -0., 0.], [-0.0309517357, -0., -0.], [-0.0309517357, 0., 0.]])
        ref_de_etf = np.array([[0.1509437772, -0., -0.], [-0.0754665642, -0., 0.], [-0.0754665642, 0., 0.]])
        self.assertAlmostEqual(abs(abs(tdnac.de) - abs(ref_de)).max(), 0, 5)
        self.assertAlmostEqual(abs(abs(tdnac.de_etf) - abs(ref_de_etf)).max(), 0, 5)

if __name__ == '__main__':
    print('Full Tests for spin-flip TDA and TDDFT non-adiabatic coupling vectors with multicollinear functionals')
    unittest.main()
