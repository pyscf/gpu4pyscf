# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf import dft, tdscf
from gpu4pyscf.properties import c6


def diagonalize_casida(a, b, nroots=4):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a        , b       ],
                  [-b.conj(),-a.conj()]])
    e = np.linalg.eig(np.asarray(h))[0]
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e = np.linalg.eig(np.asarray(a))[0]
    lowest_e = np.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        cls.mol = mol.build()

        cls.mf = cls.mol.RHF().to_gpu().run()
        
        mf_lda = cls.mol.RKS().to_gpu()
        mf_lda.xc = 'lda, vwn'
        cls.mf_lda = mf_lda.run(conv_tol=1e-10)

        mf_b3lyp = cls.mol.RKS().to_gpu()
        mf_b3lyp.xc = 'b3lyp'
        cls.mf_b3lyp = mf_b3lyp.run(conv_tol=1e-10)

        mol_h2o = gto.Mole()
        mol_h2o.verbose = 0
        mol_h2o.output = '/dev/null'
        mol_h2o.atom = """
        O 0.0000 0.0000 0.1173
        H 0.0000 0.7572 -0.4692
        H 0.0000 -0.7572 -0.4692
        """
        mol_h2o.basis = '631g'
        cls.mol_h2o = mol_h2o.build()
        cls.mf_h2o = cls.mol_h2o.RKS().to_gpu().run(xc='b3lyp')

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        cls.mol_h2o.stdout.close()

    def test_full_spectrum_tda_lda(self):
        """Test full spectrum TDA solver against explicit diagonalization."""
        td_b = self.mf_lda.TDA().set(nstates=5)
        e_benchmark, xy_benchmark = td_b.kernel()
        f_oscillator_benchmark = td_b.oscillator_strength()

        td = self.mf_lda.TDA()
        c6._solve_full_spectrum(td)
        a, b = td.get_ab()
        ref_e = diagonalize_tda(a, nroots=5)
        f_oscillator = td.oscillator_strength()
        
        self.assertAlmostEqual(abs(td.e[:5] - ref_e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e[:5] - e_benchmark).max(), 0, 6)
        self.assertAlmostEqual(abs(f_oscillator[:5] - f_oscillator_benchmark).max(), 0, 6)
        
        x_vec = td.xy[0][0]
        norm = 2.0 * np.sum(x_vec**2)
        self.assertAlmostEqual(norm, 1.0, 5)

    def test_full_spectrum_tddft_lda(self):
        """Test full spectrum TDDFT solver (Pure) against explicit diagonalization."""
        td_b = self.mf_lda.TDDFT().set(nstates=5)
        e_benchmark, xy_benchmark = td_b.kernel()
        f_oscillator_benchmark = td_b.oscillator_strength()

        td = self.mf_lda.TDDFT()
        c6._solve_full_spectrum(td)
        f_oscillator = td.oscillator_strength()
        
        a, b = td.get_ab()
        ref_e = diagonalize_casida(a, b, nroots=5)
        
        self.assertAlmostEqual(abs(td.e[:5] - ref_e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e[:5] - e_benchmark).max(), 0, 6)
        self.assertAlmostEqual(abs(f_oscillator[:5] - f_oscillator_benchmark).max(), 0, 6)

    def test_full_spectrum_tddft_b3lyp(self):
        """Test full spectrum TDDFT solver (Hybrid) against explicit diagonalization."""
        td_b = self.mf_b3lyp.TDDFT().set(nstates=5)
        e_benchmark, xy_benchmark = td_b.kernel()
        f_oscillator_benchmark = td_b.oscillator_strength()

        td = self.mf_b3lyp.TDDFT()
        c6._solve_full_spectrum(td)
        f_oscillator = td.oscillator_strength()
        
        a, b = td.get_ab()
        ref_e = diagonalize_casida(a, b, nroots=5)
        
        self.assertAlmostEqual(abs(td.e[:5] - ref_e).max(), 0, 6)
        self.assertAlmostEqual(abs(td.e[:5] - e_benchmark).max(), 0, 6)
        self.assertAlmostEqual(abs(f_oscillator[:5] - f_oscillator_benchmark).max(), 0, 6)
        
        x_vec, y_vec = td.xy[0]
        norm = np.sum(x_vec**2) - np.sum(y_vec**2)
        self.assertAlmostEqual(norm, 0.5, 5)

    def test_calc_c6_sanity(self):
        """Test C6 calculation returns positive values and runs without error."""
        td = self.mf_lda.TDDFT()
        val = c6.calc_c6(td, td, n_grid=10)
        ref_e = np.array([ 0.35545732,  0.35545732,  0.54368678,  1.1144098 ,  1.1144098 ,  1.1598993 ,
            1.35171048,  1.41194799,  1.43205768,  1.43205769,  1.52953895,  1.52953895,
            1.57505632,  1.57505632,  1.60406147,  1.90474313,  1.9660721 ,  1.9660721 ,
            2.03275446,  2.09882741,  2.22934612,  2.22934612,  2.3315765 ,  2.91523625,
            24.10618119, 24.86256929, 25.2256636 , 25.2256636 , 25.32903837, 25.77488066])

        self.assertIsInstance(val, float)
        self.assertAlmostEqual(abs(td.e - ref_e).max(), 0, 6)
        self.assertAlmostEqual(val, 1.959496362180395, 6)
        self.assertTrue(val > 0.0)

    def test_calc_c6_symmetry(self):
        """Test symmetry: C6(A, B) should equal C6(B, A)."""
        td_a = self.mf_lda.TDA()     
        td_b = self.mf_h2o.TDDFT()   
        
        c6_ab = c6.calc_c6(td_a, td_b, n_grid=12)
        c6_ba = c6.calc_c6(td_b, td_a, n_grid=12)
        
        self.assertAlmostEqual(abs(c6_ab - c6_ba), 0, 9)
        self.assertAlmostEqual(abs(c6_ab - 4.83381668540887), 0, 9)

if __name__ == "__main__":
    print("Full Tests for C6 calculations")
    unittest.main()