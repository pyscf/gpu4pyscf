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
# import cupy as cp
from pyscf import gto, lib
from gpu4pyscf.dft import rks
import gpu4pyscf.tdscf.ris as ris

PLACES = 3


def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e, xy = np.linalg.eig(np.asarray(h))
    sorted_indices = np.argsort(e)
    
    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]
    
    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # methanol
        atom = '''
        C         -4.89126        3.29770        0.00029
        H         -5.28213        3.05494       -1.01161
        O         -3.49307        3.28429       -0.00328
        H         -5.28213        2.58374        0.75736
        H         -5.23998        4.31540        0.27138
        H         -3.22959        2.35981       -0.24953
        '''
        mol = gto.M(atom=atom, basis='def2-svp',
                    output = '/dev/null',  # Suppress excessive log output
                    verbose=3)
        cls.mol = mol

        # Initialize DFT calculations with different functionals
        cls.mf_pbe = rks.RKS(mol, xc='pbe').density_fit().to_gpu().run()
        cls.mf_pbe0 = rks.RKS(mol, xc='pbe0').density_fit().to_gpu().run()
        cls.mf_wb97x = rks.RKS(mol, xc='wb97x').density_fit().to_gpu().run()
        cls.nstates = 5  # Test the first 3 excited states

    @classmethod
    def tearDownClass(cls):
        # Close the molecule's output stream
        cls.mol.stdout.close()

    ''' TDA '''
    def test_tda_pbe(self):
        """Test TDA-ris method with PBE functional"""
        mf = self.mf_pbe
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc     = td.oscillator_strength.get()

        # Reference energies  (in eV) and oscillator strengths
        ref_energies = [6.4051356, 7.7574377, 8.3537016, 8.8283863, 9.0716448]  
        ref_fosc     = [0.0022244, 0.0258598, 0.0046042, 0.0246019, 0.0465311]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(), 0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tda_pbe0(self):
        """Test TDA-ris method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc     = td.oscillator_strength.get()

        ref_energies = [7.1133437, 8.8369608, 9.1168451, 9.8721018, 10.1223936]  
        ref_fosc     = [0.0017021, 0.0521520, 0.0050341, 0.0279539, 0.0474464]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(), 0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tda_wb97x(self):
        """Test TDA-ris method with wB97x functional"""
        mf = self.mf_wb97x
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc = td.oscillator_strength.get()

        ref_energies = [7.4170489, 9.5510082, 9.5614233, 10.4997110, 10.8246613]  
        ref_fosc     = [0.0011733, 0.0036317, 0.0751882, 0.0260979, 0.0517084]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    ''' TDDFT '''
    def test_tddft_pbe(self):
        """Test TDDFT-ris method with PBE functional"""
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc     = td.oscillator_strength.get()

        # Reference energies  (in eV) and oscillator strengths
        ref_energies = [6.4017024, 7.7506866, 8.3276081, 8.8191490, 9.0424995]  
        ref_fosc     = [0.0017676, 0.0243237, 0.0039870, 0.0218969, 0.0430373]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tddft_pbe0(self):
        """Test TDDFT-ris method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc     = td.oscillator_strength.get()

        ref_energies = [7.1097179, 8.8275514, 9.0926926, 9.8638262, 10.0952592]  
        ref_fosc     = [0.0013409, 0.0493553, 0.0044438, 0.0251263, 0.0464966]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tddft_wb97x(self):
        """Test TDDFT-ris method with wB97x functional"""
        mf = self.mf_wb97x
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies.get()
        fosc = td.oscillator_strength.get()

        ref_energies = [7.4135534, 9.5252544, 9.5514112, 10.4925209, 10.7958272]  
        ref_fosc     = [0.0009041, 0.0032947, 0.0719354, 0.0237669, 0.0547606]  

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)
    
    def test_tddft_pbe_get_ab(self):
        """Test TDDFT-ris get_ab method with PBE0 functional"""
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies.get()
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, PLACES)

    def test_tddft_pbe0_get_ab(self):
        """Test TDDFT-ris get_ab method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies.get()
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, PLACES)

    def test_tddft_wb97x_get_ab(self):
        """Test TDDFT-ris get_ab method with wb97x functional"""
        mf = self.mf_wb97x
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='sp', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies.get()
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, 2) # TODO: change to PLACES


if __name__ == "__main__":
    print("Full Tests for TDDFT-RIS with PBE, PBE0, and wB97x")
    unittest.main()
