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

PLACES = 4

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


def diagonalize_tda(a, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    e, xy = np.linalg.eig(np.asarray(a))
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
        mol = gto.M(atom=atom, basis='def2-svp', verbose=3)
        mol.output = '/dev/null'  # Suppress excessive log output
        cls.mol = mol.build()

        # Initialize DFT calculations with different functionals
        cls.mf_pbe = rks.RKS(mol, xc='pbe').density_fit().to_gpu().run()
        cls.mf_pbe0 = rks.RKS(mol, xc='pbe0').density_fit().to_gpu().run()
        cls.nstates = 5  # Test the first 3 excited states

    @classmethod
    def tearDownClass(cls):
        # Close the molecule's output stream
        cls.mol.stdout.close()

    ''' TDA '''
    def test_tda_pbe(self):
        """
        Test TDA-ris method with PBE functional
        Compare with traditional TDA
        """
        mf = self.mf_pbe
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.without_ris_approx = True
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize_tda(a, self.nstates)[0]
        td_benchmark = mf.TDA()
        td_benchmark.nstates=self.nstates
        td_benchmark.kernel()
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)
        self.assertAlmostEqual(abs(energies - td_benchmark.e).max(), 0, PLACES)

    def test_tda_pbe0(self):
        """
        Test TDA-ris method with PBE0 functional
        Compare with traditional TDA
        """
        mf = self.mf_pbe0
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.without_ris_approx = True
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize_tda(a, self.nstates)[0]
        td_benchmark = mf.TDA()
        td_benchmark.nstates=self.nstates
        td_benchmark.kernel()
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)
        self.assertAlmostEqual(abs(energies - td_benchmark.e).max(), 0, PLACES)

    def test_tda_pbe_with_xckernel(self):
        """
        Test TDA-ris method with PBE functional
        Compare with traditional TDA, using RIS integral with xc kernel.
        """
        mf = self.mf_pbe
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize_tda(a, self.nstates)[0]
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)

    def test_tda_pbe0_with_xckernel(self):
        """
        Test TDA-ris method with PBE0 functional
        Compare with traditional TDA, using RIS integral with xc kernel.
        """
        mf = self.mf_pbe0
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize_tda(a, self.nstates)[0]
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)

    ''' TDDFT '''
    def test_tddft_pbe(self):
        """
        Test TDDFT-ris method with PBE functional
        Compare with traditional TDDFT
        """
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.without_ris_approx = True
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]
        td_benchmark = mf.TDDFT()
        td_benchmark.nstates=self.nstates
        td_benchmark.kernel()
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)
        self.assertAlmostEqual(abs(energies - td_benchmark.e).max(), 0, PLACES)

    def test_tddft_pbe0(self):
        """
        Test TDDFT-ris method with PBE0 functional
        Compare with traditional TDDFT
        """
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.without_ris_approx = True
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]
        td_benchmark = mf.TDDFT()
        td_benchmark.nstates=self.nstates
        td_benchmark.kernel()
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)
        self.assertAlmostEqual(abs(energies - td_benchmark.e).max(), 0, PLACES)

    def test_tddft_pbe_with_xckernel(self):
        """
        Test TDDFT-ris method with PBE functional
        Compare with traditional TDDFT, using RIS integral with xc kernel.
        """
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)

    def test_tddft_pbe0_with_xckernel(self):
        """
        Test TDDFT-ris method with PBE0 functional
        Compare with traditional TDDFT, using RIS integral with xc kernel.
        """
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', GS=True, single=False, conv_tol=1e-4)
        td.with_xc = True 
        td.kernel()  
        energies = td.energies.get()/27.21138602
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]
        self.assertAlmostEqual(abs(energies - e_ab).max(), 0, PLACES)

if __name__ == "__main__":
    print("Full Tests for TDDFT-RIS reducing to traditional TDDFT")
    unittest.main()