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

import pyscf
import numpy as np
import unittest
import pytest
from pyscf import scf, dft, tdscf
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
import gpu4pyscf.tdscf.ris as ris

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):

    def test_grad_pbe_tda(self):
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = dft.RKS(mol, xc='pbe').to_gpu()
        mf.kernel()

        td = ris.TDA(mf=mf, nstates=5, spectra=False, single=False)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.with_xc = True
        td.without_ris_approx = True
        td.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ori = mf.TDA()
        td_ori.nstates = 5
        td_ori.kernel()
        g_ori = td_ori.nuc_grad_method()
        g_ori.kernel()
        
        assert np.linalg.norm(g_ori.de - g.de) < 1.0E-5

    def test_grad_b3lyp_tda(self):
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()

        td = ris.TDA(mf=mf, nstates=5, spectra=False, single=False)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.with_xc = True
        td.without_ris_approx = True
        td.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ori = mf.TDA()
        td_ori.nstates = 5
        td_ori.kernel()
        g_ori = td_ori.nuc_grad_method()
        g_ori.kernel()
        
        assert np.linalg.norm(g_ori.de - g.de) < 1.0E-5

    def test_grad_pbe_tddft(self):
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = dft.RKS(mol, xc='pbe').to_gpu()
        mf.kernel()

        td = ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.with_xc = True
        td.without_ris_approx = True
        td.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ori = mf.TDDFT()
        td_ori.nstates = 5
        td_ori.kernel()
        g_ori = td_ori.nuc_grad_method()
        g_ori.kernel()
        
        assert np.linalg.norm(g_ori.de - g.de) < 1.0E-5

    def test_grad_b3lyp_tddft(self):
        mol = pyscf.M(atom=atom, basis='ccpvdz')
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()

        td = ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.with_xc = True
        td.without_ris_approx = True
        td.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ori = mf.TDDFT()
        td_ori.nstates = 5
        td_ori.kernel()
        g_ori = td_ori.nuc_grad_method()
        g_ori.kernel()
        
        assert np.linalg.norm(g_ori.de - g.de) < 1.0E-5


if __name__ == "__main__":
    print("Full Tests for TD-RKS RIS Gradient with more paremeter options.")
    unittest.main()
