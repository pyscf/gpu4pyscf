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
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf
import pytest
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696
"""

bas0 = "def2tzvp"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_df_nac_tda_singlet_svwn(self):
        mf = dft.RKS(mol, xc='svwn').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

    def test_df_nac_grad_tda_singlet_svwn(self):
        mf = dft.RKS(mol, xc='svwn').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9

    def test_df_nac_tda_singlet_b3lyp(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

    def test_df_nac_grad_tda_singlet_b3lyp(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9

    def test_df_nac_tda_singlet_camb3lyp(self):
        mf = dft.RKS(mol, xc='camb3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()

        # ! NOTE: Low accuracy arising from batched Z-vector solver.
        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-8
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-8

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()

        # ! NOTE: Low accuracy arising from divided energies after z-vector solver
        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-8
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-8

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()

        # ! NOTE: Low accuracy arising from divided energies after z-vector solver
        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-8
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-8

    def test_df_nac_grad_tda_singlet_camb3lyp(self):
        mf = dft.RKS(mol, xc='camb3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        # ! NOTE: Low accuracy arising from batched Z-vector solver.
        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-8
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-8
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-7

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9

    def test_df_nac_tda_singlet_tpss(self):
        mf = dft.RKS(mol, xc='tpss').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

    def test_df_nac_grad_tda_singlet_tpss(self):
        mf = dft.RKS(mol, xc='tpss').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9

    def test_df_nac_tddft_singlet_b3lyp(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(1,2,3)
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

    def test_df_nac_grad_tddft_singlet_b3lyp(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        # ! NOTE: Low accuracy arising from batched Z-vector solver.

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9

    def test_df_nac_tda_singlet_b3lyp_rissolver(self):
        mf = dft.RKS(mol, xc='svwn').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0, 1, 2, 3)
        nac_test.ris_zvector_solver = True
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(0, 1)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(0, 1)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(0, 1)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(0, 1)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 2)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 2)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(1, 3)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(1, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

        nac1 = td.nac_method()
        nac1.states=(2, 3)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(2, 3)]['de']) - np.abs(nac1.de)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-9
        assert abs(np.abs(nac_test.results[(2, 3)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-9

    def test_df_nac_grad_tda_singlet_b3lyp_rissolver(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        mf.kernel()
        td = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td.conv_tol = 1.0E-4
        td.Ktrunc = 0.0
        td.kernel()

        nac_test = td.nac_gradient_method()
        nac_test.states=(0,1,2,3)
        nac_test.grad_state = 1
        nac_test.ris_zvector_solver = True
        nac_test.kernel()

        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.ris_zvector_solver = True
        nac1.kernel()

        assert abs(np.abs(nac_test.results[(1,2)]['de']) - np.abs(nac1.de)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_scaled']) - np.abs(nac1.de_scaled)).max() < 1e-6
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf']) - np.abs(nac1.de_etf)).max() < 1e-7
        assert abs(np.abs(nac_test.results[(1,2)]['de_etf_scaled']) - np.abs(nac1.de_etf_scaled)).max() < 1e-6

        g1 = td.nuc_grad_method()
        g1.state=1
        g1.ris_zvector_solver = True
        g1.kernel()

        assert abs(np.abs(nac_test.grad_result) - np.abs(g1.de)).max() < 1e-9
    

if __name__ == "__main__":
    print("Full Tests for density-fitting batched TD-RKS-ris nonadiabatic coupling vectors between excited states.")
    unittest.main()
