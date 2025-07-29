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
import cupy as cp
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
import gpu4pyscf

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
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
    def test_nac_pbe_tdaris_singlet_vs_tda_ge(self):
        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,0)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()
        
        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 3.0E-3
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 3.0E-3
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2*np.linalg.norm(g.de - g_ris.de)

    def test_nac_pbe0_tddftris_singlet_vs_tddft_ge(self):
        mf = dft.rks.RKS(mol, xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,0)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()

        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 4.0E-3
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 4.0E-3
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2*np.linalg.norm(g.de - g_ris.de)

    def test_nac_camb3lyp_tdaris_singlet_vs_tda_ge(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").density_fit().to_gpu()
        # mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,0)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()

        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 3.0E-2
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 3.0E-3
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2*np.linalg.norm(g.de - g_ris.de)

    def test_nac_pbe_tda_singlet_vs_tda_ee(self):
        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,2)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()

        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 2.0E-2
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2.0E-2
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2 * np.linalg.norm(g.de - g_ris.de)

    def test_nac_pbe_tda_singlet_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[nstate])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4.0E-3
        
        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5
        print(fdiff_nac)
        print(delta, np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)))

    def test_nac_pbe0_tda_singlet_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = dft.rks.RKS(mol, xc="pbe0").density_fit().to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[nstate])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4.0E-3

        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5

    def test_nac_pbe0_tddft_singlet_vs_tddft_ee(self):
        mf = dft.rks.RKS(mol, xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,2)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()

        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 1.0E-2
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 1.0E-2
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2 * np.linalg.norm(g.de - g_ris.de)

    def test_nac_camb3lyp_tddft_singlet_vs_tddft_ee(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac_obj = td.nac_method()
        nac_obj.states=(1,2)
        nac_obj.kernel()
        g = td.nuc_grad_method()
        g.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()
        g_ris = td_ris.nuc_grad_method()
        g_ris.kernel()

        # compare with traditional TDDFT
        assert np.linalg.norm(np.abs(nac_obj.de) - np.abs(nac_ris.de)) < 1.0E-2
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 1.0E-2
        # check the difference between RIS and TDDFT for nacv is the same with gradient
        assert np.linalg.norm(np.abs(nac_obj.de_etf) - np.abs(nac_ris.de_etf)) < 2 * np.linalg.norm(g.de - g_ris.de)


if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RKS-ris nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
