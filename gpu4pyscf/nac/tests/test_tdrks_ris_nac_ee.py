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
import pytest
from gpu4pyscf.lib.multi_gpu import num_devices

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
    @unittest.skipIf(num_devices > 1, '')
    def test_nac_pbe_tda_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()

        ref_de = np.array(
            [[-7.46173726e-16,  9.35902790e-02, -2.89341627e-14],
             [-5.56902476e-17, -5.37437170e-02,  3.50026779e-02],
             [ 7.19306347e-16, -5.37437170e-02, -3.50026779e-02],])
        ref_de_etf = np.array(
            [[-6.19856849e-16,  9.26041619e-02, -2.85174872e-14],
             [-2.11973474e-16, -4.63020605e-02,  3.65102194e-02],
             [ 8.29062637e-16, -4.63020605e-02, -3.65102194e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_pbe_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, Ktrunc = 0.0, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 3.0E-3

        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_pbe0_tda_singlet_fdiff(self):
        """
        compare with finite difference
        """
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, Ktrunc = 0.0, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()
        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstateI = 0
        nstateJ = 1
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 4.0E-4

        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks_ris.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta=0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac_ris, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta, with_ris=True)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1.0E-5

    def test_nac_df_pbe0_tddft_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu().density_fit()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()

        ref_de = np.array(
            [[-1.51924941e-16, -1.01018969e-01, -1.39858998e-09],
             [ 1.60794931e-16,  5.75872716e-02, -3.81043482e-02],
             [ 2.01916854e-16,  5.75872738e-02,  3.81043496e-02],])
        ref_de_etf = np.array(
            [[-1.81973724e-16, -1.00688428e-01, -1.46279761e-09],
             [ 6.25879103e-17,  5.03441954e-02, -3.95286263e-02],
             [ 2.51975848e-16,  5.03441978e-02,  3.95286277e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_camb3lyp_tddft_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 6.09174478e-16, -9.04369401e-02, -2.38678612e-09],
             [-1.25420843e-15,  5.26052262e-02, -3.38655808e-02],
             [ 5.16773594e-16,  5.26052298e-02,  3.38655832e-02]])
        ref_de_etf = np.array(
            [[ 5.48922915e-16, -9.00876449e-02, -2.47854031e-09],
             [-1.15755555e-15,  4.50438148e-02, -3.53895964e-02],
             [ 4.84556048e-16,  4.50438185e-02,  3.53895989e-02],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe_tda_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.ris_zvector_solver = True
        nac_ris.kernel()

        ref_de = np.array(
            [[ 0.0000000000, -0.0941080765, -0.0000000000],
             [-0.0000000000,  0.0540026157, -0.0355220643],
             [ 0.0000000000,  0.0540026157,  0.0355220643],])
        ref_de_etf = np.array(
            [[ 0.0000000000, -0.0931219594, -0.0000000000],
             [-0.0000000000,  0.0465609593, -0.0370296058],
             [ 0.0000000000,  0.0465609593,  0.0370296058],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe0_tda_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.ris_zvector_solver = True
        nac_ris.kernel()

        ref_de = np.array(
            [[-0.0000000000, -0.1017318256, -0.0000000014],
             [ 0.0000000000,  0.0579454885, -0.0387517422],
             [ 0.0000000000,  0.0579454908,  0.0387517436],])
        ref_de_etf = np.array(
            [[-0.0000000000, -0.1013903631, -0.0000000015],
             [ 0.0000000000,  0.0506951632, -0.0401688078],
             [-0.0000000000,  0.0506951657,  0.0401688093],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_camb3lyp_tda_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,2)
        nac_ris.ris_zvector_solver = True
        nac_ris.kernel()

        ref_de = np.array(
            [[ 0.0000000000, -0.0909942185, -0.0000000010],
             [-0.0000000000,  0.0528867701, -0.0344111229],
             [ 0.0000000000,  0.0528867724,  0.0344111239],])
        ref_de_etf = np.array(
            [[ 0.0000000000, -0.0906362599, -0.0000000011],
             [-0.0000000000,  0.0453181227, -0.0359291095],
             [ 0.0000000000,  0.0453181251,  0.0359291106],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5


if __name__ == "__main__":
    print("Full Tests for TD-RKS-ris nonadiabatic coupling vectors between excited states")
    unittest.main()
