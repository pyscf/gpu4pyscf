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

# pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

bas0 = "def2-tzvp"

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
    def test_nac_pbe_tdaris_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 4.38482838e-03, -8.21102914e-14, -1.69475145e-11],
             [-1.88174873e-02,  9.56036995e-13,  8.47200352e-12],
             [-1.88174873e-02, -8.88653417e-13,  8.37251505e-12],])
        ref_de_etf = np.array(
            [[ 9.89286619e-02, -8.19838431e-14, -1.69497938e-11],
             [-4.94643370e-02,  9.57436679e-13,  8.47142257e-12],
             [-4.94643370e-02, -8.90160772e-13,  8.37200753e-12],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    @pytest.mark.slow
    def test_nac_pbe_tdaris_singlet_fdiff(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        nstate = 1
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    @pytest.mark.slow
    def test_nac_pbe0_tdaris_singlet_fdiff(self):
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        nac_ris = td_ris.nac_method()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        nstate = 1
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac_ris, (xI, xI*0.0), e_diag[0])
        delta = 0.0005
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac_ris, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    def test_nac_pbe0_tddftris_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDDFT(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 7.16220831e-04,  1.01353380e-12,  1.38070626e-11],
             [-2.01247331e-02,  5.83533772e-12, -6.67129184e-12],
             [-2.01247348e-02, -6.83851932e-12, -7.11354404e-12],])
        ref_de_etf = np.array(
            [[ 1.06105659e-01,  1.01350398e-12,  1.38098156e-11],
             [-5.30528420e-02,  5.83552982e-12, -6.67197889e-12],
             [-5.30528484e-02, -6.83822778e-12, -7.11429737e-12],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    @unittest.skipIf(num_devices > 1, '')
    def test_nac_camb3lyp_tdaris_singlet_vs_ref(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[-4.93284817e-04,  9.49072069e-14,  1.54699499e-11],
             [ 1.90729037e-02,  5.89916730e-12, -7.71312299e-12],
             [ 1.90729056e-02, -6.01789629e-12, -7.71186244e-12],])
        ref_de_etf = np.array(
            [[-1.01734827e-01,  9.49409210e-14,  1.54692207e-11],
             [ 5.08672907e-02,  5.90047865e-12, -7.71388350e-12],
             [ 5.08672977e-02, -6.01918084e-12, -7.71263288e-12],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe_tdaris_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.ris_zvector_solver = True
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[-0.0150780367, -0.0000000000, 0.0000000000],
             [ 0.0241640836,  0.0000000000, 0.0000000000],
             [ 0.0241640836, -0.0000000000, 0.0000000000],])
        ref_de_etf = np.array(
            [[-0.1096218702,  0.0000000000, 0.0000000000],
             [ 0.0548109333,  0.0000000000, 0.0000000000],
             [ 0.0548109333, -0.0000000000, 0.0000000000],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_pbe0_tdaris_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="pbe0").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.ris_zvector_solver = True
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[ 0.0127227850, -0.0000000000, 0.0000000000],
             [-0.0251899178,  0.0000000000, 0.0000000000],
             [-0.0251899197, -0.0000000000, 0.0000000000],])
        ref_de_etf = np.array(
            [[ 0.1169076233,  0.0000000000, 0.0000000000],
             [-0.0584538196,  0.0000000000, 0.0000000000],
             [-0.0584538263, -0.0000000000, 0.0000000000],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

    def test_nac_camb3lyp_tdaris_singlet_vs_ref_ris_zvector_solver(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        nac_ris = td_ris.nac_method()
        nac_ris.ris_zvector_solver = True
        nac_ris.states=(1,0)
        nac_ris.kernel()

        ref_de = np.array(
            [[-0.0105566781, -0.0000000000, 0.0000000000],
             [ 0.0241046858,  0.0000000000, 0.0000000000],
             [ 0.0241046879, -0.0000000000, 0.0000000000],])
        ref_de_etf = np.array(
            [[-0.1117981850,  0.0000000000, 0.0000000000],
             [ 0.0558990554,  0.0000000000, 0.0000000000],
             [ 0.0558990627, -0.0000000000, 0.0000000000],])

        # compare with previous calculation resusts
        assert np.linalg.norm(np.abs(nac_ris.de) - np.abs(ref_de)) < 1.0E-5
        assert np.linalg.norm(np.abs(nac_ris.de_etf) - np.abs(ref_de_etf)) < 1.0E-5

if __name__ == "__main__":
    print("Full Tests for TD-RKS-ris nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
