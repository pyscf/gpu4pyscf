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
from pyscf import lib
from gpu4pyscf import tdscf, nac
from gpu4pyscf.lib.multi_gpu import num_devices
import pytest

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
    def test_nac_pbe_tddft_singlet_ge_vs_direct(self):
        mf = mol.RKS(xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()

        mf = mol.RKS(xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_nac_b3lyp_tddft_singlet_ge_vs_direct(self):
        mf = mol.RKS(xc="b3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()
        assert abs(lib.fp(abs(nac1.de)) - 0.0052457301652304705) < 1e-6
        assert abs(lib.fp(abs(nac1.de_scaled)) - 0.018760185858556) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf)) - 0.11309082066049063) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf_scaled)) - 0.4044441379597835) < 1e-6

        mf = mol.RKS(xc="b3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(lib.fp(abs(nac2.de)) - 0.005253791783724717) < 1e-6
        assert abs(lib.fp(abs(nac2.de_scaled)) - 0.01878923218797409) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf)) - 0.11307570002834381) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf_scaled)) - 0.4043947057878392) < 1e-6
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_nac_camb3lyp_tda_singlet_ge_vs_direct(self):
        mf = mol.RKS(xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()
        assert abs(lib.fp(abs(nac1.de)) - 0.014480595744614034) < 1e-6
        assert abs(lib.fp(abs(nac1.de_scaled)) - 0.05107272540277223) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf)) - 0.11045272132213822) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf_scaled)) - 0.3895641868306882) < 1e-6

        mf = mol.RKS(xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(lib.fp(abs(nac2.de)) - 0.014489172435210002) < 1e-6
        assert abs(lib.fp(abs(nac2.de_scaled)) - 0.051103316892427836) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf)) - 0.11044433704363377) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf_scaled)) - 0.3895372203038537) < 1e-6
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_nac_pbe_tda_singlet_ee_vs_direct(self):
        mf = mol.RKS(xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        mf = mol.RKS(xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 5e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 5e-4

    @pytest.mark.slow
    def test_nac_pbe_tda_singlet_df_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = mol.RKS(xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = td.nac_method()
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    @pytest.mark.slow
    def test_nac_pbe0_tda_singlet_df_fdiff(self):
        """
        Compare the analytical nacv with finite difference nacv
        """
        mf = mol.RKS(xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        nac1 = td.nac_method()
        a, b = td.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)

        # ground-excited state
        nstate = 0
        xI = xy_diag[:, nstate]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ge(nac1, (xI, xI*0.0), e_diag[0])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ge(nac1, (xI, xI*0.0), delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

        # excited-excited state
        nstateI = 1
        nstateJ = 2
        xI = xy_diag[:, nstateI]*np.sqrt(0.5)
        xJ = xy_diag[:, nstateJ]*np.sqrt(0.5)
        ana_nac = nac.tdrks.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), e_diag[nstateI], e_diag[nstateJ])
        delta = 0.001
        fdiff_nac = nac.finite_diff.get_nacv_ee(nac1, (xI, xI*0.0), (xJ, xJ*0.0), nstateJ, delta=delta)
        assert np.linalg.norm(np.abs(ana_nac[1]) - np.abs(fdiff_nac)) < 1e-5

    def test_nac_b3lyp_tddft_singlet_ee_vs_direct(self):
        mf = mol.RKS(xc="b3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(lib.fp(abs(nac1.de)) - 0.07404315170306648) < 1e-6
        assert abs(lib.fp(abs(nac1.de_scaled)) - 1.0809728445817666) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf)) - 0.0762404494085614) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf_scaled)) - 1.113051694501974) < 1e-6

        mf = mol.RKS(xc="b3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(lib.fp(abs(nac2.de)) - 0.07405373220610881) < 1e-6
        assert abs(lib.fp(abs(nac2.de_scaled)) - 1.0812083497421323) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf)) - 0.07625116411566571) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf_scaled)) - 1.1132915636170257) < 1e-6
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 4e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 4e-4

    def test_nac_camb3lyp_tddft_singlet_ee_vs_direct(self):
        mf = mol.RKS(xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        assert abs(lib.fp(abs(nac1.de)) - 0.07127181853584236) < 1e-6
        assert abs(lib.fp(abs(nac1.de_scaled)) - 1.0088658350077309) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf)) - 0.07358370491069106) < 1e-6
        assert abs(lib.fp(abs(nac1.de_etf_scaled)) - 1.0415910162353121) < 1e-6

        mf = mol.RKS(xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(lib.fp(abs(nac2.de)) - 0.07128268447159698) < 1e-6
        assert abs(lib.fp(abs(nac2.de_scaled)) - 1.0091137105524828) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf)) - 0.0735946925708874) < 1e-6
        assert abs(lib.fp(abs(nac2.de_etf_scaled)) - 1.041843665789115) < 1e-6
        # Compare with direct TDDFT NACV
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 4e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 4e-4


if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RKS nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
