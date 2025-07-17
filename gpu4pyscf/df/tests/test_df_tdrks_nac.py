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

# pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

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
    def test_grad_pbe_tddft_singlet_df_ge(self):

        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()

        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_grad_b3lyp_tddft_singlet_df_ge(self):

        mf = dft.rks.RKS(mol, xc="b3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()

        mf = dft.rks.RKS(mol, xc="b3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_grad_camb3lyp_tda_singlet_df_ge(self):

        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,0)
        nac1.kernel()

        mf = dft.rks.RKS(mol, xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_grad_pbe_tda_singlet_df_ee(self):
        mf = dft.rks.RKS(mol, xc="pbe").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        
        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 5e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 5e-4

    def test_grad_b3lyp_tddft_singlet_df_ee(self):
        mf = dft.rks.RKS(mol, xc="b3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        
        mf = dft.rks.RKS(mol, xc="b3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 4e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 4e-4

    def test_grad_camb3lyp_tddft_singlet_df_ee(self):
        mf = dft.rks.RKS(mol, xc="camb3lyp").to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td = mf.TDDFT().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()
        
        mf = dft.rks.RKS(mol, xc="camb3lyp").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.kernel()
        td2 = mf.TDDFT().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 4e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 4e-4


if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RKS nonadiabatic coupling vectors between ground and excited state.")
    unittest.main()
