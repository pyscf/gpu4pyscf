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
    def test_grad_tdhf_singlet_df_ge(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td1 = mf.TDHF().set(nstates=5)
        td1.kernel()
        nac1 = td1.nac_method()
        nac1.states=(1,0)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDHF().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_grad_tda_singlet_df_ge(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td1 = mf.TDA().set(nstates=5)
        td1.kernel()
        nac1 = td1.nac_method()
        nac1.states=(2,0)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(2,0)
        nac2.kernel()
        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 1e-4

    def test_grad_tda_singlet_df_ee(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()

        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 3e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 3e-4

        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,3)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDA().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,3)
        nac2.kernel()

        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 3e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 3e-4

    def test_grad_tdhf_singlet_df_ee(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDHF().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDHF().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,2)
        nac2.kernel()

        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 3e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 3e-4

        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDHF().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,3)
        nac1.kernel()

        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        td2 = mf.TDHF().set(nstates=5)
        td2.kernel()
        nac2 = td2.nac_method()
        nac2.states=(1,3)
        nac2.kernel()

        assert getattr(nac2.base._scf, 'with_df', None) is not None
        assert abs(np.abs(nac1.de) - np.abs(nac2.de)).max() < 1e-4
        assert abs(np.abs(nac1.de_scaled) - np.abs(nac2.de_scaled)).max() < 3e-4
        assert abs(np.abs(nac1.de_etf) - np.abs(nac2.de_etf)).max() < 1e-4
        assert abs(np.abs(nac1.de_etf_scaled) - np.abs(nac2.de_etf_scaled)).max() < 3e-4


if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RHF nonadiabatic coupling vectors between ground and excited states.")
    unittest.main()
