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
from gpu4pyscf import tdscf
import gpu4pyscf

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "ccpvdz"

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
    def test_tda_pbe_singlet(self):
        mf = dft.rks.RKS(mol, xc="pbe").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1.0E-12
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        td_ris.conv_tol = 1.0E-4
        td_ris.kernel()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)
        ref = np.array([0.25933899, 0.33439342, 0.35638257, 0.42592451, 0.51762646])
        assert np.linalg.norm(e_diag-td_ris.energies.get()/27.21138602) < 1.0E-7
        assert np.linalg.norm(e_diag-ref) < 1.0E-7

    def test_tdaris_pbe0_singlet(self):
        mf = dft.rks.RKS(mol, xc="pbe0").density_fit().to_gpu()
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1.0E-12
        mf.kernel()

        td_ris = tdscf.ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True, Ktrunc=0.0)
        td_ris.conv_tol = 1.0E-4
        td_ris.kernel()

        a, b = td_ris.get_ab()
        e_diag, xy_diag = diagonalize_tda(a)
        ref = np.array([0.28174892, 0.35852982, 0.38054425, 0.45227567, 0.5288743])
        assert np.linalg.norm(e_diag-td_ris.energies.get()/27.21138602) < 1.0E-7
        assert np.linalg.norm(e_diag-ref) < 1.0E-7
    

if __name__ == "__main__":
    print("Full Tests for density-fitting TD-RKS-ris.")
    unittest.main()
