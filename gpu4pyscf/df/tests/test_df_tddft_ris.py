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
from gpu4pyscf.grad.tests.test_tdrhf_grad import diagonalize_tda
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.lib import logger

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


class KnownValues(unittest.TestCase):
    def test_get_Tpq_limited_memory(self):
        auxmol = pyscf.df.addons.make_auxmol(mol, 'def2-universal-jkfit')
        ris = tdscf.ris
        lower_inv = ris.get_eri2c_inv_lower(auxmol).astype(cp.float64)
        C_p = cp.eye(mol.nao)[:,:2]
        C_q = cp.eye(mol.nao)[:,2:5]
        log = logger.new_logger(mol)
        ref = ris.get_Tpq(mol, auxmol, lower_inv, C_p, C_q,
                          in_ram=False, single=False, log=log)
        nao = SortedMole.from_mol(mol, decontract=True).nao
        # Reserve forty auxiliary functions for the inner orbital contraction.
        memory = 8*nao**2*160
        for in_ram in (False, True):
            with self.subTest(in_ram=in_ram):
                with lib.temporary_env(ris, get_avail_mem=(lambda **kw: memory)):
                    result = ris.get_Tpq(mol, auxmol, lower_inv, C_p, C_q,
                                         in_ram=in_ram, single=False, log=log)
                for actual, expected in zip(result, ref):
                    np.testing.assert_allclose(cp.asnumpy(actual), expected.get(),
                                               atol=1e-10, rtol=1e-10)

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
