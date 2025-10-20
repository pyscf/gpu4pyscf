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
from pyscf import dft
from pyscf.geomopt.geometric_solver import optimize
import gpu4pyscf.tdscf.ris as ris
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
H     1.2953527433   -0.4895463266    0.8457608681
C     0.6689912970   -0.0128659340    0.0499408027
H     1.3504336752    0.5361460613   -0.6478375784
C    -0.6690192526   -0.0870427249   -0.0501820705
H    -1.4008634673    0.6483035475    0.3700152345
H    -1.2449949956   -0.8949946232   -0.5680972562
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
    @unittest.skipIf(num_devices > 1, '')
    def test_opt_rks_tda_1(self):
        mf = dft.RKS(mol, xc='pbe0').to_gpu().density_fit()
        mf.kernel()
        assert mf.converged
        td_ris = ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()
        mol_gpu = optimize(td_ris)

        mff = dft.RKS(mol_gpu, xc='pbe0').to_gpu().density_fit()
        mff.kernel()
        assert mff.converged
        tdf_ris = ris.TDA(mf=mff, nstates=5, spectra=False, single=False, gram_schmidt=True)
        tdf_ris.conv_tol = 1.0E-4
        tdf_ris.Ktrunc = 0.0
        tdf_ris.kernel()
        excited_gradf_ris = tdf_ris.nuc_grad_method()
        excited_gradf_ris.kernel()
        assert np.linalg.norm(excited_gradf_ris.de) < 3.0e-4

    @pytest.mark.slow
    def test_opt_rks_tda_2(self):
        mf = dft.RKS(mol, xc='pbe0').to_gpu().density_fit()
        mf.kernel()
        assert mf.converged
        td_ris = ris.TDA(mf=mf, nstates=5, spectra=False, single=False, gram_schmidt=True)
        td_ris.conv_tol = 1.0E-4
        td_ris.Ktrunc = 0.0
        td_ris.kernel()

        excited_grad = td_ris.nuc_grad_method().as_scanner(state=1)
        mol_gpu = excited_grad.optimizer().kernel()

        mff = dft.RKS(mol_gpu, xc='pbe0').to_gpu().density_fit()
        mff.kernel()
        assert mff.converged
        tdf_ris = ris.TDA(mf=mff, nstates=5, spectra=False, single=False, gram_schmidt=True)
        tdf_ris.conv_tol = 1.0E-4
        tdf_ris.Ktrunc = 0.0
        tdf_ris.kernel()
        excited_gradf_ris = tdf_ris.nuc_grad_method()
        excited_gradf_ris.kernel()
        assert np.linalg.norm(excited_gradf_ris.de) < 3.0e-4


if __name__ == "__main__":
    print("Full Tests for geomtry optimization for excited states using TDDFT-ris.")
    unittest.main()
