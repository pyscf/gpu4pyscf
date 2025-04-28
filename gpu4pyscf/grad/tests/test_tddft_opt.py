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
from pyscf import scf, dft, tdscf
from pyscf.geomopt.geometric_solver import optimize
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from packaging import version

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

bas0 = "631g"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_opt_rhf_tda(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=3)
        td.kernel()
        td_cpu = td.to_cpu()
        mol_gpu = optimize(td)
        mol_cpu = optimize(td_cpu)
        print(mol_gpu.atom_coords(), mol_cpu.atom_coords())
        assert np.linalg.norm(mol_gpu.atom_coords() - mol_cpu.atom_coords()) < 1e-4

    def test_opt_rks_tda(self):
        mf = dft.RKS(mol, xc='b3lyp').to_gpu()
        mf.kernel()
        td = mf.TDDFT().set(nstates=3)
        td.kernel()
        td_cpu = td.to_cpu()
        mol_gpu = optimize(td)
        mol_cpu = optimize(td_cpu)
        assert np.linalg.norm(mol_gpu.atom_coords() - mol_cpu.atom_coords()) < 1e-4


if __name__ == "__main__":
    print("Full Tests for geomtry optimization for excited states using TDHF or TDDFT.")
    unittest.main()
