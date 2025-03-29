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

import unittest
import tempfile
import numpy as np
import cupy
import pyscf
from pyscf import lib
from gpu4pyscf.scf import hf_lowmem

mol = pyscf.M(
    atom = '''
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    ''',
    basis='ccpvdz',
    spin=None,
    output = '/dev/null'
)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf_scf(self):
        e_tot = hf_lowmem.RHF(mol).kernel()
        e_ref = -76.02676567311744
        assert np.abs(e_tot - e_ref) < 1e-8

    def test_diis_on_cpu(self):
        mf = hf_lowmem.RHF(mol)
        mf.diis = mf.DIIS()
        mf.diis.incore = False
        e_tot = mf.kernel()
        e_ref = -76.02676567311744
        assert np.abs(e_tot - e_ref) < 1e-8
        assert all(isinstance(x, np.ndarray) and not isinstance(x, cupy.ndarray)
                   for x in mf.diis._buffer.values())

if __name__ == "__main__":
    print("Full Tests for hf_lowmem")
    unittest.main()
