# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf import lib
from pyscf.pbc.df import df as df_cpu
from pyscf.pbc.x2c import sfx2c1e as sfx2c1e_cpu
from gpu4pyscf.pbc.df.df import GDF
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.pbc.x2c import x2c1e

class KnownValues(unittest.TestCase):
    def test_get_pvdotp(self):
        L = 5.
        n = 11
        np.random.seed(9)
        cell = pyscf.M(
            a = np.eye(3) * L + np.random.rand(3,3) - .5,
            mesh = [n] * 3,
            atom = '''C    3.    2.       3.
                      C    1.    1.       1.''',
            basis = 'ccpvdz',
            precision = 1e-8,
            verbose = 0,
        )

        cell2 = cell.copy()
        cell2.precision = 1e-12
        cell2.build(0, 0)

        kpts4 = cell.make_kpts([4,1,1])
        sorted_cell = SortedGTO.from_cell(cell)

        ref = GDF(cell, kpts4).get_nuc(kpts=kpts4)
        v1 = x2c1e._get_pnucp(sorted_cell, kpts=kpts4, intor='nuc')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1 - ref).max() < 1e-8

        ref = sfx2c1e_cpu.get_pnucp(df_cpu.GDF(cell2))
        v1 = x2c1e._get_pnucp(sorted_cell, intor='pnucp')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1.get() - ref).max() < 1e-8

        ref = sfx2c1e_cpu.get_pnucp(df_cpu.GDF(cell2), kpts4)
        v1 = x2c1e._get_pnucp(sorted_cell, kpts=kpts4, intor='pnucp')
        v1 = sorted_cell.apply_CT_mat_C(v1)
        assert abs(v1.get() - ref).max() < 1e-8


if __name__ == '__main__':
    print("Full Tests for PBC X2C")
    unittest.main()
