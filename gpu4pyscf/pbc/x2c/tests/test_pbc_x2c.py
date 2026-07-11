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

    def get_hcore(self):
        cell = pyscf.M(
            atom='C 0 0 0; C 1.685 1.685 1.685',
            a='''
            0.00, 3.37, 3.37
            3.37, 0.00, 3.37
            3.37, 3.37, 0.00''',
            basis=('ccpvdz', [[3, [.7, 1]]])
        )
        with lib.temporary_env(lib.param, LIGHT_SPEED=10.):
            cell1 = cell.copy()
            cell1.precision = 1e-11
            cell1.build(0, 0)
            kpts = cell.make_kpts([3,2,1])
            mf = cell1.KRHF(kpts=kpts).x2c1e()
            ref = mf.get_hcore()

            mf = cell.KRHF(kpts=kpts).to_gpu().x2c1e()
            dat = mf.get_hcore()
            assert abs(ref - dat.get()).max() < 1e-8

    @unittest.skip('to_gpu is not available in pyscf 2.13')
    def test_to_cpu(self):
        cell = pyscf.M(
            unit= 'B',
            a = np.eye(3)*4,
            mesh = [11]*3,
            atom = 'H 0 0 0; H 0 0 1.8',
            basis='sto3g')
        with lib.temporary_env(lib.param, LIGHT_SPEED = 2):
            mf = cell.RHF().sfx2c1e()
            ref = mf.kernel()

            mf = mf.to_gpu()
            assert isinstance(mf, x2c1e.SFX2C1E_SCF)
            assert isinstance(mf.with_x2c, x2c1e.SpinFreeX2CHelper)
            mf.run()
            self.assertAlmostEqual(mf.e_tot, ref, 8)

            mf = mf.to_cpu()
            assert isinstance(mf, sfx2c1e_cpu.SFX2C1E_SCF)
            assert isinstance(mf.with_x2c, sfx2c1e_cpu.SpinFreeX2CHelper)

            mf = cell.KRHF(kpts=cell.make_kpts([3,1,1])).sfx2c1e()
            ref = mf.kernel()

            mf = mf.to_gpu().run()
            self.assertAlmostEqual(mf.e_tot, ref, 8)


if __name__ == '__main__':
    print("Full Tests for PBC X2C")
    unittest.main()
