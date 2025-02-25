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
import numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.ft_ao import ft_aopair, ft_aopair_kpts

def setUpModule():
    global cell
    cell = pgto.M(
        verbose=5, output='/dev/null',
        atom=''' H1   1.3    .2       .3
                 N2   .19   .1      1.1 ''',
        basis={'H1': [[3, [.5, 1.]], [4, [2., 1.]]], 'N2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2]),
        precision=1e-8)

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_ft_aopair_gamma_point(self):
        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair(cell, Gv).get()
        ref = ft_ao.ft_aopair(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair(cell, Gv).get()
        ref = ft_ao.ft_aopair(cell, Gv)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpt(self):
        kpts = cell.get_abs_kpts([6/7, 2/15, 3/8])
        kpti = kptj = kpts

        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        np.random.seed(1)
        kpti = kptj = np.random.random(3)
        dat = ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj)).get()
        ref = ft_ao.ft_aopair(cell, Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpts(self):
        kpts = cell.make_kpts([3,4,3])
        Gv = cell.get_Gv([7,3,3])
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

        Gv = cell.get_Gv([7]*3)[-257:]
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpt_no_aosym(self):
        np.random.seed(1)
        kpti, kptj = kpti_kptj = np.random.random((2,3))
        Gv = cell.get_Gv([3]*3)
        dat = ft_aopair(cell, Gv, kpti_kptj=kpti_kptj).get()
        ref = ft_ao.ft_aopair(cell, Gv, kpti_kptj=kpti_kptj)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

    def test_ft_aopair_kpts_nosym(self):
        np.random.seed(2)
        kpts = np.random.random((4,3))
        Gv = cell.get_Gv([3]*3)
        dat = ft_aopair_kpts(cell, Gv, kptjs=kpts).get()
        ref = ft_ao.ft_aopair_kpts(cell, Gv, kptjs=kpts)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 9)

if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
