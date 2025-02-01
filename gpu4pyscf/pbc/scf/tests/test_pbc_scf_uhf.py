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
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf as pscf
from gpu4pyscf.pbc.scf import kuhf

def setUpModule():
    global cell
    L = 4
    n = 21
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               verbose = 7,
               output = '/dev/null',
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]]},
               spin = 2)

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_kuhf_bands(self):
        nk = [2, 2, 1]
        kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pscf.KUHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kmf_cpu = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, kmf_cpu.e_tot, 8)
        self.assertAlmostEqual(kmf.e_tot, -4.021029656152094, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = kmf.get_bands(kpts_bands)[0]
        e_ref = kmf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 6)

    def test_uhf_bands(self):
        mf = pscf.UHF(cell).run(conv_tol=1e-9)
        mf_cpu = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_cpu.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -3.9546467710639632, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = mf.get_bands(kpts_bands)[0]
        e_ref = mf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 6)

    def test_small_system(self):
        mol = pbcgto.Cell(
            atom='H 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            spin=1,
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -0.10439957735616917, 8)

        mol = pbcgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)

    def test_density_fit(self):
        from gpu4pyscf.pbc.df.df import GDF
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.spin = 2
        cell.build()

        ref = cell.UHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.11995733902879813, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        ref = cell.UHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.11995733902879813, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

if __name__ == '__main__':
    print("Tests for PBC UHF and PBC KUHF")
    unittest.main()
