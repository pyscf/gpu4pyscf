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
from pyscf import lib
from pyscf.pbc.scf import hf as pbchf_cpu
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
                                    [0, (1.2, 1.0)]]})
        cls.cell = cell

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()

    def test_rhf_exx_ewald(self):
        cell = self.cell
        mf = scf.RHF(cell, exxdiv='ewald').run()
        self.assertAlmostEqual(mf.e_tot, -4.3511582284698633, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.double)
        kmf = scf.KRHF(cell, [[0,0,0]], exxdiv='ewald').run()
        self.assertAlmostEqual(mf.e_tot, kmf.e_tot, 8)

        # test bands
        np.random.seed(1)
        kpts_band = np.random.random((2,3))
        e1, c1 = mf.get_bands(kpts_band)
        e0, c0 = kmf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e0-e1).get().max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1[0].get()), -6.2986775452228283, 6)
        self.assertAlmostEqual(lib.fp(e1[1].get()), -7.6616273746782362, 6)

    def test_rhf_exx_ewald_with_kpt(self):
        np.random.seed(1)
        k = np.random.random(3)
        cell = self.cell
        mf = scf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.complex128)

        kmf = scf.KRHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertAlmostEqual(e0, e1, 7)

        # test bands
        np.random.seed(1)
        kpt_band = np.random.random(3)
        e1, c1 = mf.get_bands(kpt_band)
        e0, c0 = kmf.get_bands(kpt_band)
        self.assertAlmostEqual(abs(e0-e1).get().max(), 0, 7)
        self.assertAlmostEqual(lib.fp(e1.get()), -6.8312867098806249, 6)

    def test_rhf_exx_None(self):
        cell = self.cell
        mf = scf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.double)

        kmf = scf.KRHF(cell, [[0,0,0]], exxdiv=None)
        e0 = kmf.kernel()
        self.assertAlmostEqual(e0, e1, 7)

        np.random.seed(1)
        k = np.random.random(3)
        mf = scf.RHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 7)
        self.assertTrue(mf.mo_coeff.dtype == np.complex128)

        kmf = scf.KRHF(cell, k[None,:], exxdiv=None)
        kmf.init_guess = 'hcore'
        e0 = kmf.kernel()
        self.assertAlmostEqual(e0, e1, 7)

    def test_jk(self):
        cell = self.cell
        nao = cell.nao
        np.random.seed(2)
        dm = np.random.random((2,nao,nao)) + .5j*np.random.random((2,nao,nao))
        dm = dm + dm.conj().transpose(0,2,1)
        ref = pbchf_cpu.RHF(cell).get_jk(cell, dm)

        dm = cp.asarray(dm)
        vj, vk = scf.RHF(cell).get_jk(cell, dm)
        self.assertAlmostEqual(abs(vj.get() - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vk.get() - ref[1]).max(), 0, 9)

    def test_krhf_bands(self):
        nk = [2, 2, 1]
        cell = self.cell
        kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kmf_cpu = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, kmf_cpu.e_tot, 8)
        self.assertAlmostEqual(kmf.e_tot, -4.1828127052055395, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = kmf.get_bands(kpts_bands)[0]
        e_ref = kmf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 7)

    def test_density_fit(self):
        from gpu4pyscf.pbc.df.df import GDF
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.build()

        ref = cell.RHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.3740002917376214, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        ref = cell.KRHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.3740002917376214, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
