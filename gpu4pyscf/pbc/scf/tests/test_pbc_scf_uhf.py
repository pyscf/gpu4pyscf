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
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf as pscf
from gpu4pyscf.pbc.tools import magmom
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

def setUpModule():
    global cell
    L = 4
    n = 21
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               verbose = 7,
               output = '/dev/null',
               precision = 1e-10,
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
    @staticmethod
    def _magmom_cell():
        return pbcgto.Cell(
            atom='H 3 6 6; H 9 6 6',
            a=np.eye(3) * 12,
            unit='B',
            basis={'H': [[0, (2., 1.)], [1, (2., 1.)]]},
            spin=0,
            precision=1e-12,
            verbose=0,
        ).build()

    def assert_magmom_constraints(self, cell, kpts, dm, magmoms, places=10):
        s = pscf.KUHF(cell, kpts=kpts).get_ovlp()
        nkpts = len(kpts)
        charge = cp.einsum('skij,kji->', dm, s).real.get() / nkpts
        self.assertAlmostEqual(charge, cell.nelectron, places)

        spin_dm = dm[0] - dm[1]
        aoslices = cell.aoslice_by_atom()
        for ia, magmom in magmoms.items():
            p0, p1 = aoslices[ia, 2:]
            local_magmom = cp.einsum(
                'kij,kji->', spin_dm[:, p0:p1], s[:, :, p0:p1]
            ).real.get() / nkpts
            self.assertAlmostEqual(local_magmom, magmom, places)

    def test_init_guess_with_magmom_uniform_and_valence(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        magmoms = {0: 1., 1: -1.}

        dm_uniform = magmom.get_init_guess_with_magmom(
            magcell, kpts, magmoms, method='uniform', key='atom')
        self.assertEqual(dm_uniform.shape, (2, 2, 8, 8))
        self.assert_magmom_constraints(magcell, kpts, dm_uniform, magmoms)
        spin_diag = cp.diagonal(
            dm_uniform[0] - dm_uniform[1], axis1=-2, axis2=-1
        ).real.mean(axis=0)
        cp.testing.assert_allclose(
            spin_diag, cp.asarray([.25] * 4 + [-.25] * 4),
            rtol=0, atol=1e-14)

        dm_valence = magmom.get_init_guess_with_magmom(
            magcell, kpts, magmoms, method='valence', key='atom')
        self.assertEqual(dm_valence.shape, (2, 2, 8, 8))
        self.assert_magmom_constraints(magcell, kpts, dm_valence, magmoms)
        spin_diag = cp.diagonal(
            dm_valence[0] - dm_valence[1], axis1=-2, axis2=-1
        ).real.mean(axis=0)
        cp.testing.assert_allclose(
            spin_diag, cp.asarray([1., 0., 0., 0.,
                                   -1., 0., 0., 0.]),
            rtol=0, atol=1e-14)

    def test_init_guess_with_magmom_spin_sad(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        magmoms = {0: 1., 1: -1.}
        dm = magmom.get_init_guess_with_magmom(
            magcell, kpts, magmoms, method='spin_sad')

        self.assertEqual(dm.shape, (2, 2, 8, 8))
        self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
        cp.testing.assert_allclose(dm[:, 0], dm[:, 1], rtol=0, atol=0)

    def test_init_guess_with_magmom_spin_sad_fractional(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        magmoms = {0: .5, 1: -.5}
        dm = magmom.get_init_guess_with_magmom(
            magcell, kpts, magmoms, method='spin_sad')

        self.assertEqual(dm.shape, (2, 2, 8, 8))
        self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
        cp.testing.assert_allclose(dm[:, 0], dm[:, 1], rtol=0, atol=0)

        parity_cell = pbcgto.Cell(
            atom='He 3 6 6; He 9 6 6',
            a=np.eye(3) * 12,
            unit='B',
            basis={'He': [[0, (2., 1.)], [0, (1., 1.)], [1, (2., 1.)]]},
            spin=0,
            precision=1e-12,
            verbose=0,
        ).build()
        kpts = parity_cell.make_kpts([2, 1, 1])
        magmoms = {0: 1., 1: -1.}
        dm = magmom.get_init_guess_with_magmom(
            parity_cell, kpts, magmoms, method='spin_sad')
        self.assert_magmom_constraints(parity_cell, kpts, dm, magmoms)

    def test_init_guess_with_magmom_spin_sad_pseudo(self):
        magcell = pbcgto.Cell(
            atom='C 5 10 10; C 20 10 10',
            a=np.eye(3) * 30,
            unit='B',
            basis='gth-szv',
            pseudo='gth-pade',
            spin=0,
            precision=1e-12,
            verbose=0,
        ).build()
        kpts = magcell.make_kpts([2, 1, 1])
        magmoms = {0: 2, 1: -2}
        dm = magmom.get_init_guess_with_magmom(
            magcell, kpts, magmoms, method='spin_sad')

        # self.assertEqual(dm.shape, (2, 2, 2, 2))
        self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
        cp.testing.assert_allclose(dm[:, 0], dm[:, 1], rtol=0, atol=0)

    def test_init_guess_with_zero_magmoms(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        dm_ref = pscf.KUHF(magcell, kpts=kpts).get_init_guess(key='atom')

        for magmoms in ({}, {0: 0., 1: 0.}):
            dm = magmom.get_init_guess_with_magmom(
                magcell, kpts, magmoms, method='valence', key='atom')
            cp.testing.assert_allclose(dm, dm_ref, rtol=0, atol=0)
            cp.testing.assert_allclose(dm[0], dm[1], rtol=0, atol=1e-14)

    def test_get_spin_flip_magmom(self):
        magcell = self._magmom_cell()
        nkpts = 3
        nao = magcell.nao_nr()
        dm = cp.arange(2 * nkpts * nao * nao, dtype=cp.float64)
        dm = dm.reshape(2, nkpts, nao, nao)
        dm_original = dm.copy()

        flipped = magmom.get_spin_flip_magmom(magcell, dm, [1])
        expected = dm.copy()
        p0, p1 = magcell.aoslice_by_atom()[1, 2:]
        expected[0, :, p0:p1, p0:p1] = dm[1, :, p0:p1, p0:p1]
        expected[1, :, p0:p1, p0:p1] = dm[0, :, p0:p1, p0:p1]

        cp.testing.assert_array_equal(flipped, expected)
        cp.testing.assert_array_equal(dm, dm_original)

    def test_get_spin_flip_magmom_empty_atom_indices(self):
        magcell = self._magmom_cell()
        dm = cp.zeros((2, 2, magcell.nao_nr(),
                       magcell.nao_nr()))

        flipped = magmom.get_spin_flip_magmom(magcell, dm, [])

        self.assertIsNot(flipped, dm)
        cp.testing.assert_array_equal(flipped, dm)

    def test_kuhf_bands(self):
        nk = [2, 2, 1]
        kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pscf.KUHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kmf_cpu = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, kmf_cpu.e_tot, 8)
        self.assertAlmostEqual(kmf.e_tot, -4.021029656152094, 8)
        pop = kmf.analyze()[0][0]
        self.assertAlmostEqual(lib.fp(pop), 0.02897067698093582, 5)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = kmf.get_bands(kpts_bands)[0]
        e_ref = kmf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 6)

    def test_uhf_bands(self):
        mf = pscf.UHF(cell).run(conv_tol=1e-9)
        mf_cpu = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_cpu.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -3.9546467710639632, 7)
        pop = mf.analyze()[0][0]
        self.assertAlmostEqual(lib.fp(pop), -0.04691820429296646, 5)

        np.random.seed(1)
        kpts_bands = np.random.random((4,3))
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

    def test_rsjk(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.build()

        ref = -0.36989524966775006
        mf = cell.UHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        self.assertAlmostEqual(mf.e_tot, ref, 8)

        mf = cell.KUHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        self.assertAlmostEqual(mf.e_tot, ref, 8)

        mf = cell.KUHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        ref = -0.35369830482164666
        self.assertAlmostEqual(mf.e_tot, ref, 8)

    def test_rsjk_with_df(self):
        ref = cell.UHF(exxdiv='ewald').to_gpu().run()
        mf = cell.UHF(exxdiv='ewald').to_gpu().density_fit()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 6)
        self.assertAlmostEqual(mf.e_tot, -3.954646833686388, 6)

        kmf = cell.KUHF(exxdiv='ewald', kpts=cell.make_kpts([2,1,1])).to_gpu()
        kmf.rsjk = PBCJKMatrixOpt(cell)
        kmf.j_engine = PBCJMatrixOpt(cell)
        kmf.run()
        self.assertAlmostEqual(kmf.e_tot, -3.994410799375493, 6)

    def test_initial_guess_tag(self):
        mf = cell.UHF().to_gpu()
        s = mf.get_ovlp()

        dm = mf.get_init_guess(key='minao', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='hcore', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='atom', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='mod_huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        kmesh = [2, 2, 1]
        kpts = cell.make_kpts(kmesh)
        mf = cell.KUHF(kpts=kpts).to_gpu()
        s = mf.get_ovlp()

        dm = mf.get_init_guess(key='minao', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='hcore', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='atom', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='mod_huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

if __name__ == '__main__':
    print("Tests for PBC UHF and PBC KUHF")
    unittest.main()
