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
from unittest import mock

import cupy as cp
import numpy as np
from pyscf.pbc import gto as pbcgto

from gpu4pyscf.pbc import scf as pscf
from gpu4pyscf.pbc.tools import magmom


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
        self.assertAlmostEqual(charge, cell.tot_electrons(nkpts) / nkpts, places)
        cp.testing.assert_allclose(
            dm, dm.swapaxes(-1, -2).conj(), rtol=0, atol=1e-12)

        spin_dm = dm[0] - dm[1]
        aoslices = cell.aoslice_by_atom()
        for ia in range(cell.natm):
            p0, p1 = aoslices[ia, 2:]
            local_magmom = cp.einsum(
                'kij,kji->', spin_dm[:, p0:p1], s[:, :, p0:p1]
            ).real.get() / nkpts
            self.assertAlmostEqual(local_magmom, magmoms.get(ia, 0.), places)

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

    def test_spin_sad_reuses_equivalent_atomic_density(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        magcell.spin = 4
        magmoms = {0: 1., 1: 1.}

        with mock.patch.object(
                magmom.scf, 'UHF', wraps=magmom.scf.UHF) as atomic_uhf:
            dm = magmom.get_init_guess_with_magmom(
                magcell, kpts, magmoms, method='spin_sad')

        self.assertEqual(atomic_uhf.call_count, 1)
        self.assert_magmom_constraints(magcell, kpts, dm, magmoms)

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

        nao = magcell.nao_nr()
        self.assertEqual(dm.shape, (2, len(kpts), nao, nao))
        self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
        cp.testing.assert_allclose(dm[:, 0], dm[:, 1], rtol=0, atol=0)

    def test_init_guess_with_zero_magmoms(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        for spin in (0, 2):
            magcell.spin = spin
            dm_ref = pscf.KUHF(magcell, kpts=kpts).get_init_guess(key='atom')
            for method in ('spin_sad', 'uniform', 'valence'):
                for magmoms in ({}, {0: 0., 1: 0.}):
                    with self.subTest(spin=spin, method=method, magmoms=magmoms):
                        with mock.patch.object(magmom.logger, 'warn') as warn:
                            dm = magmom.get_init_guess_with_magmom(
                                magcell, kpts, magmoms, method=method, key='atom')
                        warn.assert_not_called()
                        cp.testing.assert_allclose(dm, dm_ref, rtol=0, atol=0)

    def test_nonzero_total_magmom(self):
        magcell = self._magmom_cell()
        cases = [
            (1, 2, {0: 1., 1: 1.}),
            (2, 2, {0: .5, 1: .5}),
            (4, 2, {0: .25, 1: .25}),
            (2, -2, {0: -.5, 1: -.5}),
            (2, 2, {np.int64(0): np.float64(1.)}),
        ]
        for method in ('spin_sad', 'uniform', 'valence'):
            for nkpts, spin, magmoms in cases:
                with self.subTest(method=method, nkpts=nkpts, spin=spin,
                                  magmoms=magmoms):
                    magcell.spin = spin
                    kpts = magcell.make_kpts([nkpts, 1, 1])
                    with mock.patch.object(magmom.logger, 'warn') as warn:
                        dm = magmom.get_init_guess_with_magmom(
                            magcell, kpts, magmoms, method=method, key='atom')
                    warn.assert_not_called()
                    self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
                    s = pscf.KUHF(magcell, kpts=kpts).get_ovlp()
                    nelec = cp.einsum('skij,kji->s', dm, s).real.get()
                    np.testing.assert_allclose(
                        nelec, pscf.KUHF(magcell, kpts=kpts).nelec,
                        atol=1e-10, rtol=0)

    def test_magmom_spin_mismatch_warns_without_changing_input(self):
        magcell = self._magmom_cell()
        cases = [
            (2, 0, {0: .5, 1: .5}),
            (2, 2, {0: 1., 1: 1.}),
            (1, 0, {0: .25, 1: .25}),
        ]
        for method in ('spin_sad', 'uniform', 'valence'):
            for nkpts, spin, magmoms in cases:
                with self.subTest(method=method, nkpts=nkpts, spin=spin):
                    magcell.spin = spin
                    kpts = magcell.make_kpts([nkpts, 1, 1])
                    kpts_original = kpts.copy()
                    magmoms_original = magmoms.copy()
                    with mock.patch.object(magmom.logger, 'warn') as warn:
                        dm = magmom.get_init_guess_with_magmom(
                            magcell, kpts, magmoms, method=method, key='atom')
                    warn.assert_called_once()
                    self.assertIn('does not match cell.spin', warn.call_args[0][1])
                    self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
                    self.assertEqual(magcell.spin, spin)
                    np.testing.assert_array_equal(kpts, kpts_original)
                    self.assertEqual(magmoms, magmoms_original)

    def test_init_guess_with_dm_init(self):
        magcell = self._magmom_cell()
        kpts = magcell.make_kpts([2, 1, 1])
        dm_init = pscf.KUHF(magcell, kpts=kpts).get_init_guess(key='atom')
        dm_original = dm_init.copy()
        for method in ('uniform', 'valence'):
            for magmoms in ({0: 1., 1: -1.}, {0: 0., 1: 0.}, {}):
                with self.subTest(method=method, magmoms=magmoms):
                    with mock.patch.object(pscf.KRHF, 'get_init_guess') as guess:
                        dm = magmom.get_init_guess_with_magmom(
                            magcell, kpts, magmoms, method=method, dm_init=dm_init)
                    guess.assert_not_called()
                    self.assert_magmom_constraints(magcell, kpts, dm, magmoms)
                    cp.testing.assert_allclose(
                        dm.sum(axis=0), dm_init.sum(axis=0), rtol=0, atol=1e-14)
                    cp.testing.assert_array_equal(dm_init, dm_original)

    def test_invalid_magmom_inputs(self):
        magcell = self._magmom_cell()
        cases = [
            ([], TypeError), ({-1: 0.}, IndexError),
            ({magcell.natm: 0.}, IndexError), ({.5: 1.}, TypeError),
            ({True: 1.}, TypeError), ({'0': 1.}, TypeError),
            ({0: '1'}, TypeError), ({0: 1j}, TypeError),
            ({0: [1.]}, TypeError), ({0: True}, TypeError),
            ({0: np.nan}, ValueError), ({0: np.inf}, ValueError),
            ({0: -np.inf}, ValueError),
        ]
        for method in ('spin_sad', 'uniform', 'valence'):
            for magmoms, error in cases:
                with self.subTest(method=method, magmoms=magmoms):
                    with self.assertRaises(error):
                        magmom.get_init_guess_with_magmom(
                            magcell, np.zeros((1, 3)), magmoms, method=method)

    def test_spin_sad_moment_bounds(self):
        magcell = self._magmom_cell()
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            magmom.get_init_guess_with_magmom(
                magcell, np.zeros((1, 3)), {0: 2., 1: -2.})
        for method in ('spin_sad', 'uniform', 'valence'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'Total magnetic moment'):
                    magmom.get_init_guess_with_magmom(
                        magcell, np.zeros((1, 3)), {0: 2., 1: 2.}, method=method)

        magcell = pbcgto.Cell(
            atom='He 0 0 0; He 6 0 0', a=np.eye(3) * 12, unit='B',
            basis='sto-3g', spin=0, verbose=0).build()
        with self.assertRaisesRegex(ValueError, 'cannot be represented'):
            magmom.get_init_guess_with_magmom(
                magcell, np.zeros((1, 3)), {0: 1., 1: -1.})
        magcell.spin = 2
        with self.assertRaisesRegex(ValueError, 'orbital capacity'):
            magmom.get_init_guess_with_magmom(
                magcell, np.zeros((1, 3)), {0: 1., 1: 1.})

    def test_charged_cell(self):
        magcell = self._magmom_cell()
        magcell.charge = 1
        magcell.spin = 1
        kpts = magcell.make_kpts([2, 1, 1])
        magmoms = {0: .5}
        with self.assertRaisesRegex(ValueError, 'neutral cell'):
            magmom.get_init_guess_with_magmom(magcell, kpts, magmoms)
        for method in ('uniform', 'valence'):
            with self.subTest(method=method):
                dm = magmom.get_init_guess_with_magmom(
                    magcell, kpts, magmoms, method=method, key='atom')
                self.assert_magmom_constraints(magcell, kpts, dm, magmoms)

        magcell.charge = 0
        magcell.spin = 0
        magcell.nelectron = 1
        with self.assertRaisesRegex(ValueError, 'neutral cell'):
            magmom.get_init_guess_with_magmom(
                magcell, kpts, {0: .5, 1: -.5})
        magcell.nelectron = 2
        dm = magmom.get_init_guess_with_magmom(
            magcell, kpts, {0: .5, 1: -.5})
        self.assert_magmom_constraints(magcell, kpts, dm, {0: .5, 1: -.5})

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


if __name__ == '__main__':
    print('Tests for atom-resolved magnetic moment tools')
    unittest.main()
