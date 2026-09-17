# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import cupy as cp

from pyscf.data import nist
from pyscf.pbc import gto
from pyscf.pbc.df import ft_ao as ft_ao_cpu
from gpu4pyscf.pbc.properties import berry
from gpu4pyscf.pbc.properties.berry import polarization as polarization_lib
from gpu4pyscf.pbc.properties.berry.overlap import _commensurate_bvk_mesh


def _random_unitary(rng, n):
    matrix = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(matrix)
    return q * np.exp(-1j * np.angle(np.diag(r)))[None, :]


class KnownValues(unittest.TestCase):
    @staticmethod
    def _mock_mf(kmesh=(1, 4, 1)):
        cell = gto.Cell(
            atom='He 0 0 0', a=np.diag([5., 6., 7.]), unit='Bohr',
            basis={'He': [[0, (1.2, 1.)], [0, (.5, 1.)]]}, verbose=0)
        cell.build()
        kpts = cell.make_kpts(kmesh)
        return SimpleNamespace(
            cell=cell, kpts=kpts,
            mo_coeff=cp.ones((len(kpts), cell.nao, 1)),
            mo_occ=cp.full((len(kpts), 1), 2.), converged=True)

    def test_kpoint_mesh_boundary_images(self):
        cell = gto.Cell(
            atom='He 0 0 0',
            a=np.asarray([[4.0, .2, .1], [.1, 4.3, .2], [.2, .1, 4.5]]),
            basis={'He': [[0, (1.2, 1.)]]},
            verbose=0,
        )
        cell.build()
        kmesh = np.asarray([3, 2, 2])
        kpts = cell.make_kpts(
            kmesh, wrap_around=True, scaled_center=[.17, -.11, .07])
        topology = berry.KPointMesh(cell, kpts, kmesh)

        for direction in range(3):
            neighbors, shifts = topology.neighbors(direction)
            neighbor_images = cell.get_abs_kpts(
                topology.scaled_kpts[neighbors] + shifts)
            expected = kpts + topology.reciprocal_step(direction)
            np.testing.assert_allclose(neighbor_images, expected, atol=1e-12)

            strings = topology.strings(direction)
            self.assertEqual(
                strings.shape,
                (np.prod(np.delete(kmesh, direction)), kmesh[direction]))
            np.testing.assert_array_equal(
                neighbors[strings], np.roll(strings, -1, axis=1))

    def test_wilson_centers_are_gauge_invariant(self):
        rng = np.random.default_rng(4)
        nlinks = 5
        expected_centers = np.asarray([.17, .63])
        link = np.diag(np.exp(-2j * np.pi * expected_centers / nlinks))
        overlaps = np.repeat(link[None], nlinks, axis=0)
        strings = np.arange(nlinks).reshape(1, nlinks)

        gauges = np.asarray([
            _random_unitary(rng, len(expected_centers))
            for _ in range(nlinks)])
        transformed = np.empty_like(overlaps)
        for k in range(nlinks):
            transformed[k] = (
                gauges[k].conj().T @ overlaps[k] @ gauges[(k + 1) % nlinks])

        centers, phases = berry.hybrid_wannier_centers(
            cp.asarray(transformed), strings)
        diagonal_centers = berry.diagonal_wannier_centers(
            cp.asarray(overlaps), strings)
        expected_phase = np.angle(
            np.exp(-2j * np.pi * expected_centers.sum()))

        np.testing.assert_allclose(
            cp.asnumpy(centers[0]), expected_centers, atol=1e-12)
        np.testing.assert_allclose(
            cp.asnumpy(diagonal_centers[0]), expected_centers, atol=1e-12)
        self.assertAlmostEqual(float(phases[0].get()), expected_phase, 12)

    def test_wilson_centers_with_nonunitary_links_and_degeneracy(self):
        rng = np.random.default_rng(51)
        angles = np.asarray([.2, .8, -.7, -.7, 2.1])
        expected = np.sort(np.mod(-angles / (2 * np.pi), 1.))
        link = np.diag(np.linspace(.6, .95, 5) * np.exp(1j * angles / 3))
        gauges = [_random_unitary(rng, 5) for _ in range(3)]
        overlaps = np.asarray([
            gauges[k].conj().T @ link @ gauges[(k + 1) % 3]
            for k in range(3)])
        centers, phases = berry.hybrid_wannier_centers(overlaps, [[0, 1, 2]])
        np.testing.assert_allclose(cp.asnumpy(centers[0]), expected, atol=1e-10)
        expected_phase = np.angle(np.prod(np.exp(1j * angles)))
        self.assertAlmostEqual(float(phases[0].get()), expected_phase, 10)

    def test_gpu_mmn_matches_pyscf_ft_aopair(self):
        cell = gto.Cell(
            atom='H .4 .7 1.1; H 1.7 1.4 .8',
            a=np.asarray([[4.0, .2, .1], [.1, 4.3, .2], [.2, .1, 4.5]]),
            basis={'H': [[0, (1.1, 1.)]]},
            precision=1e-10,
            verbose=0,
        )
        cell.build()
        kmesh = np.asarray([2, 1, 2])
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        topology = berry.KPointMesh(cell, kpts, kmesh)

        rng = np.random.default_rng(9)
        coeff = (
            rng.standard_normal((len(kpts), cell.nao, cell.nao)) +
            1j * rng.standard_normal((len(kpts), cell.nao, cell.nao)))

        for direction in range(3):
            value = berry.build_mmn(
                cell, cp.asarray(coeff), kpts, kmesh, direction,
                topology=topology)
            neighbors, shifts = topology.neighbors(direction)
            reference = np.empty_like(cp.asnumpy(value))
            for k, neighbor in enumerate(neighbors):
                neighbor_image = cell.get_abs_kpts(
                    topology.scaled_kpts[neighbor] + shifts[k])
                raw = ft_ao_cpu.ft_aopair(
                    cell, (kpts[k] - neighbor_image).reshape(1, 3),
                    kpti_kptj=np.asarray([neighbor_image, kpts[k]]),
                    q=np.zeros(3))[0]
                s_ao = raw.conj().T
                reference[k] = (
                    coeff[k].conj().T @ s_ao @ coeff[neighbor])
            np.testing.assert_allclose(
                cp.asnumpy(value), reference, atol=2e-9, rtol=2e-9)

    def test_diagonal_center_sum_uses_localized_band_centers(self):
        cell = gto.Cell(
            atom='He 0 0 0',
            a=np.diag([5., 6., 7.]),
            unit='Bohr',
            basis={'He': [[0, (1.2, 1.)], [0, (.5, 1.)]]},
            verbose=0,
        )
        cell.build()
        kmesh = np.asarray([2, 1, 1])
        kpts = cell.make_kpts(kmesh)
        coeff = cp.broadcast_to(cp.eye(2), (len(kpts), 2, 2))
        gauge = cp.broadcast_to(cp.eye(2), (len(kpts), 2, 2))
        mf = SimpleNamespace(
            cell=cell,
            kpts=kpts,
            mo_coeff=coeff,
            mo_occ=cp.full((len(kpts), 2), 2.),
            converged=True,
        )
        overlaps = cp.asarray([
            [[np.exp(-.2j), .35], [.1j, np.exp(-.5j)]],
            [[np.exp(-.25j), -.2j], [.15, np.exp(-.45j)]],
        ])

        target = (
            'gpu4pyscf.pbc.properties.berry.polarization.'
            'build_mmn')
        with mock.patch(target, return_value=overlaps):
            result = berry.eval_wannier_centers(
                mf, kmesh=kmesh, method='diagonal',
                wannier_gauge=gauge)

        expected = [
            np.mean(np.sum(result.centers[direction], axis=1))
            for direction in range(3)
        ]
        np.testing.assert_allclose(
            result.center_sums_fractional, expected, atol=1e-14)

    def test_atomic_limit_centers_and_total_polarization(self):
        lattice = np.diag([7., 8., 9.])
        atom_fractional = np.asarray([.23, .31, .17])
        cell = gto.Cell(
            atom=[['He', atom_fractional @ lattice]],
            a=lattice,
            unit='Bohr',
            basis={'He': [[0, (1.2, 1.)]]},
            precision=1e-10,
            verbose=0,
        )
        cell.build()
        kmesh = np.asarray([2, 2, 2])
        kpts = cell.make_kpts(kmesh)

        overlap = np.asarray(
            cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
        coeff = (1. / np.sqrt(overlap[:, 0, 0])).reshape(-1, 1, 1)
        mf = SimpleNamespace(
            cell=cell,
            kpts=kpts,
            mo_coeff=cp.asarray(coeff),
            mo_occ=cp.full((len(kpts), 1), 2.),
            converged=True,
        )

        result = berry.eval_polarization(
            mf, kmesh=kmesh, return_details=True)
        for direction in range(3):
            centers = result.wannier.centers[direction]
            error = np.mod(
                centers - atom_fractional[direction] + .5, 1.) - .5
            np.testing.assert_allclose(error, 0., atol=2e-8)

        np.testing.assert_allclose(
            result.wannier.center_sums_cartesian,
            atom_fractional @ lattice, atol=2e-8)
        np.testing.assert_allclose(
            result.electronic + result.ionic, 0., atol=2e-8)

        result_si = berry.eval_polarization(
            mf, kmesh=kmesh, unit='C/m^2', return_details=True)
        np.testing.assert_allclose(
            result_si.ionic,
            result.ionic * nist.E_CHARGE / nist.BOHR_SI**2,
            rtol=1e-12)

    def test_shifted_gpu_mmn_matches_cpu(self):
        cell = gto.Cell(
            atom='H .4 .7 1.1; H 1.7 1.4 .8', unit='Bohr',
            a=np.asarray([[4., .2, .1], [.1, 4.3, .2], [.2, .1, 4.5]]),
            basis={'H': [[0, (.45, 1.)]]}, precision=1e-10, verbose=0)
        cell.build()
        kmesh = np.asarray([2, 1, 2])
        kpts = cell.make_kpts(kmesh, wrap_around=True, scaled_center=[.25, 0, 0])
        kpts = kpts[[2, 0, 3, 1]]
        topology = berry.KPointMesh(cell, kpts)
        np.testing.assert_array_equal(_commensurate_bvk_mesh(cell, kpts, kmesh), [4, 1, 2])
        rng = np.random.default_rng(31)
        coeff = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
        for direction in range(3):
            neighbors, shifts = topology.neighbors(direction)
            reference = []
            for k, neighbor in enumerate(neighbors):
                image = cell.get_abs_kpts(topology.scaled_kpts[neighbor] + shifts[k])
                raw = ft_ao_cpu.ft_aopair(
                    cell, (kpts[k] - image)[None],
                    kpti_kptj=[image, kpts[k]], q=np.zeros(3))[0]
                # Independent scalar-index contraction from pywannier90.
                ref = np.einsum('nu,vm,uv->nm',
                                coeff[neighbor].T.conj(), coeff[k], raw).conj().T
                reference.append(ref)
                if k == 0:
                    np.testing.assert_allclose(
                        cp.asnumpy(berry.periodic_ao_overlap(cell, kpts[k], image)),
                        raw.conj().T, atol=2e-9, rtol=2e-9)
            for batch in (1, 3, None):
                value = berry.build_mmn(cell, coeff, kpts, kmesh, direction, batch)
                np.testing.assert_allclose(cp.asnumpy(value), reference, atol=2e-9, rtol=2e-9)

    def test_invalid_mesh_and_bvk_inputs(self):
        mf = self._mock_mf()
        for kpts in (np.empty((0, 3)), mf.kpts[[0, 0, 2, 3]]):
            with self.subTest(kpts=kpts), self.assertRaises(ValueError):
                berry.KPointMesh(mf.cell, kpts)
        with self.assertRaises(ValueError):
            berry.KPointMesh(mf.cell, mf.kpts, [1, 4.5, 1])
        topology = berry.KPointMesh(mf.cell, mf.kpts)
        for method in (topology.neighbors, topology.strings, topology.reciprocal_step):
            with self.assertRaises(TypeError):
                method(.5)
        shifted = mf.cell.get_abs_kpts([[.25 + 4e-7, 0, 0]])
        with self.assertRaises(ValueError):
            _commensurate_bvk_mesh(mf.cell, shifted)

    def test_invalid_occupations_and_gauge(self):
        mf = self._mock_mf()
        for occupation in (1.99999, np.nan, -1., np.inf, 1., .5):
            mf.mo_occ = cp.full((4, 1), occupation)
            with self.subTest(occupation=occupation), self.assertRaises(ValueError):
                polarization_lib._occupied_coefficients(mf)
        mf.mo_occ = cp.full((5, 1), 2.)
        with self.assertRaises(ValueError):
            polarization_lib._occupied_coefficients(mf)
        mf.mo_occ = cp.full((4, 1), 2. - 1e-8)
        coefficients = polarization_lib._occupied_coefficients(mf)
        self.assertEqual(coefficients.shape, (4, 2, 1))
        mf.converged = np.bool_(False)
        with self.assertRaises(RuntimeError):
            polarization_lib._occupied_coefficients(mf)
        gauge = cp.full((4, 1, 1), np.nan)
        with self.assertRaises(ValueError):
            polarization_lib._apply_wannier_gauge(coefficients, gauge)
        with self.assertRaises(ValueError):
            berry.unwrap_polarization([np.nan, 0, 0], np.zeros(3), mf.cell)

    def test_invalid_overlap_inputs(self):
        overlaps = cp.ones((2, 1, 1), dtype=cp.complex128)
        functions = (berry.berry_phase, berry.hybrid_wannier_centers,
                     berry.diagonal_wannier_centers)
        for function in functions:
            for strings in ([[0., 1.]], [[0, 2]], [[-1, 0]], [[]], [0, 1]):
                with self.subTest(function=function, strings=strings), \
                        self.assertRaises(ValueError):
                    function(overlaps, strings)
            with self.assertRaises(ValueError):
                function(cp.full((2, 1, 1), np.nan), [[0, 1]])
            for tol in (0, -1, np.nan, np.inf):
                with self.subTest(function=function, tol=tol), self.assertRaises(ValueError):
                    function(overlaps, [[0, 1]], tol)
        for tol in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                berry.unitary_part(overlaps, tol)
        for function in functions:
            with self.assertRaises(np.linalg.LinAlgError):
                function(cp.zeros_like(overlaps), [[0, 1]])
        with self.assertRaises(np.linalg.LinAlgError):
            berry.berry_phase(overlaps * 1e-12, [[0, 1]], singular_tol=1e-10)
        empty, phase = berry.hybrid_wannier_centers(cp.empty((2, 0, 0)), [[0, 1]])
        self.assertEqual(empty.shape, (1, 0))
        np.testing.assert_array_equal(cp.asnumpy(phase), [0.])

    def test_principal_phase_contract_and_phase_only_path(self):
        mf = self._mock_mf()
        kmesh = [1, 4, 1]
        principal = np.pi * np.asarray([.9, -.9, -.8, .8])
        overlaps = cp.asarray(np.exp(1j * principal)[:, None, None])
        target = 'gpu4pyscf.pbc.properties.berry.polarization.build_mmn'
        with mock.patch(target, return_value=overlaps):
            with mock.patch.object(np.linalg, 'eigvals',
                                   side_effect=AssertionError('no Wilson solve')):
                with mock.patch.object(cp.linalg, 'slogdet', wraps=cp.linalg.slogdet) as slogdet:
                    phases = berry.eval_berry_phase(mf, kmesh)
                    self.assertEqual(slogdet.call_count, 3)
            with mock.patch.object(cp.linalg, 'slogdet', wraps=cp.linalg.slogdet) as slogdet:
                result = berry.eval_wannier_centers(mf, kmesh)
                self.assertEqual(slogdet.call_count, 3)
        for direction in range(3):
            np.testing.assert_allclose(
                phases[direction], result.berry_phases[direction])
        np.testing.assert_allclose(result.berry_phases[0], principal)
        expected_sum = -.5 * np.mean([.9, 1.1, 1.2, .8])
        self.assertAlmostEqual(result.center_sums_fractional[0], expected_sum)
        phase = berry.berry_phase(cp.asarray([[[-1. + 0j]]]), [[0]])
        self.assertAlmostEqual(float(phase[0].get()), -np.pi)

    def test_transverse_winding_and_ambiguous_branches(self):
        for phases, kmesh in (([0, .5, -1, -.5], [1, 4, 1]),
                              ([0, .75, -.75, .9], [1, 2, 2]),
                              ([0, -1], [1, 2, 1])):
            phase = cp.asarray(phases) * np.pi
            with self.subTest(phases=phases):
                with self.assertRaisesRegex(ValueError, 'continuous periodic'):
                    polarization_lib._unwrap_transverse_phases(phase, kmesh, 0)
                centers = cp.mod(-phase[:, None] / (2 * np.pi), 1.)
                with self.assertRaisesRegex(ValueError, 'continuous periodic'):
                    polarization_lib._sum_diagonal_centers(centers, kmesh, 0)
        mf = self._mock_mf()
        phase = cp.asarray([0, .5, -1, -.5]) * np.pi
        target = 'gpu4pyscf.pbc.properties.berry.polarization.build_mmn'
        with mock.patch(target, return_value=cp.exp(1j * phase)[:, None, None]):
            berry.eval_berry_phase(mf)
            with self.assertRaisesRegex(ValueError, 'continuous periodic'):
                berry.eval_wannier_centers(mf)

    def test_unrestricted_mean_field_is_rejected(self):
        mf = self._mock_mf()
        mf.mo_coeff = cp.stack((mf.mo_coeff, mf.mo_coeff))
        mf.mo_occ = cp.stack((cp.ones((4, 1)), cp.zeros((4, 1))))
        with self.assertRaisesRegex(NotImplementedError, 'Unrestricted'):
            berry.eval_wannier_centers(mf)

    def test_polarization_branch_for_triclinic_cell(self):
        cell = gto.Cell(
            atom='He 0 0 0',
            a=np.asarray([[4.0, .7, .2], [.1, 4.3, .6], [.3, .2, 4.5]]),
            basis={'He': [[0, (1.2, 1.)]]},
            verbose=0,
        )
        cell.build()
        quantum = berry.polarization_quantum(cell)
        reference = np.asarray([.012, -.023, .007])
        residual = np.asarray([2e-4, -1e-4, 3e-4])
        branch = np.asarray([2, -3, 1])
        polarization = reference + residual + branch @ quantum

        matched, selected_branch = berry.unwrap_polarization(
            polarization, reference, cell, return_branch=True)
        np.testing.assert_allclose(matched, reference + residual, atol=1e-14)
        np.testing.assert_array_equal(selected_branch, branch)
        np.testing.assert_allclose(
            berry.polarization_difference(
                polarization, reference, cell),
            residual, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
