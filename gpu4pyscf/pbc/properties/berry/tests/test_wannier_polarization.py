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


def _random_unitary(rng, n):
    matrix = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(matrix)
    return q * np.exp(-1j * np.angle(np.diag(r)))[None, :]


class KnownValues(unittest.TestCase):
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
            'build_mmn_channels')
        with mock.patch(target, return_value=(overlaps,)):
            result = berry.eval_wannier_centers(
                mf, kmesh=kmesh, method='diagonal',
                wannier_gauge=gauge)

        expected = [
            np.mean(np.sum(result.centers[0][direction], axis=1))
            for direction in range(3)
        ]
        np.testing.assert_allclose(
            result.center_sums_fractional[0], expected, atol=1e-14)

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
            centers = result.wannier.centers[0][direction]
            error = np.mod(
                centers - atom_fractional[direction] + .5, 1.) - .5
            np.testing.assert_allclose(error, 0., atol=2e-8)

        np.testing.assert_allclose(
            result.wannier.center_sums_cartesian[0],
            atom_fractional @ lattice, atol=2e-8)
        np.testing.assert_allclose(
            result.electronic + result.ionic, 0., atol=2e-8)

        result_si = berry.eval_polarization(
            mf, kmesh=kmesh, unit='C/m^2', return_details=True)
        np.testing.assert_allclose(
            result_si.ionic,
            result.ionic * nist.E_CHARGE / nist.BOHR_SI**2,
            rtol=1e-12)

        unrestricted_mf = SimpleNamespace(
            cell=cell,
            kpts=kpts,
            mo_coeff=cp.stack((cp.asarray(coeff), cp.asarray(coeff))),
            mo_occ=cp.ones((2, len(kpts), 1)),
            converged=True,
        )
        unrestricted = berry.eval_polarization(
            unrestricted_mf, kmesh=kmesh, return_details=True)
        np.testing.assert_allclose(
            unrestricted.electronic, result.electronic, atol=2e-8)
        np.testing.assert_allclose(
            unrestricted.total, result.total, atol=2e-8)

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
