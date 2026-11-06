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

"""Full uniform k meshes, shared with GPU Berry postprocessing."""
import itertools
import operator
import numpy as np
from pyscf.pbc.lib.kpts import KPoints

def _direction_index(direction):
    direction = operator.index(direction)
    if direction not in (0, 1, 2):
        raise ValueError(f'direction must be 0, 1, or 2; got {direction}')
    return direction


def _periodic_axis_values(values, tol):
    values = np.mod(values, 1.)
    values[np.abs(values - 1.) < tol] = 0.
    values[np.abs(values) < tol] = 0.
    values = np.sort(values)

    unique = []
    for value in values:
        if not unique or value - unique[-1] > tol:
            unique.append(value)
        else:
            unique[-1] = .5 * (unique[-1] + value)
    return np.asarray(unique)


class KPointMesh:
    '''Topology of a full, uniform Monkhorst-Pack mesh.

    Args:
        cell : pyscf.pbc.gto.Cell
            Periodic cell defining the reciprocal lattice.
        kpts : array_like, shape (nkpts, 3)
            Cartesian k-points in Bohr^-1. Their order is preserved.
            Shifted meshes and points outside the first Brillouin zone
            are allowed; symmetry-reduced KPoints objects are unsupported.
        kmesh : array_like of int, shape (3,), optional
            Number of sampled points along each reciprocal lattice vector.
            Inferred from kpts if omitted.
        tol : float
            Tolerance in fractional reciprocal coordinates for identifying
            mesh points and checking uniform spacing.

    Attributes:
        kpts : numpy.ndarray, shape (nkpts, 3)
            Input Cartesian k-points in Bohr^-1.
        scaled_kpts : numpy.ndarray, shape (nkpts, 3)
            Input k-points in fractional reciprocal coordinates.
        kmesh : numpy.ndarray, shape (3,)
            Sampling mesh dimensions, not the phase-compatible BvK mesh.
        addresses : numpy.ndarray, shape (nkpts, 3)
            Integer mesh addresses, ordered by wrapped fractional coordinates.
        index_by_address : dict
            Map from a mesh-address tuple to the corresponding input index.
    '''

    def __init__(self, cell, kpts, kmesh=None, tol=1e-7):
        if isinstance(kpts, KPoints):
            raise NotImplementedError(
                'Symmetry-reduced k-point meshes are not supported. '
                'Run the SCF calculation on the full Monkhorst-Pack mesh.')

        kpts = kpts.get() if hasattr(kpts, "get") else np.asarray(kpts)
        if kpts.ndim != 2 or kpts.shape[1] != 3 or len(kpts) == 0:
            raise ValueError(f'kpts must have shape (nk, 3), got {kpts.shape}')

        if not np.all(np.isfinite(kpts)):
            raise ValueError("kpts must be finite")
        scaled_kpts = np.asarray(cell.get_scaled_kpts(kpts))
        wrapped_kpts = np.mod(scaled_kpts, 1.)
        wrapped_kpts[np.abs(wrapped_kpts - 1.) < tol] = 0.
        wrapped_kpts[np.abs(wrapped_kpts) < tol] = 0.
        axis_values = tuple(
            _periodic_axis_values(wrapped_kpts[:, dim], tol)
            for dim in range(3))

        inferred_kmesh = np.asarray([len(values) for values in axis_values])
        if kmesh is None:
            kmesh = inferred_kmesh
        else:
            kmesh = kmesh.get() if hasattr(kmesh, "get") else np.asarray(kmesh)
            if not np.array_equal(kmesh, inferred_kmesh):
                raise ValueError(
                    f'kmesh {kmesh.tolist()} is inconsistent with the '
                    f'k-points ({inferred_kmesh.tolist()} unique coordinates)')
            kmesh = inferred_kmesh

        if np.prod(kmesh) != len(kpts):
            raise ValueError(
                'A full Monkhorst-Pack mesh is required: '
                f'prod(kmesh)={np.prod(kmesh)}, nkpts={len(kpts)}')

        for dim, values in enumerate(axis_values):
            if len(values) > 1:
                gaps = np.diff(np.append(values, values[0] + 1.))
                if np.max(np.abs(gaps - 1. / kmesh[dim])) > tol:
                    raise ValueError(
                        f'k-points are not uniformly spaced along direction {dim}')

        addresses = np.empty((len(kpts), 3), dtype=int)
        for dim, values in enumerate(axis_values):
            distance = np.abs(
                np.mod(wrapped_kpts[:, dim, None] - values[None, :] + .5, 1.) - .5)
            addresses[:, dim] = np.argmin(distance, axis=1)

        index_by_address = {
            tuple(address): k for k, address in enumerate(addresses)}
        if len(index_by_address) != len(kpts):
            raise ValueError('A full Monkhorst-Pack mesh is required')

        self.cell = cell
        self.kpts = kpts
        self.scaled_kpts = scaled_kpts
        self.kmesh = kmesh
        self.addresses = addresses
        self.index_by_address = index_by_address

    def neighbors(self, direction):
        '''Return neighbors along reciprocal axis direction (0, 1, or 2).

        Returns:
            neighbor_indices : numpy.ndarray of int, shape (nkpts,)
                Indices into the original kpts array.
            image_shifts : numpy.ndarray of int, shape (nkpts, 3)
                Reciprocal-lattice shifts satisfying
                kpts[neighbor_indices] + image_shifts @ reciprocal_vectors
                = kpts + reciprocal_step(direction). Closing links retain
                the reciprocal shift across the Brillouin-zone boundary.
        '''
        direction = _direction_index(direction)

        neighbor_indices = np.empty(len(self.kpts), dtype=int)
        image_shifts = np.empty((len(self.kpts), 3), dtype=int)
        step = np.zeros(3)
        step[direction] = 1. / self.kmesh[direction]

        for k, address in enumerate(self.addresses):
            neighbor_address = address.copy()
            neighbor_address[direction] += 1
            neighbor_address[direction] %= self.kmesh[direction]
            neighbor = self.index_by_address[tuple(neighbor_address)]
            neighbor_indices[k] = neighbor

            target = self.scaled_kpts[k] + step
            shift = np.rint(target - self.scaled_kpts[neighbor]).astype(int)
            image_shifts[k] = shift

        return neighbor_indices, image_shifts

    def strings(self, direction):
        '''Return oriented closed strings along axis direction (0, 1, or 2).

        Returns:
            numpy.ndarray of int, shape (nstrings, kmesh[direction])
                Input k-point indices ordered along the positive reciprocal
                direction. nstrings = nkpts / kmesh[direction]; rows follow
                the lexicographic order of the two transverse mesh addresses.
        '''
        direction = _direction_index(direction)

        transverse = [dim for dim in range(3) if dim != direction]
        strings = []
        for transverse_address in itertools.product(
                *(range(self.kmesh[dim]) for dim in transverse)):
            address = np.zeros(3, dtype=int)
            address[transverse] = transverse_address
            string = []
            for parallel_address in range(self.kmesh[direction]):
                address[direction] = parallel_address
                string.append(self.index_by_address[tuple(address)])
            strings.append(string)
        return np.asarray(strings, dtype=int)

    def reciprocal_step(self, direction):
        '''Return the positive step along reciprocal axis 0, 1, or 2.

        Returns:
            numpy.ndarray, shape (3,)
                Cartesian reciprocal vector divided by kmesh[direction],
                in Bohr^-1.
        '''
        direction = _direction_index(direction)
        return (np.asarray(self.cell.reciprocal_vectors())[direction] /
                self.kmesh[direction])
