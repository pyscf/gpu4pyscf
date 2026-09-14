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

'''Neighboring-k-point overlaps for Wannier-center calculations.'''

import itertools
import numpy as np
import cupy as cp

from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.pbc.df import ft_ao

__all__ = [
    'KPointMesh',
    'periodic_ao_overlap',
    'build_mmn',
    'build_mmn_channels',
]


def _asnumpy(array):
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


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

    The small integer topology arrays are kept on the host. Wavefunctions,
    overlap matrices, and all linear algebra remain on the GPU.
    '''

    def __init__(self, cell, kpts, kmesh=None, tol=1e-7):
        if isinstance(kpts, KPoints):
            raise NotImplementedError(
                'Symmetry-reduced k-point meshes are not supported. '
                'Run the SCF calculation on the full Monkhorst-Pack mesh.')

        kpts = _asnumpy(kpts)
        if kpts.ndim != 2 or kpts.shape[1] != 3:
            raise ValueError(f'kpts must have shape (nk, 3), got {kpts.shape}')

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
            kmesh = np.asarray(kmesh, dtype=int)
            if kmesh.shape != (3,) or np.any(kmesh < 1):
                raise ValueError(f'kmesh must contain three positive integers, got {kmesh}')
            if not np.array_equal(kmesh, inferred_kmesh):
                raise ValueError(
                    f'kmesh {kmesh.tolist()} is inconsistent with the '
                    f'k-points ({inferred_kmesh.tolist()} unique coordinates)')

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
            if np.max(np.min(distance, axis=1)) > tol:
                raise ValueError(f'Failed to map k-points along direction {dim}')

        index_by_address = {}
        for k, address in enumerate(addresses):
            key = tuple(address)
            if key in index_by_address:
                raise ValueError(f'Duplicate k-point mesh address {key}')
            index_by_address[key] = k

        expected_addresses = itertools.product(*(range(n) for n in kmesh))
        missing = [address for address in expected_addresses
                   if address not in index_by_address]
        if missing:
            raise ValueError(f'Incomplete k-point mesh; missing address {missing[0]}')

        self.cell = cell
        self.kpts = kpts
        self.scaled_kpts = scaled_kpts
        self.kmesh = kmesh
        self.addresses = addresses
        self.index_by_address = index_by_address
        self.tol = tol

    def neighbors(self, direction):
        '''Return the +b neighbor index and reciprocal-lattice image shift.'''
        direction = int(direction)
        if direction not in (0, 1, 2):
            raise ValueError(f'direction must be 0, 1, or 2; got {direction}')

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
            error = target - self.scaled_kpts[neighbor] - shift
            if np.max(np.abs(error)) > self.tol:
                raise ValueError(
                    f'Failed to identify the reciprocal image for k-point {k}')
            image_shifts[k] = shift

        return neighbor_indices, image_shifts

    def strings(self, direction):
        '''Return k-point indices grouped into oriented closed strings.'''
        direction = int(direction)
        if direction not in (0, 1, 2):
            raise ValueError(f'direction must be 0, 1, or 2; got {direction}')

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
        return (np.asarray(self.cell.reciprocal_vectors())[direction] /
                self.kmesh[direction])


def periodic_ao_overlap(cell, kpt, neighbor_kpt):
    r'''Compute ``<f_mu,k | f_nu,k'>`` in the reference cell on the GPU.

    ``neighbor_kpt`` may lie outside the first Brillouin zone. This is needed
    for the closing link, where it is ``k_0 + G`` rather than merely ``k_0``.
    '''
    kpt = _asnumpy(kpt).reshape(3)
    neighbor_kpt = _asnumpy(neighbor_kpt).reshape(3)
    raw = ft_ao.ft_aopair(
        cell, (kpt - neighbor_kpt).reshape(1, 3),
        kpti_kptj=np.asarray([neighbor_kpt, kpt]), q=np.zeros(3))[0]

    # ft_aopair follows the Fourier-transform AO-pair convention used by
    # pywannier90.get_M_mat. Its matrix is the Hermitian transpose of
    # <f_mu,k | f_nu,k'> in the row/column convention used below.
    return raw.conj().T


def build_mmn(cell, mo_coeff_kpts, kpts, kmesh, direction, batch_size=None,
              topology=None):
    r'''Build ``M_mn(k,b) = <u_mk | u_n,k+b>`` on the GPU.'''
    return build_mmn_channels(
        cell, (mo_coeff_kpts,), kpts, kmesh, direction, batch_size, topology)[0]


def build_mmn_channels(cell, mo_coeff_channels, kpts, kmesh, direction,
                       batch_size=None, topology=None):
    '''Build neighboring-k-point MO overlaps for one or more spin channels.

    AO Fourier transforms are shared by all channels. Each coefficient array
    must have shape ``(nkpts, nao, nband)``.
    '''
    if cell.dimension != 3:
        raise NotImplementedError(
            'Wannier-center polarization currently supports 3D cells only')
    if topology is None:
        topology = KPointMesh(cell, kpts, kmesh)
    kpts = topology.kpts
    kmesh = topology.kmesh
    nkpts = len(kpts)

    coeff_channels = tuple(cp.asarray(coeff) for coeff in mo_coeff_channels)
    for coeff in coeff_channels:
        if coeff.ndim != 3 or coeff.shape[:2] != (nkpts, cell.nao):
            raise ValueError(
                'Each mo_coeff array must have shape '
                f'({nkpts}, {cell.nao}, nband); got {coeff.shape}')

    neighbor_indices = topology.neighbors(direction)[0]
    neighbor_indices_gpu = cp.asarray(neighbor_indices)
    reciprocal_step = topology.reciprocal_step(direction)
    Gv = (-reciprocal_step).reshape(1, 3)

    ft_opt = ft_ao.FTOpt(cell, kmesh)
    ft_opt.permutation_symmetry = False
    ft_kernel = ft_opt.gen_ft_kernel()

    if batch_size is None:
        batch_size = nkpts
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, got {batch_size}')

    overlaps = tuple(
        cp.empty((nkpts, coeff.shape[2], coeff.shape[2]),
                 dtype=cp.complex128)
        for coeff in coeff_channels)

    for p0 in range(0, nkpts, batch_size):
        p1 = min(p0 + batch_size, nkpts)
        raw = ft_kernel(
            Gv, q=np.zeros(3), kpts=kpts[p0:p1])[:, 0]
        s_ao = raw.conj().transpose(0, 2, 1)
        neighbors = neighbor_indices_gpu[p0:p1]

        for coeff, mmn in zip(coeff_channels, overlaps):
            coeff_left = coeff[p0:p1]
            coeff_right = coeff[neighbors]
            tmp = cp.matmul(s_ao, coeff_right)
            mmn[p0:p1] = cp.matmul(
                coeff_left.conj().transpose(0, 2, 1), tmp)

        del raw, s_ao

    return overlaps
