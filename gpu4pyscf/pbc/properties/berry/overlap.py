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
import operator
import numpy as np
import cupy as cp
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

__all__ = [
    'KPointMesh',
    'periodic_ao_overlap',
    'build_mmn',
]


def _asnumpy(array):
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def _direction_index(direction):
    direction = operator.index(direction)
    if direction not in (0, 1, 2):
        raise ValueError(f'direction must be 0, 1, or 2; got {direction}')
    return direction


def _commensurate_bvk_mesh(cell, kpts, kmesh=None):
    # FTOpt folds outer BvK images without a twist phase. Every absolute
    # k-point, not just the spacing of the mesh, must be commensurate.
    scaled = np.asarray(cell.get_scaled_kpts(kpts))
    if (kmesh is not None and
            np.max(np.abs(scaled * kmesh - np.rint(scaled * kmesh))) <= 1e-8):
        return kmesh
    try:
        return kpts_to_kmesh(
            cell, kpts, precision=1e-10, bound_by_supmol=False)
    except RuntimeError as err:
        raise ValueError(
            'The GPU Fourier transform requires k-points commensurate '
            'with a finite BvK mesh') from err


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
    '''
    Topology of a full, uniform Monkhorst-Pack mesh.
    '''

    def __init__(self, cell, kpts, kmesh=None, tol=1e-7):
        if isinstance(kpts, KPoints):
            raise NotImplementedError(
                'Symmetry-reduced k-point meshes are not supported. '
                'Run the SCF calculation on the full Monkhorst-Pack mesh.')

        kpts = _asnumpy(kpts)
        if kpts.ndim != 2 or kpts.shape[1] != 3 or len(kpts) == 0:
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
            kmesh = _asnumpy(kmesh)
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
        '''Return the +b neighbor index and reciprocal-lattice image shift.'''
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
        '''Return k-point indices grouped into oriented closed strings.'''
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
        direction = _direction_index(direction)
        return (np.asarray(self.cell.reciprocal_vectors())[direction] /
                self.kmesh[direction])


def periodic_ao_overlap(cell, kpt, neighbor_kpt):
    '''Compute <f_mu,k | f_nu,k'> in the reference cell.
    neighbor_kpt may lie outside the first Brillouin zone. This is needed
    for the closing link, where it is k_0 + G rather than merely k_0.
    '''
    kpt = _asnumpy(kpt).reshape(3)
    neighbor_kpt = _asnumpy(neighbor_kpt).reshape(3)
    bvk_mesh = _commensurate_bvk_mesh(cell, kpt[None])
    ft_opt = ft_ao.FTOpt(cell, bvk_mesh)
    ft_opt.permutation_symmetry = False
    raw = ft_opt.gen_ft_kernel()(
        (kpt - neighbor_kpt).reshape(1, 3),
        q=np.zeros(3), kpts=kpt[None])[0, 0]

    # ft_aopair follows the convention pywannier90.get_M_mat. 
    # Its matrix is the Hermitian transpose of
    # <f_mu,k | f_nu,k'> in the row/column convention used below.
    return raw.conj().T


def build_mmn(cell, mo_coeff_kpts, kpts, kmesh, direction, batch_size=None,
              topology=None):
    '''
    M_mn(k,b) = <u_mk | u_n,k+b>.
    '''
    if cell.dimension != 3:
        raise NotImplementedError(
            'Wannier-center polarization currently supports 3D cells only')
    if topology is None:
        topology = KPointMesh(cell, kpts, kmesh)
    kpts = topology.kpts
    kmesh = topology.kmesh
    nkpts = len(kpts)

    mo_coeff_kpts = cp.asarray(mo_coeff_kpts)
    if (mo_coeff_kpts.ndim != 3 or
            mo_coeff_kpts.shape[:2] != (nkpts, cell.nao)):
        raise ValueError(
            'mo_coeff_kpts must have shape '
            f'({nkpts}, {cell.nao}, nband); got {mo_coeff_kpts.shape}')

    neighbor_indices = topology.neighbors(direction)[0]
    neighbor_indices_gpu = cp.asarray(neighbor_indices)
    reciprocal_step = topology.reciprocal_step(direction)
    Gv = (-reciprocal_step).reshape(1, 3)

    if batch_size is None:
        batch_size = nkpts
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, got {batch_size}')

    bvk_mesh = _commensurate_bvk_mesh(cell, kpts, kmesh)
    ft_opt = ft_ao.FTOpt(cell, bvk_mesh)
    ft_opt.permutation_symmetry = False
    ft_kernel = ft_opt.gen_ft_kernel()

    overlaps = cp.empty(
        (nkpts, mo_coeff_kpts.shape[2], mo_coeff_kpts.shape[2]),
        dtype=cp.complex128)

    for p0 in range(0, nkpts, batch_size):
        p1 = min(p0 + batch_size, nkpts)
        raw = ft_kernel(
            Gv, q=np.zeros(3), kpts=kpts[p0:p1])[:, 0] # Gv (-b) avoids the boundray G problem.
        s_ao = raw.conj().transpose(0, 2, 1)
        neighbors = neighbor_indices_gpu[p0:p1]

        coeff_left = mo_coeff_kpts[p0:p1]
        coeff_right = mo_coeff_kpts[neighbors]
        tmp = cp.matmul(s_ao, coeff_right)
        overlaps[p0:p1] = cp.matmul(
            coeff_left.conj().transpose(0, 2, 1), tmp)

        del raw, s_ao, coeff_left, coeff_right, tmp
    return overlaps
