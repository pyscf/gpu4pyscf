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

'''
Neighboring-k-point overlaps for Wannier-center calculations.
Reference:
- tutorial: https://arxiv.org/abs/1202.1831v1
- working equations: PhysRevB.47.1651, PhysRevB.89.155114
- reference codes: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pywannier90.py
'''

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


def periodic_ao_overlap(cell, kpt, neighbor_kpt):
    '''Compute the overlap of cell-periodic Bloch AO factors.

    Args:
        cell : pyscf.pbc.gto.Cell
            Periodic cell defining the AO basis.
        kpt, neighbor_kpt : array_like, shape (3,)
            Cartesian k-points in Bohr^-1. For a closing link,
            neighbor_kpt must include the reciprocal-lattice image shift,
            e.g. k_0 + G, rather than only the wrapped k_0.

    Returns:
        cupy.ndarray, shape (nao, nao)
            Dimensionless overlap <f_mu,k | f_nu,k'> integrated over the
            reference cell, with f_mu,k(r) = exp(-i k.r) phi_mu,k(r).
            Rows belong to kpt and columns to neighbor_kpt.
    '''
    kpt = _asnumpy(kpt).reshape(3)
    neighbor_kpt = _asnumpy(neighbor_kpt).reshape(3)
    # Gv already contains the full momentum transfer, so q is zero.
    return ft_ao.ft_aopair(
        cell, (neighbor_kpt - kpt).reshape(1, 3),
        kpti_kptj=np.asarray([kpt, neighbor_kpt]), q=np.zeros(3))[0]


def build_mmn(cell, mo_coeff_kpts, kpts, kmesh=None, topology=None):
    '''Build M_mn(k,b) = <u_mk | u_n,k+b> for all three mesh directions.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional periodic cell defining the AO basis.
        mo_coeff_kpts : array_like, shape (nkpts, nao, nband)
            Bloch orbital coefficients in the same k-point order as kpts.
            Supply occupied orbitals for Berry phases and polarization.
        kpts : array_like, shape (nkpts, 3)
            Full uniform mesh in Cartesian coordinates, in Bohr^-1.
            Shifted meshes must be compatible with a finite BvK supercell.
        kmesh : array_like of int, shape (3,), optional
            Sampling mesh dimensions. Inferred from kpts if omitted.
        topology : KPointMesh, optional
            Precomputed topology for the same cell and kpts. When supplied,
            its kpts and kmesh are used.

    Returns:
        cupy.ndarray, shape (3, nkpts, nband, nband)
            Dimensionless overlaps for b_d = reciprocal_vectors[d]/kmesh[d].
            The first axis selects d = 0, 1, 2. Rows are bands at k and
            columns are bands at its +b_d neighbor, including the reciprocal
            image shift on closing links. All directions share one Fourier
            transform and are held in GPU memory.
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

    neighbor_indices = cp.asarray(np.asarray([
        topology.neighbors(direction)[0] for direction in range(3)]))
    reciprocal_steps = np.asarray([
        topology.reciprocal_step(direction) for direction in range(3)])

    bvk_mesh = _commensurate_bvk_mesh(cell, kpts, kmesh)
    ft_opt = ft_ao.FTOpt(cell, bvk_mesh)
    ft_opt.permutation_symmetry = False
    ft_kernel = ft_opt.gen_ft_kernel()

    # With k_j = k and Gv = -b, raw[k,d] is <f_mu,k+b | f_nu,k>.
    # Its adjoint gives the forward link while sharing kpts for all directions.
    raw = ft_kernel(-reciprocal_steps, q=np.zeros(3), kpts=kpts)
    s_ao = raw.conj().transpose(1, 0, 3, 2)
    return cp.matmul(
        mo_coeff_kpts.conj().transpose(0, 2, 1)[None],
        cp.matmul(s_ao, mo_coeff_kpts[neighbor_indices]))
