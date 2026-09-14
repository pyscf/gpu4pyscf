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

'''Berry phases and Wannier centers from neighboring-k-point overlaps.'''

import numpy as np
import cupy as cp

__all__ = [
    'berry_phase',
    'hybrid_wannier_centers',
    'diagonal_wannier_centers',
    'unitary_part',
]

TWO_PI = 2. * np.pi


def _wrap_phase(phase):
    return cp.angle(cp.exp(1j * phase))


def unitary_part(overlaps, singular_tol=1e-10):
    '''Return the unitary polar factor of a batch of overlap matrices.'''
    overlaps = cp.asarray(overlaps)
    if overlaps.ndim < 2 or overlaps.shape[-1] != overlaps.shape[-2]:
        raise ValueError(
            f'overlaps must end in square matrix dimensions, got {overlaps.shape}')
    if overlaps.shape[-1] == 0:
        return overlaps.copy()

    u, singular_values, vh = cp.linalg.svd(overlaps)
    minimum = float(singular_values.min().get())
    if minimum < singular_tol:
        raise np.linalg.LinAlgError(
            'The occupied subspaces at neighboring k-points have a '
            f'near-singular overlap (minimum singular value {minimum:.3e}). '
            'Use a denser k-point mesh or verify that the system is insulating.')
    return cp.matmul(u, vh)


def _unitary_eigenvalues(matrices, diagonal_tol=1e-9):
    '''Diagonalize unitary matrices through commuting Hermitian parts.'''
    matrices_h = matrices.conj().transpose(0, 2, 1)
    real_part = .5 * (matrices + matrices_h)
    imag_part = (matrices - matrices_h) / (2j)
    identity = cp.eye(matrices.shape[-1])

    for coefficient in (np.sqrt(2.), np.sqrt(3.), np.pi):
        _, vectors = cp.linalg.eigh(real_part + coefficient * imag_part)
        rotated = cp.matmul(
            vectors.conj().transpose(0, 2, 1),
            cp.matmul(matrices, vectors))
        eigenvalues = cp.diagonal(rotated, axis1=1, axis2=2)
        off_diagonal = rotated - eigenvalues[:, :, None] * identity
        if float(cp.max(cp.abs(off_diagonal)).get()) < diagonal_tol:
            return eigenvalues / cp.abs(eigenvalues)

    raise np.linalg.LinAlgError(
        'Failed to resolve the eigenphases of the unitary Wilson loop')


def berry_phase(overlaps, strings):
    r'''Compute the many-band Berry phase for each closed k-point string.

    The phase is accumulated as ``sum(arg(det(M_k)))`` and wrapped only after
    completing a string. ``slogdet`` avoids determinant overflow and underflow.
    '''
    overlaps = cp.asarray(overlaps)
    strings = cp.asarray(strings, dtype=cp.int64)
    if overlaps.ndim != 3 or overlaps.shape[1] != overlaps.shape[2]:
        raise ValueError(
            f'overlaps must have shape (nkpts, nband, nband), got {overlaps.shape}')
    if strings.ndim != 2:
        raise ValueError(f'strings must have shape (nstring, nlink), got {strings.shape}')
    if overlaps.shape[1] == 0:
        return cp.zeros(strings.shape[0])

    sign, logabsdet = cp.linalg.slogdet(overlaps)
    if not bool(cp.all(cp.isfinite(logabsdet)).get()):
        raise np.linalg.LinAlgError(
            'A neighboring-k-point overlap matrix is singular')
    link_phases = cp.angle(sign)
    return _wrap_phase(cp.sum(link_phases[strings], axis=1))


def hybrid_wannier_centers(overlaps, strings, singular_tol=1e-10):
    r'''Compute hybrid Wannier centers from Wilson-loop eigenphases.

    Returns:
        centers : cupy.ndarray
            Fractional centers in ``[0, 1)`` with shape
            ``(nstring, noccupied)``.
        phases : cupy.ndarray
            The determinant Berry phase for each string in ``[-pi, pi)``.
    '''
    overlaps = cp.asarray(overlaps)
    strings = cp.asarray(strings, dtype=cp.int64)
    if overlaps.ndim != 3 or overlaps.shape[1] != overlaps.shape[2]:
        raise ValueError(
            f'overlaps must have shape (nkpts, nband, nband), got {overlaps.shape}')
    if strings.ndim != 2:
        raise ValueError(f'strings must have shape (nstring, nlink), got {strings.shape}')

    nstrings = strings.shape[0]
    nband = overlaps.shape[1]
    phases = berry_phase(overlaps, strings)
    if nband == 0:
        return cp.empty((nstrings, 0)), phases

    unitary_links = unitary_part(overlaps, singular_tol)
    loops = cp.broadcast_to(cp.eye(nband, dtype=cp.complex128),
                            (nstrings, nband, nband)).copy()
    for link in range(strings.shape[1]):
        loops = cp.matmul(loops, unitary_links[strings[:, link]])

    eigenvalues = _unitary_eigenvalues(loops)
    eigenphases = cp.angle(eigenvalues)
    centers = cp.mod(-eigenphases / TWO_PI, 1.)
    centers.sort(axis=1)
    return centers, phases


def diagonal_wannier_centers(overlaps, strings, overlap_tol=1e-12):
    r'''Compute band-resolved centers in an externally fixed Wannier gauge.

    This formula is intended for overlaps already rotated by a localized
    Wannier gauge ``U(k)``. Without such a gauge, individual centers are not
    physical; use :func:`hybrid_wannier_centers` instead.
    '''
    overlaps = cp.asarray(overlaps)
    strings = cp.asarray(strings, dtype=cp.int64)
    if overlaps.ndim != 3 or overlaps.shape[1] != overlaps.shape[2]:
        raise ValueError(
            f'overlaps must have shape (nkpts, nband, nband), got {overlaps.shape}')
    if strings.ndim != 2:
        raise ValueError(f'strings must have shape (nstring, nlink), got {strings.shape}')

    diagonal = cp.diagonal(overlaps, axis1=1, axis2=2)
    if diagonal.size and float(cp.min(cp.abs(diagonal)).get()) < overlap_tol:
        raise np.linalg.LinAlgError(
            'A diagonal overlap vanishes in the supplied Wannier gauge')
    phases = _wrap_phase(cp.sum(cp.angle(diagonal[strings]), axis=1))
    return cp.mod(-phases / TWO_PI, 1.)
