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

'''Berry (Zak) phases and Wannier centers from neighboring-k-point overlaps.'''

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
    return (phase + np.pi) % TWO_PI - np.pi


def _check_singular_values(singular_values, singular_tol):
    if not bool(cp.all(cp.isfinite(singular_values)).get()):
        raise np.linalg.LinAlgError('Non-finite overlap singular values')
    minimum = float(singular_values.min().get())
    if minimum < singular_tol:
        raise np.linalg.LinAlgError(
            'The occupied subspaces at neighboring k-points have a '
            f'near-singular overlap (minimum singular value {minimum:.3e}). '
            'Use a denser k-point mesh or verify that the system is insulating.')


def unitary_part(overlaps, singular_tol=1e-10):
    '''Return the unitary polar factor of a batch of overlap matrices.'''
    overlaps = cp.asarray(overlaps)

    if overlaps.shape[-1] == 0:
        return overlaps.copy()

    u, singular_values, vh = cp.linalg.svd(overlaps)
    _check_singular_values(singular_values, singular_tol)
    # leave the singular values, i.e. signular values are set to 1.
    return cp.matmul(u, vh)


def _unitary_eigenvalues(matrices):
    '''Return Wilson-loop eigenvalues.'''
    # TODO: there may be some gpu version of eigvals
    matrices = cp.asnumpy(matrices)
    eigenvalues = np.stack(
        [np.linalg.eigvals(matrix) for matrix in matrices])
    eigenvalues /= np.abs(eigenvalues)
    return cp.asarray(eigenvalues)


def berry_phase(overlaps, strings, singular_tol=None):
    '''
    Compute the many-band Berry (Zak) phase for each closed k-point string.
    The phase is accumulated as sum(arg(det(M_k))) and wrapped only after
    completing a string. Returns principal phases in [-pi, pi). 
    An optional singular_tol checks near-singular links without computing
    Wilson-loop eigenvectors.
    '''
    if overlaps.shape[1] == 0:
        return cp.zeros(strings.shape[0])
    if singular_tol is not None:
        singular_values = cp.linalg.svd(overlaps, compute_uv=False)
        _check_singular_values(singular_values, singular_tol)

    sign, logabsdet = cp.linalg.slogdet(overlaps)
    if not bool(cp.all(cp.isfinite(logabsdet)).get()):
        raise np.linalg.LinAlgError(
            'A neighboring-k-point overlap matrix is singular')
    link_phases = cp.angle(sign)
    return _wrap_phase(cp.sum(link_phases[strings], axis=1))


def hybrid_wannier_centers(overlaps, strings, singular_tol=1e-10):
    '''Compute hybrid Wannier centers from Wilson-loop eigenphases.

    Returns:
        centers : cupy.ndarray
            Fractional centers in [0, 1) with shape
            (nstring, noccupied).
        phases : cupy.ndarray
            The determinant Berry phase for each string in [-pi, pi).
    '''

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
    '''Compute band-resolved centers in an externally fixed Wannier gauge.

    This formula is intended for overlaps already rotated by a localized
    Wannier gauge U(k). Without such a gauge, individual centers are not
    physical; use hybrid_wannier_centers instead.
    '''

    diagonal = cp.diagonal(overlaps, axis1=1, axis2=2)
    if diagonal.size and float(cp.min(cp.abs(diagonal)).get()) < overlap_tol:
        raise np.linalg.LinAlgError(
            'A diagonal overlap vanishes in the supplied Wannier gauge')
    phases = _wrap_phase(cp.sum(cp.angle(diagonal[strings]), axis=1))
    return cp.mod(-phases / TWO_PI, 1.)
