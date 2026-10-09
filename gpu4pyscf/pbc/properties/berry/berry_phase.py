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
Berry (Zak) phases and Wannier centers from neighboring-k-point overlaps.
Reference:
- tutorial: https://arxiv.org/abs/1202.1831v1
- working equations: PhysRevB.47.1651, PhysRevB.89.155114
- reference codes: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pywannier90.py
'''


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
    '''Return the unitary polar factors of overlap matrices.

    Args:
        overlaps : array_like, shape (..., nband, nband)
            Dimensionless square overlap matrices.
        singular_tol : float
            Minimum allowed singular value.

    Returns:
        cupy.ndarray, same shape as overlaps
            U @ Vh for each SVD overlaps = U @ diag(s) @ Vh. Replacing the
            singular values by one removes the nonunitary part of each link.

    Raises:
        numpy.linalg.LinAlgError
            A singular value is non-finite or smaller than singular_tol.
    '''
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
    '''Compute the many-band Berry (Zak) phase of each closed k-point string.

    Args:
        overlaps : cupy.ndarray, shape (nkpts, nband, nband)
            Dimensionless links M(k,b) = <u_mk | u_n,k+b> for one direction.
            Closing links must include the reciprocal-lattice image shift.
        strings : numpy.ndarray of int, shape (nstrings, nlinks)
            Indices into overlaps, ordered along each closed string.
        singular_tol : float or None
            Optional minimum overlap singular value. None skips the SVD
            check; exactly singular determinants are always rejected.

    Returns:
        cupy.ndarray, shape (nstrings,)
            Principal phases in radians in [-pi, pi). Each phase is
            sum_k arg(det(M(k,b))), wrapped after completing the string.
            No transverse unwrapping or Wilson-loop eigensolve is performed.

    Raises:
        numpy.linalg.LinAlgError
            A determinant is singular or non-finite, or an overlap fails
            the optional singular-value check.
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
    return _wrap_phase(cp.sum(link_phases[strings], axis=1)) # (nstring,) overall phases


def hybrid_wannier_centers(overlaps, strings, singular_tol=1e-10):
    '''Compute hybrid Wannier centers from unitary Wilson-loop eigenphases.

    Args:
        overlaps : cupy.ndarray, shape (nkpts, noccupied, noccupied)
            Dimensionless occupied-band links for one reciprocal direction,
            including the reciprocal image shift on each closing link.
        strings : numpy.ndarray of int, shape (nstrings, nlinks)
            Overlap indices ordered along each closed k-point string.
        singular_tol : float
            Minimum allowed overlap singular value before unitarization.

    Returns:
        centers : cupy.ndarray, shape (nstrings, noccupied)
            Dimensionless fractional centers in [0, 1) along the corresponding
            direct-lattice vector, given by -arg(lambda)/(2*pi) modulo one.
            Values are sorted per string; columns do not track bands between
            strings. These are hybrid centers, localized along one direction.
        phases : cupy.ndarray, shape (nstrings,)
            Principal determinant Berry phases in radians in [-pi, pi).
            These describe the full occupied subspace, not individual bands.

    Notes:
        The center spectrum is invariant under unitary rotations of occupied
        orbitals at each k-point. Non-finite or near-singular overlap singular
        values raise numpy.linalg.LinAlgError.
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
    # * Different strings may have different orders of bands, sort them
    #   to ensure the centers are in the same order. It does not mean
    #   the centers in the same places are for the same band!
    centers.sort(axis=1) # (nstring, noccupied)
    return centers, phases


def diagonal_wannier_centers(overlaps, strings, overlap_tol=1e-12):
    '''Compute band-resolved centers in an externally localized Wannier gauge.

    Args:
        overlaps : cupy.ndarray, shape (nkpts, noccupied, noccupied)
            Dimensionless links already rotated as U(k)^dagger M(k,b) U(k+b)
            by a localized Wannier gauge, including closing-link image shifts.
        strings : numpy.ndarray of int, shape (nstrings, nlinks)
            Overlap indices ordered along each closed k-point string.
        overlap_tol : float
            Minimum allowed magnitude of a diagonal overlap.

    Returns:
        cupy.ndarray, shape (nstrings, noccupied)
            Fractional centers -sum_k arg(M_nn(k,b))/(2*pi) modulo one,
            in [0, 1) along the corresponding direct-lattice vector. The
            supplied band order is preserved; no transverse unwrapping occurs.

    Notes:
        This function does not construct or verify localization of the gauge.
        Without a localized gauge, individual centers are not physical; use
        hybrid_wannier_centers for gauge-invariant center spectra. On a finite
        mesh the diagonal estimator need not equal the Wilson-loop estimator.
        A diagonal magnitude below overlap_tol raises numpy.linalg.LinAlgError.
    '''

    diagonal = cp.diagonal(overlaps, axis1=1, axis2=2)
    if diagonal.size and float(cp.min(cp.abs(diagonal)).get()) < overlap_tol:
        raise np.linalg.LinAlgError(
            'A diagonal overlap vanishes in the supplied Wannier gauge')
    phases = _wrap_phase(cp.sum(cp.angle(diagonal[strings]), axis=1))
    return cp.mod(-phases / TWO_PI, 1.)
