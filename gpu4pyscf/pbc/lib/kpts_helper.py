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

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.tools import k2gamma

def conj_images_in_bvk_cell(kmesh, return_pair=False):
    '''Map BvK images from L to -L modulo the supercell lattice.

    kmesh gives the number of primitive cells along each direction in the
    periodic BvK supercell (returned by kpts_to_kmesh). It is not MP k-point
    mesh.

    Return an integer array of length prod(kmesh).
    If return_pair is True, return each image pair once, including self-pairs.
    '''
    # Periodic image inversion happens to be identical to the conj_kpts for ther
    # larger MP mesh with gamma point.
    return conj_kpts(kmesh, with_gamma_point=True, return_pair=return_pair)

def conj_kpts(mp_mesh, with_gamma_point=True, return_pair=False):
    '''Map each MP k-point to its -k partner

    mp_mesh gives the sampling mesh, not the periodic BvK mesh.
    Points must follow cell.make_kpts(mp_mesh, with_gamma_point=with_gamma_point)
    ordering, with no additional shift or symmetry reduction. Wrapping points
    into the first Brillouin zone does not change the map.

    Return an integer array of length prod(mp_mesh). If return_pair is True,
    return an (npairs, 2) array with each pair once, including self-pairs.
    '''
    indices = [np.arange(n) for n in mp_mesh]
    if with_gamma_point:
        partners = [(-idx) % n for idx, n in zip(indices, mp_mesh)]
    else:
        # For k_j = (j + .5)/n - .5, negating k_j gives index n-1-j.
        partners = [n-1-idx for idx, n in zip(indices, mp_mesh)]
    nx, ny, nz = mp_mesh
    conj_kx, conj_ky, conj_kz = partners
    mapping = (conj_kx[:,None,None]*(ny*nz) +
               conj_ky[None,:,None]*nz + conj_kz).ravel()
    if not return_pair:
        return mapping

    idx = np.arange(len(mapping))
    mask = idx <= mapping
    return np.stack((idx[mask], mapping[mask])).T

def kk_adapted_iter(kmesh, with_gamma_point=True, return_pair_index=True):
    '''Group orbital k-point pairs by momentum transfer q = kj - ki.

    kmesh is the k-point sampling mesh supplied to
    cell.make_kpts(kmesh, with_gamma_point=with_gamma_point)

    q_mesh is the transfer mesh. It has the same value as the kmesh for orbitals.
    q_mesh must include gamma, regardless whether kpts are generated with
    "with_gamma_point", because the common k-shift in kj and ki cancels.
    For off-gamma sampling [2, 3, 1], q_mesh is [2, 3, 1], not the larger
    periodic BvK mesh [4, 3, 1] (by kpts_to_kmesh).

    The transfer k-points for q_mesh have the same ordering as
    qpts = cell.make_kpts(q_mesh, with_gamma_point=True).

    Return is an iterator for unique transfer k-point pairs, including
    self-conjugate transfers. Each item in the iterator is
    (q_idx, minus_q_idx, ki_idx, kj_idx),
    where the first two indices select q and -q from qpts. ki_idx and kj_idx are
    arrays index the k-points satisfying kj - ki = q . ki_idx is sorted.

    Don't use the q_idx and minus_q_idx to index orbital k-points. It should
    only be used for transfer mesh (though transfer mesh can be identical to the
    kmesh when with_gamma_point=True).
    '''
    kmesh = np.asarray(kmesh)
    nkpts = np.prod(kmesh)
    # The transfer mesh (q_mesh) always includes Gamma, regardless of
    # with_gamma_point.
    q_mesh_pairs = conj_kpts(kmesh, with_gamma_point=True, return_pair=True)

    if not return_pair_index:
        return q_mesh_pairs

    # double_translation_indices was originally used for Gamma-included
    # meshes. Given orbital indices i and j, kk_conserv[i, j] gives the index
    # of kj - ki on that same mesh.
    # For off-Gamma sampling, this array instead indexes the transfer mesh,
    # which follows the cell.make_kpts(kmesh, with_gamma_point=True) ordering.
    # To see why the same mapping works, consider one axis with n points:
    # off-Gamma coordinates are k_j = (j + .5)/n - .5. Shifting every point
    # by s = .5 - .5/n gives k_j + s = j/n, the Gamma-containing mesh with
    # the same index ordering. Each pair (i, j) keeps its transfer because
    # (k_j + s) - (k_i + s) = k_j - k_i. Thus off-Gamma meshes give the same
    # transfer index.
    kk_conserv = k2gamma.double_translation_indices(kmesh)

    # Evaluate ((kp[x], kp_conj[x], *np.where(kk_conserv == kp[x]))
    #           for x in independent_idx)
    # It is slow to execute np.where(kk_conserv == kp[x]).
    # np.unique is significantly faster

    _, inv_idx, counts = np.unique(
        kk_conserv.ravel(), return_inverse=True, return_counts=True)
    idx = np.argsort(inv_idx, kind='stable')
    splits = counts.cumsum()[:-1]
    kk_addrs = np.split(idx, splits)

    return ((q, q_conj, *divmod(kk_addrs[q], nkpts)) for q, q_conj in q_mesh_pairs)

def reset_kpts(kpts, cell):
    '''
    Update the absolute k-points of an object wrt the input cell,
    while preserving the same fractional (scaled) k-point coordinates.
    '''
    assert isinstance(kpts, KPoints)
    if hasattr(kpts, 'reset'):
        kpts = kpts.reset(cell)
    else: # kpts.reset() is not available in pyscf 2.10
        kpts.cell = cell
        kpts._built = False
        kpts.kpts = kpts.kpts_ibz = cell.get_abs_kpts(kpts.kpts_scaled)
        kpts.build(space_group_symmetry=cell.space_group_symmetry,
                   time_reversal_symmetry=kpts.time_reversal)
    return kpts

def fft_matrix(mp_mesh, with_gamma_point=True):
    '''Fourier coefficients from a periodic BvK supercell to MP k-points.

    mp_mesh gives the Monkhorst-Pack sampling mesh, not the larger
    periodic BvK mesh used by integral routines. If with_gamma_point is False,
    each even dimension is doubled to form the periodic supercell. Odd
    meshes are unchanged.

    The result has shape (bvk_ncells, nkpts). For example, shifted [2, 3, 1]
    sampling gives a (12, 6) matrix. Its coefficients are exp(1j*L.dot(k)),
    with no normalization factor, and transform integrals as

    S_k = einsum('iLj,LK->Kij', S_bvk, fft_matrix)

    Rows follow translation_vectors_for_kmesh for the periodic supercell.
    Columns follow cell.make_kpts(mp_mesh, with_gamma_point=with_gamma_point).
    '''
    mp_mesh = np.asarray(mp_mesh)
    nkpts = np.prod(mp_mesh)
    nx, ny, nz = mp_mesh
    if with_gamma_point:
        bvk_kmesh = mp_mesh
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kz = np.fft.fftfreq(nz)
    else:
        # For even n, kpts_to_kmesh creates larger BvK mesh, which requires
        # twice cells for periodic boundaries
        bvk_kmesh = np.where(mp_mesh % 2 == 0, 2*mp_mesh, mp_mesh)
        # cell.make_kpts(with_gamma_point=False) uses this ks ordering
        kx = (np.arange(nx) + .5) / nx - .5
        ky = (np.arange(ny) + .5) / ny - .5
        kz = (np.arange(nz) + .5) / nz - .5
    Lx = np.arange(bvk_kmesh[0])
    Ly = np.arange(bvk_kmesh[1])
    Lz = np.arange(bvk_kmesh[2])
    Fx = cp.exp(cp.asarray(2j*np.pi * Lx[:,None] * kx))
    Fy = cp.exp(cp.asarray(2j*np.pi * Ly[:,None] * ky))
    Fz = cp.exp(cp.asarray(2j*np.pi * Lz[:,None] * kz))
    expLk = (Fx[:,None,None,:,None,None] *
             Fy[None,:,None,None,:,None] *
             Fz[None,None,:,None,None,:])
    return expLk.reshape(np.prod(bvk_kmesh), nkpts)

def _normalize_kpts(kpts):
    '''Normalize k-points to an (nkpts, 3) array and flag single-point input.

    PySCF kpts convention:
        The input form determines whether integral results have a k-point axis.
        For one-electron integrals such as the overlap matrix:

        * None selects a single Gamma point and returns (nao, nao).
        * An array of shape (3,) selects a single k-point and returns (nao, nao).
        * An array of shape (nkpts, 3) selects a batch and returns
          (nkpts, nao, nao), even when nkpts == 1. For example,
          kpts=[[0, 0, 0]] retains the leading axis of length one.

        With None, the supplied cell is sampled at Gamma with periodic boundary
        conditions. Integrals are real when the operator permits it. Explicit
        single-point or batched input may produce complex arrays even when
        their values could be represented as real numbers.

        When present, axes are ordered as (spin, kpts, components, nao, nao).
        Ordinary one-electron integrals have no spin or component axis. Their
        derivatives have shape (ncomp, nao, nao) for single-point input and
        (nkpts, ncomp, nao, nao) for batched input.

    Args:
        kpts: None, array_like, or KPoints instance
            Sampling in the convention above. A KPoints instance supplies
            its full kpts array, rather than kpts_ibz, and is treated as a batch.

    Returns:
        kpts: ndarray of shape (nkpts, 3)
            Normalized sampling. None becomes a zero array of shape (1, 3).
        is_single_kpt: bool
            True for None or a (3,) input; False for batched input.
            This describes the input form, not whether sampling is at Gamma.
            The helper only normalizes sampling; callers use this flag to
            determine whether to omit the k-point axis from their output.
    '''
    is_single_kpt = True
    if kpts is None:
        return np.zeros((1, 3)), is_single_kpt

    if isinstance(kpts, KPoints):
        return kpts.kpts, not is_single_kpt

    kpts = np.asarray(kpts)
    if kpts.ndim == 1:
        return kpts.reshape(1, 3), is_single_kpt

    assert kpts.ndim == 2
    return kpts, not is_single_kpt

def _get_mp_mesh(kpts, bvk_mesh=None):
    '''Recover MP mesh from their minimal periodic BvK embedding.

    This temporary helper assumes a complete mesh in cell.make_kpts order
    and bvk_mesh from kpts_to_kmesh. It does not validate arbitrary
    k-point lists.

    Return (mp_mesh, with_gamma_point), which can be used to reproduce kpts via
    cell.make_kpts(mp_mesh, with_gamma_point)
    '''
    # TODO: if isinstance(kpts, KPoints):
    # TODO:     return kpts.mp_mesh, kpts.with_gamma_point

    # Assume kpts comes directly from cell.make_kpts, with no extra shift,
    # reordering or symmetry reduction. The BvK mesh must be the minimal
    # periodic embedding, not a larger user-selected or cutoff-limited mesh.
    # Under these assumptions, its size is 1, 2, 4 or 8 times len(kpts).
    assert bvk_mesh is not None
    mp_mesh = np.array(bvk_mesh, copy=True)
    bvk_ncells = np.prod(mp_mesh)
    nkpts = len(kpts)
    if bvk_ncells == nkpts:
        # Shifted all-odd meshes have the same size but a different ordering.
        assert abs(kpts[0]).max() < 1e-10, 'Expected Gamma as the first k-point'
        fft_order = True
        return mp_mesh, fft_order

    mp_mesh[mp_mesh % 2 == 0] //= 2
    if np.prod(mp_mesh) != nkpts:
        raise ValueError('bvk_mesh and k-point count do not describe a complete MP mesh')
    fft_order = False
    return mp_mesh, fft_order
