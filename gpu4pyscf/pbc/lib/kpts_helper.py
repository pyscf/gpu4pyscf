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

    kmesh gives the periodic BvK dimensions. Image order follows super_cell.
    If return_pair is True, return each pair once, including self-pairs.
    For orbital k-point conjugation, use conj_kpts instead.
    '''
    nx, ny, nz = kmesh
    kx = np.arange(nx)
    ky = np.arange(ny)
    kz = np.arange(nz)
    conj_kx = (-kx) % nx
    conj_ky = (-ky) % ny
    conj_kz = (-kz) % nz

    nyz = ny * nz
    # When constructing super-cell in pbc.tools.super_cell, images are placed in
    # the cartesian_prod order, the image indices can be generated as:
    Ls_idx_conj = conj_kx[:,None,None]*nyz + conj_ky[:,None]*nz + conj_kz
    if not return_pair:
        return Ls_idx_conj.ravel()

    Ls_idx = kx[:,None,None]*nyz + ky[:,None]*nz + kz
    mask = Ls_idx <= Ls_idx_conj
    return np.vstack((Ls_idx[mask], Ls_idx_conj[mask])).T

def conj_kpts(mp_mesh, with_gamma_point=True, return_pair=False):
    '''Map each MP k-point to its -k partner

    mp_mesh gives the sampling dimensions, not the periodic BvK dimensions.
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
    '''Generates kpt which is adapted to the kpt_aux of the metric in RI
    for (ij| RI |kl). The metric is computed as (-kpt_aux|kpt_aux) where
    kpt_aux = kj - ki. The output is [idx(k), idx(-k), kpti_idx, kptj_idx].
    The kpti_idx in the output are sorted.

    This function provides the similar functionality as the
    pyscf.pbc.lib.kpts_helper.kk_adapted_iter .
    Note, the kk_adapted_iter function from pyscf supports arbitrary k-points,
    while this function only works for the k-mesh that contains gamma-point.
    '''
    assert with_gamma_point
    kmesh = np.asarray(kmesh)
    nkpts = np.prod(kmesh)
    nx, ny, nz = kmesh
    kx = np.fft.fftfreq(nx, 1./nx).astype(int)
    ky = np.fft.fftfreq(ny, 1./ny).astype(int)
    kz = np.fft.fftfreq(nz, 1./nz).astype(int)
    kxyz = lib.cartesian_prod([kx, ky, kz])
    # conjugated pairs are those kpt + kpt_conj = 2n\pi
    pair = (kxyz[None,:,:] + kxyz[:,None,:]).reshape(nkpts, nkpts, 3)
    pair %= kmesh # to apply wrap_around
    kp, kp_conj = np.where(pair.sum(axis=2) == 0)
    independent_idx = np.sort(np.where(kp <= kp_conj)[0])

    if not return_pair_index:
        return zip(kp[independent_idx], kp_conj[independent_idx])

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

    return ((kp[x], kp_conj[x], *divmod(kk_addrs[x], nkpts))
            for x in independent_idx)

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

    mp_mesh gives the Monkhorst-Pack sampling dimensions, not the larger
    periodic BvK mesh used by integral routines. If with_gamma_point is False,
    each even dimension is doubled to form the periodic supercell. Odd
    dimensions are unchanged.

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
