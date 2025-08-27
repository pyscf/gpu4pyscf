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
    '''
    When BvK super-cell is constructed by the pbc.tools.super_cell function,
    images within the BvK cell has certain symmetry correspondence. These images
    correspond to the Monkhorst-Pack (MP) k-point grids. This function finds the
    mapping: given image index k, mapping[k] shows the conjugation image in MP grids.
    When return_pair is enabled, each entry in the pair shows the corresponding
    image indices, i.e. pair[i,0] and pair[i,1] are conjugated images.
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
    return np.column_stack((Ls_idx[mask], Ls_idx_conj[mask]))

def kk_adapted_iter(kmesh, with_gamma_point=True):
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
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    for x in independent_idx:
        ki, kj = np.where(kk_conserv == kp[x])
        yield kp[x], kp_conj[x], ki, kj

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

def fft_matrix(kmesh, with_gamma_point=True):
    '''
    A square matrix for the 3D Fourier transform coefficients. This matrix can
    be used to transform the integral from the BvK super-cell representation to
    unit-cell k-points representation:

    S_k = einsum('iLj,LK->Kij', S_bvk, fft_matrix)

    The transformation matrix is identical to
    Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh)
    exp(1j*Ls.dot(cell.make_kpts(kmesh, with_gamma_point=True).T))

    This function only works for the gamma-point included k-point mesh.
    '''
    assert with_gamma_point
    kmesh = np.asarray(kmesh)
    nkpts = np.prod(kmesh)
    nx, ny, nz = kmesh
    kx = cp.fft.fftfreq(nx)
    ky = cp.fft.fftfreq(ny)
    kz = cp.fft.fftfreq(nz)
    Lx = cp.arange(nx)
    Ly = cp.arange(ny)
    Lz = cp.arange(nz)
    Fx = cp.exp(2j*np.pi * Lx[:,None] * kx)
    Fy = cp.exp(2j*np.pi * Ly[:,None] * ky)
    Fz = cp.exp(2j*np.pi * Lz[:,None] * kz)
    expLk = (Fx[:,None,None,:,None,None] *
             Fy[None,:,None,None,:,None] *
             Fz[None,None,:,None,None,:])
    return expLk.reshape(nkpts, nkpts)
