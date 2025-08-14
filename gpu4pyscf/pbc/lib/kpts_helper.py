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
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints

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

def kk_adapted_iter(kmesh):
    '''Generates kpt which is adapted to the kpt_p in (ij|p)

    This function provides the similar functionality as the
    pyscf.pbc.lib.kpts_helper.kk_adapted_iter .
    '''
    kmesh = np.asarray(kmesh)
    nkpts = np.prod(kmesh)
    nx, ny, nz = kmesh
    kx = np.fft.fftfreq(nx, 1./nx).astype(int)
    ky = np.fft.fftfreq(ny, 1./ny).astype(int)
    kz = np.fft.fftfreq(nz, 1./nz).astype(int)

    kxyz = lib.cartesian_prod([kx, ky, kz])
    dk = (kxyz[None,:,:] - kxyz[:,None,:]).reshape(-1, 3)

    dk %= kmesh
    wrap_around_mask = dk >= (kmesh+1)//2
    dk[wrap_around_mask[:,0],0] -= nx
    dk[wrap_around_mask[:,1],1] -= ny
    dk[wrap_around_mask[:,2],2] -= nz
    uniq_ks, uniq_index, uniq_inverse = np.unique(
        dk, axis=0, return_index=True, return_inverse=True)

    ks_conj = -uniq_ks
    strides = np.array((ny*nz, nz, 1))
    ks_idx = (uniq_ks % kmesh).dot(strides)
    ks_idx_conj = (ks_conj % kmesh).dot(strides)

    independent_idx = np.sort(np.nonzero(ks_idx <= ks_idx_conj)[0])
    for x in independent_idx:
        kp = ks_idx[x]
        kp_conj = ks_idx_conj[x]
        kpt_ij_idx = np.where(uniq_inverse == x)[0]
        kpti_idx = kpt_ij_idx // nkpts
        kptj_idx = kpt_ij_idx % nkpts
        yield kp, kp_conj, kpti_idx, kptj_idx

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
