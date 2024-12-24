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

