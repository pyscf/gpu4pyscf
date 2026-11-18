# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
from pyscf.pbc.tools.k2gamma import translation_map
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix, conj_kpts
from gpu4pyscf.pbc.lib.kpts_helper import kpts_to_bvkmesh as kpts_to_kmesh
from gpu4pyscf.lib.cupy_helper import asarray

def double_translation_indices(kmesh):
    '''Indices to utilize the translation symmetry in the 2D matrix.

    D[M,N] = D[N-M]

    The return index maps the 2D subscripts to 1D subscripts.

    D2 = D1[double_translation_indices()]

    D1 holds all the symmetry unique elements in D2
    '''

    tx = cp.array(translation_map(kmesh[0]), dtype=np.int32)
    ty = cp.array(translation_map(kmesh[1]), dtype=np.int32)
    tz = cp.array(translation_map(kmesh[2]), dtype=np.int32)
    idx = cp.ravel_multi_index([tx[:,None,None,:,None,None],
                                ty[None,:,None,None,:,None],
                                tz[None,None,:,None,None,:]],
                                tuple(int(n) for n in kmesh))
    nk = np.prod(kmesh)
    return idx.reshape(nk, nk)

def gamma2k_phase(kmesh, with_gamma_point=True):
    '''Mix conjugate k-point orbitals into real supercell orbitals.

    The returned phase matrix has shape (nkpts, nkpts), with nkpts = prod(kmesh).
    Together with expLk[L,k] = exp(i L.k) for translations L in the *minimal*
    BvK supercell, it transforms the k-sampled orbitals as

    C_super = np.einsum('Rk,kum,kh->Ruhm', expLk, C_k, k_phase)

    Orbitals at conjugate points must be complex conjugates. Self-conjugate
    points are left unchanged and their orbitals must already be real.

    kmesh gives the MP sampling dimensions. The k-points must follow
    cell.make_kpts(kmesh, with_gamma_point=with_gamma_point) ordering.

    For with_gamma_point=True, the *minimal* supercell has periodic boundaries
    and is sampled at Gamma. For with_gamma_point=False, its boundaries are
    antiperiodic along even MP dimensions and periodic along odd dimensions.
    This corresponds to one supercell k-point calculation with k=1/2 along
    each even dimension.
    '''
    Nk = np.prod(kmesh)
    k_phase = np.eye(Nk, dtype=np.complex128)
    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    pairs = conj_kpts(kmesh, with_gamma_point, return_pair=True)
    for idx in pairs:
        if idx[0] != idx[1]:
            k_phase[idx[:,None],idx] = r2x2
    return asarray(k_phase)
