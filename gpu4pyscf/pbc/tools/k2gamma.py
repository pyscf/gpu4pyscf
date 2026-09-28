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

from functools import reduce
from fractions import Fraction
import itertools
import numpy as np
import cupy as cp
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_map
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix, kk_adapted_iter
from gpu4pyscf.lib.cupy_helper import asarray

def kpts_to_kmesh(cell, kpts, precision=None):
    '''Return the dimensions of a full, uniform Monkhorst-Pack sampling mesh.

    A common shift does not change the sampling mesh. For lattice sums with
    periodic boundary conditions, use kpts_to_bvkmesh instead.
    '''
    if kpts is None:
        return np.ones(3, dtype=int)
    scaled = cell.get_scaled_kpts(np.asarray(kpts).reshape(-1, 3))
    if precision is None:
        precision = max(1e-6, cell.precision * 1e2)
    scaled = (scaled - scaled[0]) % 1
    scaled[abs(scaled - 1) < precision] = 0
    kmesh = np.array([
        1 + np.count_nonzero(np.diff(np.sort(x)) > precision)
        for x in scaled.T])
    indices = np.rint(scaled * kmesh).astype(int)
    error = scaled - indices / kmesh
    if (abs(error).max() > precision or np.prod(kmesh) != len(scaled)
        or len(np.unique(indices % kmesh, axis=0)) != len(scaled)):
        raise ValueError('kpts must form a full uniform Monkhorst-Pack mesh')
    return kmesh

def kpts_to_bvkmesh(cell, kpts, precision=None, rcut=None, bound_by_supmol=True):
    '''Search a phase-compatible, periodic BvK mesh for lattice sums.

    Unlike the sampling mesh, this mesh includes the denominator of the
    common k-point shift, so that exp(i k.L) = 1 across its boundaries.

    bound_by_supmol:
        If True, fall back to the cutoff-derived supercell when a commensurate
        period cannot be found. If False, require an exact periodic mesh.
    '''
    if kpts is None:
        return np.ones(3, dtype=int)
    assert kpts.ndim == 2
    if is_zero(kpts):
        return np.ones(3, dtype=int)

    scaled_kpts = cell.get_scaled_kpts(kpts)
    logger.debug3(cell, '    scaled_kpts kpts %s', scaled_kpts)
    if rcut is None:
        kmesh = np.asarray(cell.nimgs) * 2 + 1
    else:
        nimgs = cell.get_bounding_sphere(rcut)
        kmesh = nimgs * 2 + 1

    if precision is None:
        precision = max(1e-6, cell.precision * 1e2)
    for i in range(3):
        floats = scaled_kpts[:,i]
        uniq_floats_idx = np.unique((floats/precision+.5).astype(int), return_index=True)[1]
        uniq_floats = floats[uniq_floats_idx]
        max_denominator = max(int(kmesh[i])+10, 2*len(uniq_floats))
        fracs = [Fraction(x).limit_denominator(max_denominator) for x in uniq_floats]
        denominators = np.unique([x.denominator for x in fracs])
        common_denominator = reduce(np.lcm, denominators)
        fs = [(x * common_denominator).numerator for x in fracs]
        if cell.verbose >= logger.DEBUG3:
            logger.debug3(cell, 'dim=%d common_denominator %d  error %g',
                          i, common_denominator, abs(fs - np.rint(fs)).max())
            logger.debug3(cell, '    unique kpts %s', uniq_floats)
            logger.debug3(cell, '    frac kpts %s', fracs)
        if abs(uniq_floats - np.rint(fs)/common_denominator).max() < precision:
            kmesh[i] = common_denominator
        elif not bound_by_supmol:
            raise RuntimeError(f'Unable to find periodic BvK mesh for {kpts}')
    return kmesh

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
                                tz[None,None,:,None,None,:]], kmesh)
    nk = np.prod(kmesh)
    return idx.reshape(nk, nk)

def gamma2k_phase(kmesh, with_gamma_point=True):
    '''
    The k_phase can transform the k-points MOs to gamma-point MOs:
    C_gamma = np.einsum('Rk,kum,kh->Ruhm', fft_matrix(kmesh), C_k, k_phase)
    '''
    Nk = np.prod(kmesh)
    k_phase = np.eye(Nk, dtype=np.complex128)
    r2x2 = np.array([[1., 1j], [1., -1j]]) * .5**.5
    pairs = [[k, k_conj] for k, k_conj, _, _ in kk_adapted_iter(kmesh, with_gamma_point)
             if k != k_conj]
    for idx in np.array(pairs):
        k_phase[idx[:,None],idx] = r2x2
    return asarray(k_phase)
