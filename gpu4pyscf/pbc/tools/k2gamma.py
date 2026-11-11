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

def kpts_to_kmesh(cell, kpts, precision=None, rcut=None, bound_by_supmol=True):
    '''Infer a mesh for constructing a BvK supercell with periodic boundary
    conditions for the given k-points.

    The k-points need not include Gamma, and prod(kmesh) need not equal
    len(kpts). For example, a shifted [2, 2, 2] Monkhorst-Pack mesh can be
    represented by a [2, 2, 2] supercell with antiperiodic boundary conditions.
    However, this function returns [4, 4, 4], corresponding to a larger
    supercell with periodic boundary conditions.

    Args:
        cell : Cell
            Provides the reciprocal lattice and lattice-sum cutoff.
        kpts : ndarray of shape (nkpts, 3) or None
            Absolute k-points in Bohr^-1. None selects Gamma.

    Kwargs:
        precision : float
            Tolerance for fractional k-point reconstruction. Defaults to
            max(1e-6, cell.precision * 1e2).
        rcut : float
            Lattice-sum radius used to set the rational-search denominator
            limit and fallback mesh. Defaults to using cell.nimgs.
        bound_by_supmol : bool
            If True, the k-mesh is bound by the radius cutoff.
            If False, the k-mesh are not capped by the radius cutoff.

    Returns:
        Integer ndarray of shape (3,) giving the BvK replication counts.
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
        fracs = [Fraction(x).limit_denominator(int(kmesh[i])+10) for x in uniq_floats]
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
            raise RuntimeError(f'Unable to find Monkhorst-Pack k-point mesh for {kpts}')
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
                                tz[None,None,:,None,None,:]],
                                tuple(int(n) for n in kmesh))
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
