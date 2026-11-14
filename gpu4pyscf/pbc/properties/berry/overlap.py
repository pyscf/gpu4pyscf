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
Neighboring-k-point overlaps for Wannier-center calculations.
Reference:
- tutorial: https://arxiv.org/abs/1202.1831v1
- working equations: PhysRevB.47.1651, PhysRevB.89.155114
- reference codes: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pywannier90.py
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from .kpt_string import KPointMesh

__all__ = [
    'KPointMesh',
    'periodic_ao_overlap',
    'build_mmn',
]


def _asnumpy(array):
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def _commensurate_bvk_mesh(cell, kpts, kmesh=None):
    # FTOpt folds outer BvK images without a twist phase. Every absolute
    # k-point, not just the spacing of the mesh, must be commensurate.
    scaled = np.asarray(cell.get_scaled_kpts(kpts))
    if (kmesh is not None and
            np.max(np.abs(scaled * kmesh - np.rint(scaled * kmesh))) <= 1e-8):
        return kmesh
    try:
        return kpts_to_kmesh(
            cell, kpts, precision=1e-10, bound_by_supmol=False)
    except RuntimeError as err:
        raise ValueError(
            'The GPU Fourier transform requires k-points commensurate '
            'with a finite BvK mesh') from err


def periodic_ao_overlap(cell, kpt, neighbor_kpt):
    '''Compute the overlap of cell-periodic Bloch AO factors.

    Args:
        cell : pyscf.pbc.gto.Cell
            Periodic cell defining the AO basis.
        kpt, neighbor_kpt : array_like, shape (3,)
            Cartesian k-points in Bohr^-1. For a closing link,
            neighbor_kpt must include the reciprocal-lattice image shift,
            e.g. k_0 + G, rather than only the wrapped k_0.

    Returns:
        cupy.ndarray, shape (nao, nao)
            Dimensionless overlap <f_mu,k | f_nu,k'> integrated over the
            reference cell, with f_mu,k(r) = exp(-i k.r) phi_mu,k(r).
            Rows belong to kpt and columns to neighbor_kpt.
    '''
    kpt = _asnumpy(kpt).reshape(3)
    neighbor_kpt = _asnumpy(neighbor_kpt).reshape(3)
    # Gv already contains the full momentum transfer, so q is zero.
    return ft_ao.ft_aopair(
        cell, (neighbor_kpt - kpt).reshape(1, 3),
        kpti_kptj=np.asarray([kpt, neighbor_kpt]), q=np.zeros(3))[0]


def build_mmn(cell, mo_coeff_kpts, kpts, kmesh=None, topology=None):
    '''Build M_mn(k,b) = <u_mk | u_n,k+b> for all three mesh directions.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional periodic cell defining the AO basis.
        mo_coeff_kpts : array_like, shape (nkpts, nao, nband)
            Bloch orbital coefficients in the same k-point order as kpts.
            Supply occupied orbitals for Berry phases and polarization.
        kpts : array_like, shape (nkpts, 3)
            Full uniform mesh in Cartesian coordinates, in Bohr^-1.
            Shifted meshes must be compatible with a finite BvK supercell.
        kmesh : array_like of int, shape (3,), optional
            Sampling mesh dimensions. Inferred from kpts if omitted.
        topology : KPointMesh, optional
            Precomputed topology for the same cell and kpts. When supplied,
            its kpts and kmesh are used.

    Returns:
        cupy.ndarray, shape (3, nkpts, nband, nband)
            Dimensionless overlaps for b_d = reciprocal_vectors[d]/kmesh[d].
            The first axis selects d = 0, 1, 2. Rows are bands at k and
            columns are bands at its +b_d neighbor, including the reciprocal
            image shift on closing links. All directions share one Fourier
            transform and are held in GPU memory.
    '''
    if cell.dimension != 3:
        raise NotImplementedError(
            'Wannier-center polarization currently supports 3D cells only')
    if topology is None:
        topology = KPointMesh(cell, kpts, kmesh)
    kpts = topology.kpts
    kmesh = topology.kmesh
    nkpts = len(kpts)

    mo_coeff_kpts = cp.asarray(mo_coeff_kpts)
    if (mo_coeff_kpts.ndim != 3 or
            mo_coeff_kpts.shape[:2] != (nkpts, cell.nao)):
        raise ValueError(
            'mo_coeff_kpts must have shape '
            f'({nkpts}, {cell.nao}, nband); got {mo_coeff_kpts.shape}')

    neighbor_indices = cp.asarray(np.asarray([
        topology.neighbors(direction)[0] for direction in range(3)]))
    reciprocal_steps = np.asarray([
        topology.reciprocal_step(direction) for direction in range(3)])

    bvk_mesh = _commensurate_bvk_mesh(cell, kpts, kmesh)
    ft_opt = ft_ao.FTOpt(cell, bvk_mesh)
    ft_opt.permutation_symmetry = False
    ft_kernel = ft_opt.gen_ft_kernel()

    # With k_j = k and Gv = -b, raw[k,d] is <f_mu,k+b | f_nu,k>.
    # Its adjoint gives the forward link while sharing kpts for all directions.
    raw = ft_kernel(-reciprocal_steps, q=np.zeros(3), kpts=kpts)
    s_ao = raw.conj().transpose(1, 0, 3, 2)
    return cp.matmul(
        mo_coeff_kpts.conj().transpose(0, 2, 1)[None],
        cp.matmul(s_ao, mo_coeff_kpts[neighbor_indices]))
