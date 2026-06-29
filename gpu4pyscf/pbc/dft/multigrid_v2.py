#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import warnings

import numpy as np
import cupy as cp
import cupyx.scipy.fft as fft
import scipy

import pyscf.pbc.gto as gto
from pyscf import lib
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.gto.mole import ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF
from pyscf.pbc.dft import gen_grid as pbc_gen_grid_cpu
from pyscf.pbc import tools as pbc_tools_cpu
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
from pyscf.pbc.lib.kpts_helper import is_gamma_point
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.gto.cell import get_Gv
from gpu4pyscf.pbc.tools import pbc as pbc_tools
import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
from gpu4pyscf.lib.cupy_helper import contract, tag_array, load_library, get_avail_mem


__all__ = ['MultiGridNumInt']

libgpbc = load_library("libmgrid_v2")
libgpbc.evaluate_density_driver.restype = ctypes.c_int
libgpbc.evaluate_xc_driver.restype = ctypes.c_int
libgpbc.evaluate_xc_gradient_driver.restype = ctypes.c_int
libgpbc.count_non_trivial_pairs.restype = ctypes.c_int
libgpbc.screen_gaussian_pairs.restype = ctypes.c_int
libgpbc.count_pairs_on_blocks.restype = ctypes.c_int


def complex_type(dtype):
    if dtype == cp.float32:
        return cp.complex64
    elif dtype == cp.float64:
        return cp.complex128
    else:
        raise ValueError("Invalid dtype")


def cast_to_pointer(array):
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError("Invalid array type")


def fft_in_place(x):
    return fft.fftn(x, axes=(-3, -2, -1), overwrite_x=True)


def ifft_in_place(x):
    return fft.ifftn(x, axes=(-3, -2, -1), overwrite_x=True)


def unique_with_sort(x):
    # This function does the same thing as cp.unique(x, return_inverse=True).
    # It's not super optimized, but for whatever reason, cp.unique is very slow, so this one is better.
    assert type(x) is cp.ndarray and (x.dtype == cp.int32 or x.dtype == cp.int64) and x.ndim == 1
    n = x.shape[0]
    if n <= 1:
        return x, cp.zeros(n)

    sort_index = cp.argsort(x)
    inverse_sort = cp.empty(n, dtype = cp.int64)
    inverse_sort[sort_index] = cp.arange(0, n, dtype = cp.int64)
    x = x[sort_index]

    mask = cp.empty(n, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = (x[1:] != x[:-1])

    x = x[mask]
    inverse_unique = cp.cumsum(mask, dtype=cp.int64) - 1

    return x, inverse_unique[inverse_sort]


def image_pair_to_difference(
    vectors_to_neighboring_images,
    lattice_vectors,
):
    '''
    Find unique image pairs for double lattice-sums associated with orbital products.

    When k-point phases are applied to orbital products with double lattice sum
        einsum('MmNn,Mk,Nk->kMN', orbital_prod_with_double_latsum, k_phase.conj(), k_phase)
    where k_phase = exp(1j*lattice_sum_images.dot(kpts)), the double lattice sum
    can be simplified to
        einsum('Tmn,Tk->kmn', orbital_prod, exp(1j*image_pair_diff.dot(kpts)))
    Here, T is the image_pair_to_difference produced by this function.
    The double lattice-sum over M,N within the orbital product can be pre-summed
    to certain images in T.

    Args:
        vectors_to_neighboring_images:
            Lattice sum vectors.
        lattice_vectors:
            Lattice vectors to define periodicity.

    Returns:
        A tuple containing:
        - The reduced lattice-sum vectors T for the unique image pairs.
        - A inverse mapping that restores the index of double lattice-sum from T.
    '''
    vectors_to_neighboring_images = cp.asarray(vectors_to_neighboring_images)
    lattice_vectors = cp.asarray(lattice_vectors)

    translation_vectors = cp.asarray(
        cp.linalg.solve(lattice_vectors.T, vectors_to_neighboring_images.T).T,
    )
    translation_vectors = cp.asarray(cp.round(translation_vectors), dtype = cp.int32)
    difference_images, inverse = _unique_image_pair(translation_vectors)
    difference_images = difference_images @ lattice_vectors

    # Given our pair data structure, the difference_images here should be interpretted as R2 - R1,
    # where R1 is associated with the first orbital in a pair, and R2 associated to the second.
    return cp.asarray(difference_images), cp.asarray(inverse, dtype=cp.int32)

def _unique_image_pair(translation_vectors):
    '''
    unqiue((-L[:,None] + L).reshape(-1, 3), axis=0, return_inverse=True)
    '''
    image_difference_full = (
        # -k_i + k_j corresponding to <i|j>
        translation_vectors[None,:,:] - translation_vectors[:,None,:]
    ).reshape(-1, 3)

    max_offset = (translation_vectors.max(axis=0) - translation_vectors.min(axis=0)).max() + 1
    assert (max_offset * 2)**3 < np.iinfo(np.int32).max
    image_difference_3in1 = image_difference_full
    image_difference_3in1 += max_offset
    image_difference_3in1 = image_difference_3in1[:, 0] * (max_offset * 2)**2 \
                          + image_difference_3in1[:, 1] * (max_offset * 2) \
                          + image_difference_3in1[:, 2]

    image_difference_3in1, inverse = unique_with_sort(image_difference_3in1)

    translation_vectors = cp.empty([image_difference_3in1.shape[0], 3], dtype = cp.int32)
    translation_vectors[:, 0] = image_difference_3in1 // (max_offset * 2)**2
    translation_vectors[:, 1] = (image_difference_3in1 % (max_offset * 2)**2) // (max_offset * 2)
    translation_vectors[:, 2] = image_difference_3in1 % (max_offset * 2)
    translation_vectors -= max_offset
    return translation_vectors, inverse

def image_phase_for_kpts(cell, neighboring_images, kpts=None):
    n_images = len(neighboring_images)
    if kpts is None or is_gamma_point(kpts):
        phase_diff_among_images = cp.asarray([[1.0]])
        image_pair_difference_index = cp.zeros((n_images, n_images), dtype=cp.int32)
    else:
        lattice_vectors = cell.lattice_vectors()
        difference_images, image_pair_difference_index = image_pair_to_difference(
            neighboring_images,
            lattice_vectors,
        )
        phase_diff_among_images = cp.exp(
            1j * cp.asarray(kpts.reshape(-1, 3)).dot(difference_images.T)
        )
    return phase_diff_among_images, image_pair_difference_index

def count_non_trivial_pairs(
    i_angular,
    j_angular,
    i_shells,
    j_shells,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    threshold_in_log,
):
    n_i_shells = len(i_shells)
    n_j_shells = len(j_shells)
    n_images = len(vectors_to_neighboring_images)
    n_pairs = cp.zeros(1, dtype=cp.int32)
    err = libgpbc.count_non_trivial_pairs(
        cast_to_pointer(n_pairs),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(i_shells),
        ctypes.c_int(n_i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(n_j_shells),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(n_images),
        (ctypes.c_int * 3)(*mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_double(threshold_in_log),
    )
    if err != 0:
        raise RuntimeError(f'count_non_trivial_pairs for li={i_angular} lj={j_angular} failed')
    return int(n_pairs[0])


def screen_gaussian_pairs(
    i_angular,
    j_angular,
    i_shells,
    j_shells,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    threshold_in_log,
):
    n_i_shells = len(i_shells)
    n_j_shells = len(j_shells)
    n_images = len(vectors_to_neighboring_images)
    n_pairs = count_non_trivial_pairs(
        i_angular,
        j_angular,
        i_shells,
        j_shells,
        vectors_to_neighboring_images,
        mesh,
        atm,
        bas,
        env,
        threshold_in_log,
    )
    screened_shell_pairs = cp.full(n_pairs, -1, dtype=cp.int32)
    image_indices = cp.full(n_pairs, -1, dtype=cp.int32)
    pairs_to_blocks_begin = cp.full((3, n_pairs), -1, dtype=cp.int32)
    pairs_to_blocks_end = cp.full((3, n_pairs), -1, dtype=cp.int32)
    err = libgpbc.screen_gaussian_pairs(
        cast_to_pointer(screened_shell_pairs),
        cast_to_pointer(image_indices),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(i_shells),
        ctypes.c_int(n_i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(n_j_shells),
        ctypes.c_int(n_pairs),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(n_images),
        (ctypes.c_int * 3)(*mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_double(threshold_in_log),
    )
    if err != 0:
        raise RuntimeError(f'screen_gaussian_pairs for li={i_angular} lj={j_angular} failed')
    return (
        screened_shell_pairs,
        image_indices,
        pairs_to_blocks_begin,
        pairs_to_blocks_end,
    )


def assign_pairs_to_blocks(
    pairs_to_blocks_begin,
    pairs_to_blocks_end,
    n_blocks_abc,
    n_indices,
    non_trivial_pairs,
    i_shells,
    j_shells,
    image_indices,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    has_warned_instability
):
    n_blocks = np.prod(n_blocks_abc)
    n_pairs_on_blocks = cp.zeros(n_blocks + 1, dtype=cp.int32)
    n_unstable_pairs_on_blocks = cp.zeros(n_blocks + 1, dtype = cp.int32)
    err = libgpbc.count_pairs_on_blocks(
        cast_to_pointer(n_pairs_on_blocks),
        cast_to_pointer(n_unstable_pairs_on_blocks),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(len(non_trivial_pairs)),
        cast_to_pointer(non_trivial_pairs),
        cast_to_pointer(i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(len(j_shells)),
        cast_to_pointer(image_indices),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(len(vectors_to_neighboring_images)),
        cast_to_pointer(mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env)
    )
    has_unstable_pairs = (n_unstable_pairs_on_blocks[-1] > 0)
    if not has_warned_instability and has_unstable_pairs:
        warnings.warn("Numerical instability may occur due to presence of core electrons or insufficient ke_cutoff.")
        has_warned_instability = True


    if err != 0:
        raise RuntimeError('count_pairs_on_blocks failed')

    n_contributing_blocks = int(n_pairs_on_blocks[-1])
    if n_contributing_blocks == 0:
        return (None, None, None, None)
    n_pairs_on_blocks = n_pairs_on_blocks[:-1]
    sorted_block_index = cp.asarray(cp.argsort(-n_pairs_on_blocks), dtype=cp.int32)
    accumulated_n_pairs_per_block = cp.zeros(n_blocks + 1, dtype=cp.int32)
    accumulated_n_pairs_per_block[1:] = cp.cumsum(n_pairs_on_blocks, dtype=cp.int32)
    sorted_block_index = sorted_block_index[:n_contributing_blocks]
    pairs_on_blocks = cp.full(n_indices, -1, dtype=cp.int32)
    libgpbc.put_pairs_on_blocks(
        cast_to_pointer(pairs_on_blocks),
        cast_to_pointer(accumulated_n_pairs_per_block),
        cast_to_pointer(sorted_block_index),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_contributing_blocks),
        ctypes.c_int(len(non_trivial_pairs)),
        cast_to_pointer(non_trivial_pairs),
        cast_to_pointer(i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(len(j_shells)),
        cast_to_pointer(image_indices),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(len(vectors_to_neighboring_images)),
        cast_to_pointer(mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env)
    )

    return (
        pairs_on_blocks,
        accumulated_n_pairs_per_block,
        sorted_block_index,
        has_warned_instability
    )


def multi_grids_tasks_lowmem(cell, fft_mesh=None, verbose=None, gamma_point=False, unrestricted=False):
    assert multigrid.TASKS_TYPE == 'ke_cut', "rcut scheme not supported yet"
    return multi_grids_tasks_for_ke_cut_lowmem(cell, fft_mesh, verbose, gamma_point, unrestricted)


def multi_grids_tasks_for_ke_cut_lowmem(cell, fft_mesh=None, verbose=None, gamma_point=False, unrestricted=False):
    """
        Modified from pyscf.pbc.dft.multigrid.multigrid.multi_grids_tasks_for_ke_cut()
        This function includes logic to split dense shells if the resulting fock matrix requires too much GPU memory.
    """
    log = logger.new_logger(cell, verbose)
    if fft_mesh is None:
        fft_mesh = cell.mesh

    # Split shells based on rcut
    rcuts_pgto, kecuts_pgto = multigrid._primitive_gto_cutoff(cell)
    ao_loc = cell.ao_loc_nr()

    # cell that needs dense integration grids
    def make_cell_dense_exp(shls_dense, ke0, ke1):
        cell_dense = cell.copy(deep=False)
        cell_dense._bas = cell._bas.copy()
        cell_dense._env = cell._env.copy()

        rcut_atom = [0] * cell.natm
        ke_cutoff = 0
        for ib in shls_dense:
            ke = kecuts_pgto[ib]
            idx = np.where((ke0 < ke) & (ke <= ke1))[0]
            nprim1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            nprim, nc = cs.shape
            if nprim1 < nprim:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_dense._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_dense._env[pexp:pexp+nprim1] = cell.bas_exp(ib)[idx]
                cell_dense._bas[ib,NPRIM_OF] = nprim1

            ke_cutoff = max(ke_cutoff, ke[idx].max())

            ia = cell.bas_atom(ib)
            rcut_atom[ia] = max(rcut_atom[ia], rcuts_pgto[ib][idx].max())
        cell_dense._bas = cell_dense._bas[shls_dense]
        ao_idx = np.hstack([np.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_dense])
        cell_dense.rcut = max(rcut_atom)
        return cell_dense, ao_idx, ke_cutoff, rcut_atom

    # cell that needs sparse integration grids
    def make_cell_sparse_exp(shls_sparse, ke0):
        cell_sparse = cell.copy(deep=False)
        cell_sparse._bas = cell._bas.copy()
        cell_sparse._env = cell._env.copy()

        for ib in shls_sparse:
            idx = np.where(kecuts_pgto[ib] <= ke0)[0]
            nprim1 = len(idx)
            cs = cell._libcint_ctr_coeff(ib)
            nprim, nc = cs.shape
            if nprim1 < nprim:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_sparse._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_sparse._env[pexp:pexp+nprim1] = cell.bas_exp(ib)[idx]
                cell_sparse._bas[ib,NPRIM_OF] = nprim1
        cell_sparse._bas = cell_sparse._bas[shls_sparse]
        ao_idx = np.hstack([np.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_sparse])
        return cell_sparse, ao_idx

    def get_nao_of_extracted_cell(shls_dense, original_cell, kecuts_pgto, ke0 = 0, ke1 = np.inf):
        nao_sph_nctr = 0 # Actual number of orbitals
        nao_cart_nprim = 0 # Number of primitives used for kernel
        for i_bas in shls_dense:
            bas = original_cell._bas[i_bas]
            ang = bas[ANG_OF]
            per_nao_sph = (2 * ang + 1)
            per_nao_cart = (ang + 2) * (ang + 1) // 2
            nctr = bas[NCTR_OF]

            # nprim = bas[NPRIM_OF]
            ke = kecuts_pgto[i_bas]
            idx = np.where((ke0 < ke) & (ke <= ke1))[0]
            nprim = len(idx)

            nao_sph_nctr += per_nao_sph * nctr
            nao_cart_nprim += per_nao_cart * nprim
        return nao_sph_nctr, nao_cart_nprim

    # Compute the max possible n_difference_images for memory partition
    if gamma_point:
        n_difference_images = 0
    else:
        max_neighboring_images = cp.asarray(gto.eval_gto.get_lattice_Ls(cell))
        fake_kpts = np.array([[0.5,0.5,0.5]])
        img_phase = image_phase_for_kpts(cell, max_neighboring_images, fake_kpts)
        phase_diff_among_images, image_pair_difference_index = img_phase
        n_difference_images = int(phase_diff_among_images.shape[1])
    n_channel = 2 if unrestricted else 1

    a = cell.lattice_vectors()
    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        init_mesh = multigrid.INIT_MESH_ORTH
    else:
        init_mesh = multigrid.INIT_MESH_NONORTH
    ke_cutoff_min = pbc_tools_cpu.mesh_to_cutoff(cell.lattice_vectors(), init_mesh)
    ke_cutoff_max = max([ke.max() for ke in kecuts_pgto])
    ke1 = ke_cutoff_min.min()
    ke_delimeter = [0, ke1]
    while ke1 < ke_cutoff_max:
        ke1 *= multigrid.KE_RATIO
        ke_delimeter.append(ke1)

    tasks = []
    for ke0, ke1 in zip(ke_delimeter[:-1], ke_delimeter[1:]):
        # shells which have high exps (small rcut)
        shls_dense = [ib for ib, ke in enumerate(kecuts_pgto)
                      if np.any((ke0 < ke) & (ke <= ke1))]
        if len(shls_dense) == 0:
            continue

        mesh = pbc_tools_cpu.cutoff_to_mesh(a, ke1)
        if multigrid.TO_EVEN_GRIDS:
            mesh = int((mesh+1)//2) * 2  # to the nearest even number

        ke1_capped = ke1
        if np.all(mesh >= fft_mesh):
            # Including all rest shells
            shls_dense = [ib for ib, ke in enumerate(kecuts_pgto)
                          if np.any(ke0 < ke)]
            ke1_capped = ke_cutoff_max+1

        dense_nao, dense_nprim_cart = get_nao_of_extracted_cell(shls_dense, cell, kecuts_pgto, ke0, ke1_capped)
        dense_nao, dense_nprim_cart = int(dense_nao), int(dense_nprim_cart)

        # shells which have low exps (big rcut)
        shls_sparse = [ib for ib, ke in enumerate(kecuts_pgto)
                       if np.any(ke <= ke0)]

        if len(shls_sparse) == 0:
            sparse_nao, sparse_nprim_cart = 0, 0
        else:
            sparse_nao, sparse_nprim_cart = get_nao_of_extracted_cell(shls_sparse, cell, kecuts_pgto, 0, ke0)
        sparse_nao, sparse_nprim_cart = int(sparse_nao), int(sparse_nprim_cart)

        sum_nprim_cart = dense_nprim_cart + sparse_nprim_cart

        fock_size = n_channel * n_difference_images * dense_nprim_cart * sum_nprim_cart
        if gamma_point:
            fock_nbytes_per_element = np.dtype(np.float64).itemsize
        else:
            # Why does it require a float64 and a complex128? Because when rotating the fock in image space to k space,
            # it converts the float64 fock matrix into complex128, so at that point, we need to store both.
            fock_nbytes_per_element = np.dtype(np.float64).itemsize + np.dtype(np.complex128).itemsize
        fock_nbytes = fock_size * fock_nbytes_per_element

        # At this stage almost no other memory is allocated,
        # and this number can be much lower when the fock matrix is actually built.
        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.2)

        n_split = (fock_nbytes + available_gpu_memory - 1) // available_gpu_memory
        if n_split > 1:
            log.warn(f"Warning: at dense shell ke range ({ke0}, {ke1_capped}], "
                     f"the fock matrix size ({fock_nbytes / 2**30} GiB) is too large, "
                     f"so the dense shells are split into {n_split} parts")

        def split_list_evenly(lst, n_piece):
            N = len(lst)
            if n_piece >= N:
                n_piece = N
            q, r = divmod(N, n_piece)
            out = []
            offset = 0
            for i in range(n_piece):
                size = q + (1 if i < r else 0)
                out.append(lst[offset:offset + size])
                offset += size
            return out

        shls_dense_split = split_list_evenly(shls_dense, n_split)
        shls_dense_cross = []

        mesh = np.min([mesh, fft_mesh], axis=0)

        if len(shls_sparse) == 0:
            cell_sparse = None
            ao_idx_sparse = []
        else:
            cell_sparse, ao_idx_sparse = make_cell_sparse_exp(shls_sparse, ke0)
            cell_sparse.mesh = mesh

        if cell_sparse is None:
            grids_sparse = None
        else:
            grids_sparse = pbc_gen_grid_cpu.UniformGrids(cell_sparse)
            grids_sparse.ao_idx = ao_idx_sparse

        for shls_dense in shls_dense_split:
            cell_dense, ao_idx_dense, _, _ = \
                        make_cell_dense_exp(shls_dense, ke0, ke1_capped)
            cell_dense.mesh = mesh

            grids_dense = pbc_gen_grid_cpu.UniformGrids(cell_dense)
            grids_dense.ao_idx = ao_idx_dense

            log.debug('mesh %s nao dense/sparse %d %d  rcut %g',
                    mesh, len(ao_idx_dense), len(ao_idx_sparse), cell_dense.rcut)

            if len(shls_dense_cross) > 0:
                cell_dense_cross, ao_idx_dense_cross, _, _ = \
                            make_cell_dense_exp(shls_dense_cross, ke0, ke1_capped)
                cell_dense_cross.mesh = mesh

                if cell_sparse is None:
                    grids_lower_triangular = pbc_gen_grid_cpu.UniformGrids(cell_dense_cross)
                    grids_lower_triangular.ao_idx = ao_idx_dense_cross
                else:
                    cell_lower_triangular = cell_sparse + cell_dense_cross
                    cell_lower_triangular._bas[cell_sparse.nbas:, ATOM_OF] -= len(cell_sparse._atm)

                    # Sort by atom first (later index has higher priority) to make aoslices work
                    bas_sort_by_atom_index = np.lexsort((cell_lower_triangular._bas[:,ANG_OF], cell_lower_triangular._bas[:,ATOM_OF]))

                    reverse_sort = np.argsort(bas_sort_by_atom_index)
                    ao_sort_by_atom_index = [[] for _ in range(cell_lower_triangular.nbas)]
                    ao_offset = 0
                    for i in range(cell_lower_triangular.nbas):
                        L = cell_lower_triangular._bas[i, ANG_OF]
                        nL = ((L+1)*(L+2)//2) if cell_lower_triangular.cart else (2*L+1)
                        nctr = cell_lower_triangular._bas[i, NCTR_OF]
                        ao_sort_by_atom_index[reverse_sort[i]] = np.arange(nL * nctr) + ao_offset
                        ao_offset += nL * nctr

                    ao_sort_by_atom_index = [int(item) for row in ao_sort_by_atom_index for item in row]
                    assert len(ao_sort_by_atom_index) == cell_lower_triangular.nao

                    cell_lower_triangular._bas = cell_lower_triangular._bas[bas_sort_by_atom_index]
                    ao_idx_lower_triangular = np.concatenate((ao_idx_sparse, ao_idx_dense_cross))
                    ao_idx_lower_triangular = ao_idx_lower_triangular[ao_sort_by_atom_index]
                    grids_lower_triangular = pbc_gen_grid_cpu.UniformGrids(cell_lower_triangular)
                    grids_lower_triangular.ao_idx = ao_idx_lower_triangular

                tasks.append([grids_dense, grids_lower_triangular])
            else:
                tasks.append([grids_dense, grids_sparse])

            shls_dense_cross.extend(shls_dense)

        if np.all(mesh >= fft_mesh):
            break
    return tasks


def sort_gaussian_pairs(mydf, xc_type="LDA", gamma_point=False, unrestricted=False):
    cell = mydf.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    vol = cell.vol
    block_size = np.array([4, 4, 4])
    lattice_vectors = cell.lattice_vectors()
    off_diagonal = lattice_vectors - np.diag(lattice_vectors.diagonal())
    is_non_orthogonal = np.any(np.abs(off_diagonal) > 1e-10)
    if is_non_orthogonal:
        is_non_orthogonal = 1
    else:
        is_non_orthogonal = 0
    reciprocal_lattice_vectors = np.asarray(np.linalg.inv(lattice_vectors.T), order="C")

    reciprocal_norms = np.linalg.norm(reciprocal_lattice_vectors, axis=1)
    libgpbc.update_lattice_vectors(
        lattice_vectors.ctypes,
        reciprocal_lattice_vectors.ctypes,
        reciprocal_norms.ctypes
    )

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        tasks = multi_grids_tasks_lowmem(cell, mydf.mesh, log, gamma_point, unrestricted)
        mydf.tasks = tasks

    t0 = log.timer("task generation", *t0)
    t1 = t0
    pairs = []
    for grids_localized, grids_diffused in tasks:
        subcell_in_localized_region = grids_localized.cell
        # the original grids_localized.mesh has dtype=np.int64, which can cause
        # misalignment when the pointer is passed to the C code.
        mesh = np.asarray(grids_localized.mesh, dtype=np.int32)

        fft_grid = list(
            map(
                lambda n_mesh_points: cp.round(cp.fft.fftfreq(
                    n_mesh_points, 1.0 / n_mesh_points
                )).astype(cp.int32),
                mesh,
            )
        )

        dxyz_dabc = lattice_vectors / mesh[:,None]
        libgpbc.update_dxyz_dabc(dxyz_dabc.ctypes)
        n_blocks_abc = np.asarray(np.ceil(mesh / block_size), dtype=cp.int32)
        equivalent_cell_in_localized, coeff_in_localized = (
            subcell_in_localized_region.decontract_basis(to_cart=True, aggregate=True)
        )

        n_primitive_gtos_in_localized = multigrid._pgto_shells(
            subcell_in_localized_region
        )

        # theoretically we can use the rcut defined in localized cell to reduce the
        # number of images, but somehow it can introduce some error when the lattice
        # is super small, for example primitive diamond cell. Using the rcut defined
        # in the global cell can fix this.
        vectors_to_neighboring_images = cp.asarray(gto.eval_gto.get_lattice_Ls(cell))

        if grids_diffused is None:
            grouped_cell = equivalent_cell_in_localized
            concatenated_coeff = scipy.linalg.block_diag(coeff_in_localized)
        else:
            subcell_in_diffused_region = grids_diffused.cell
            equivalent_cell_in_diffused, coeff_in_diffused = (
                subcell_in_diffused_region.decontract_basis(
                    to_cart=True, aggregate=True
                )
            )

            grouped_cell = equivalent_cell_in_localized + equivalent_cell_in_diffused

            grouped_cell._bas[n_primitive_gtos_in_localized:, ATOM_OF] -= len(
                subcell_in_localized_region._atm
            )

            concatenated_coeff = scipy.linalg.block_diag(
                coeff_in_localized, coeff_in_diffused
            )
        concatenated_coeff = cp.asarray(concatenated_coeff)

        n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
        rad = vol**(-1./3) * cell.rcut + 1
        surface = 4*np.pi * rad**2
        lattice_sum_factor = surface
        precision = cell.precision / lattice_sum_factor
        threshold_in_log = np.log(precision)

        shell_to_ao_indices = cp.asarray(
            gto.moleintor.make_loc(grouped_cell._bas, "cart"), dtype=cp.int32
        )
        ao_indices_in_localized = cp.asarray(grids_localized.ao_idx, dtype=cp.int32)
        if grids_diffused is None:
            ao_indices_in_diffused = cp.array([], dtype=cp.int32)
        else:
            ao_indices_in_diffused = cp.asarray(grids_diffused.ao_idx, dtype=cp.int32)

        concatenated_ao_indices = cp.concatenate(
            (ao_indices_in_localized, ao_indices_in_diffused)
        )
        coeff_in_localized = cp.asarray(coeff_in_localized)
        per_angular_pairs = []

        i_angulars = grouped_cell._bas[:n_primitive_gtos_in_localized, multigrid.ANG_OF]
        i_angulars_unique = np.unique(i_angulars)
        sorted_i_shells = []
        for l in i_angulars_unique:
            i_shells = cp.asarray(np.where(i_angulars == l)[0], dtype=cp.int32)
            sorted_i_shells.append(i_shells)

        j_angulars = grouped_cell._bas[
            :n_primitive_gtos_in_two_regions, multigrid.ANG_OF
        ]
        j_angulars_unique = np.unique(j_angulars)
        sorted_j_shells = []
        for l in j_angulars_unique:
            j_shells = cp.asarray(np.where(j_angulars == l)[0], dtype=cp.int32)
            sorted_j_shells.append(j_shells)

        atm = cp.asarray(grouped_cell._atm, dtype=cp.int32)
        bas = cp.asarray(grouped_cell._bas, dtype=cp.int32)
        env = cp.asarray(grouped_cell._env)

        t1 = log.timer_debug2("routines before screening", *t1)
        has_warned_instability = False
        for i_angular, i_shells in zip(i_angulars_unique, sorted_i_shells):
            for j_angular, j_shells in zip(j_angulars_unique, sorted_j_shells):
                (
                    screened_shell_pairs,
                    image_indices,
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                ) = screen_gaussian_pairs(
                    i_angular,
                    j_angular,
                    i_shells,
                    j_shells,
                    vectors_to_neighboring_images,
                    mesh,
                    atm,
                    bas,
                    env,
                    threshold_in_log,
                )
                t1 = log.timer_debug2(
                    "screening in angular pair" + str((i_angular, j_angular)), *t1
                )
                contributing_block_ranges = (
                    pairs_to_blocks_end - pairs_to_blocks_begin + 1
                )
                n_contributing_blocks_per_pair = cp.prod(
                    contributing_block_ranges, axis=0
                )
                n_indices = int(cp.sum(n_contributing_blocks_per_pair))
                (
                    gaussian_pair_indices,
                    accumulated_counts,
                    sorted_contributing_blocks,
                    has_warned_instability
                ) = assign_pairs_to_blocks(
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                    n_blocks_abc,
                    n_indices,
                    screened_shell_pairs,
                    i_shells,
                    j_shells,
                    image_indices,
                    vectors_to_neighboring_images,
                    mesh,
                    atm,
                    bas,
                    env,
                    has_warned_instability
                )
                if gaussian_pair_indices is None:
                    continue
                t1 = log.timer_debug2(
                    "assigning pairs to blocks in angular pair"
                    + str((i_angular, j_angular)),
                    *t1
                )
                per_angular_pairs.append(
                    {
                        "angular": (i_angular, j_angular),
                        "screened_shell_pairs": screened_shell_pairs,
                        "pair_indices_per_block": gaussian_pair_indices,
                        "accumulated_counts_per_block": accumulated_counts,
                        "sorted_block_index": sorted_contributing_blocks,
                        "image_indices": image_indices,
                        "i_shells": i_shells,
                        "j_shells": j_shells,
                        "shell_to_ao_indices": shell_to_ao_indices,
                    }
                )

        pairs.append(
            {
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": vectors_to_neighboring_images,
                "grouped_cell": grouped_cell,
                "mesh": mesh,  # this one is on cpu memory
                "fft_grid": fft_grid,
                "ao_indices_in_localized": ao_indices_in_localized,
                "ao_indices_in_diffused": ao_indices_in_diffused,
                "concatenated_ao_indices": concatenated_ao_indices,
                "coeff_in_localized": coeff_in_localized,
                "concatenated_coeff": concatenated_coeff,
                "atm": atm,
                "bas": bas,
                "env": env,
                "dxyz_dabc": dxyz_dabc,
                "is_non_orthogonal": is_non_orthogonal,
            }
        )

    mydf.sorted_gaussian_pairs = pairs

    t0 = log.timer("sort_gaussian_pairs", *t0)
    return mydf


def evaluate_density_wrapper(pairs_info, dm_slice, img_phase, ignore_imag=True, with_tau=False):
    if with_tau:
        c_driver = libgpbc.evaluate_density_tau_driver
    else:
        c_driver = libgpbc.evaluate_density_driver
    n_images = pairs_info["neighboring_images"].shape[0]
    phase_diff_among_images, image_pair_difference_index = img_phase
    n_k_points, n_difference_images = phase_diff_among_images.shape
    if n_k_points == 1 and n_difference_images == 1:
        density_matrix_with_translation = dm_slice
    else:
        # The conjugate here change e^{i \vec{k} \cdot (\vec{R}_2 - \vec{R}_1)} to
        # e^{i \vec{k} \cdot (\vec{R}_1 - \vec{R}_2)}
        # Because during grid density evaluation, rho = \sum_{\mu\nu} D_{\mu\nu} \mu \nu^*
        # The conjugate is on \nu, which is different from other Fock integrals
        density_matrix_with_translation = cp.einsum(
            "kt, ikpq->itpq", phase_diff_among_images.conj(), dm_slice
        )

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape

    if not ignore_imag:
        raise NotImplementedError
    else:
        pass
        # real_dm_imag_threshold = 1e-6
        # assert abs(density_matrix_with_translation.imag).max() < real_dm_imag_threshold, \
        #     f"The dm transformed into real space contains large imaginary part " \
        #     f"(max = {abs(density_matrix_with_translation.imag).max()}) >= {real_dm_imag_threshold}"
    density_matrix_with_translation_real_part = cp.asarray(
        density_matrix_with_translation.real, order="C"
    )

    if density_matrix_with_translation_real_part.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        assert density_matrix_with_translation_real_part.dtype == cp.float64
        use_float_precision = ctypes.c_int(0)
    assert density_matrix_with_translation_real_part.size < np.iinfo(np.int32).max

    if with_tau:
        density = cp.zeros((n_channels, 2, ) + tuple(pairs_info["mesh"]), dtype=density_matrix_with_translation_real_part.dtype)
    else:
        density = cp.zeros((n_channels,) + tuple(pairs_info["mesh"]), dtype=density_matrix_with_translation_real_part.dtype)

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]

        assert n_i_functions * n_j_functions < np.iinfo(np.int32).max
        # n_channels * n_i_functions * n_j_functions * n_difference_images is allowed to exceed int32
        err = c_driver(
            cast_to_pointer(density),
            cast_to_pointer(density_matrix_with_translation_real_part),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(image_pair_difference_index),
            ctypes.c_int(n_difference_images),
            (ctypes.c_int * 3)(*pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )
        if err != 0:
            raise RuntimeError(f'evaluate_density_driver for li={i_angular} lj={j_angular} failed')

    return density

def evaluate_density_on_g_mesh(mydf, dm_kpts, kpts=None, xc_type='LDA'):
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    n_channels, n_k_points = dms.shape[:2]
    if mydf.sorted_gaussian_pairs is None:
        mydf.build(xc_type)

    with_tau = False
    if xc_type == "LDA" or xc_type == 'HF':
        density_slices = 1
    elif xc_type == "GGA":
        density_slices = 4
    elif xc_type == "MGGA":
        density_slices = 5
        with_tau = True
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    cell = mydf.cell

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros(
        (n_channels, density_slices, nx, ny, nz), dtype=cp.complex128
    )
    for pairs in mydf.sorted_gaussian_pairs:

        mesh = pairs["mesh"]

        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1.0 / n_k_points * cell.vol / n_grid_points

        density_matrix_with_rows_in_localized = dms[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]

        density_matrix_with_rows_in_diffused = dms[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        n_ao_in_localized = density_matrix_with_rows_in_diffused.shape[3]
        density_matrix_with_rows_in_localized[
            :, :, :, n_ao_in_localized:
        ] += density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2).conj()

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkij,pi->nkpj",
            density_matrix_with_rows_in_localized,
            pairs["coeff_in_localized"],
        )

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkpj, qj -> nkpq",
            coeff_sandwiched_density_matrix,
            pairs["concatenated_coeff"],
        )

        libgpbc.update_dxyz_dabc(pairs["dxyz_dabc"].ctypes)

        img_phase = image_phase_for_kpts(cell, pairs["neighboring_images"], kpts)
        density = (
            evaluate_density_wrapper(
                pairs, coeff_sandwiched_density_matrix, img_phase, with_tau = with_tau
            )
            * weight_per_grid_point
        )

        if with_tau:
            assert density.shape[1] == 2
            tau = density[:, 1]
            density = density[:, 0]

        density = fft_in_place(density)

        density_on_g_mesh[
            :,
            0,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ] += density

        if with_tau:
            tau = fft_in_place(tau)

            density_on_g_mesh[
                :,
                4,
                pairs["fft_grid"][0][:, None, None],
                pairs["fft_grid"][1][:, None],
                pairs["fft_grid"][2],
            ] += tau

    density_on_g_mesh = density_on_g_mesh.reshape([n_channels, density_slices, -1])
    if xc_type == 'GGA' or xc_type == 'MGGA':
        density_on_g_mesh[:, 1:4] = get_Gv(mydf.cell, mydf.mesh).T
        density_on_g_mesh[:, 1:4] *= density_on_g_mesh[:, :1] * 1j
    return density_on_g_mesh
_eval_rhoG = evaluate_density_on_g_mesh

def evaluate_xc_wrapper(pairs_info, xc_weights, img_phase, with_tau=False):
    if with_tau:
        assert xc_weights.ndim == 3+2 and xc_weights.shape[1] == 2
        n_channels = xc_weights.shape[0]
        # density_slices = 2
    else:
        assert (xc_weights.ndim == 3+2 and xc_weights.shape[1] == 1) or (xc_weights.ndim == 3+1)
        n_channels = xc_weights.shape[0]
        # density_slices = 1

    if with_tau:
        c_driver = libgpbc.evaluate_xc_with_tau_driver
    else:
        c_driver = libgpbc.evaluate_xc_driver
    n_i_functions = len(pairs_info["coeff_in_localized"])
    n_j_functions = len(pairs_info["concatenated_coeff"])

    phase_diff_among_images, image_pair_difference_index = img_phase
    n_k_points, n_difference_images = phase_diff_among_images.shape
    n_images = pairs_info["neighboring_images"].shape[0]

    fock = cp.zeros(
        (n_channels, n_difference_images, n_i_functions, n_j_functions),
        dtype=xc_weights.dtype,
    )
    if xc_weights.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        assert xc_weights.dtype == cp.float64
        use_float_precision = ctypes.c_int(0)

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]

        assert n_i_functions * n_j_functions < np.iinfo(np.int32).max
        # n_channels * n_i_functions * n_j_functions * n_difference_images is allowed to exceed int32
        err = c_driver(
            cast_to_pointer(fock),
            cast_to_pointer(xc_weights),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(image_pair_difference_index),
            ctypes.c_int(n_difference_images),
            cast_to_pointer(pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )
        if err != 0:
            raise RuntimeError(f'evaluate_xc_driver for li={i_angular} lj={j_angular} failed')

    if not (n_k_points == 1 and n_difference_images == 1):
        return cp.einsum(
            "kt, ntij -> nkij", phase_diff_among_images, fock
        )
    else:
        return fock


def convert_xc_on_g_mesh_to_fock(
    mydf,
    xc_on_g_mesh,
    hermi=1,
    kpts=None,
    with_tau=False,
):
    cell = mydf.cell
    nao = cell.nao_nr()

    if with_tau:
        if xc_on_g_mesh.ndim == 2:
            assert xc_on_g_mesh.shape[0] == 2
            n_channels = 1
        elif xc_on_g_mesh.ndim == 3:
            assert xc_on_g_mesh.shape[1] == 2
            n_channels = xc_on_g_mesh.shape[0]
        else:
            raise ValueError("Incorrect shape of xc_on_g_mesh = {xc_on_g_mesh.shape}")
        density_slices = 2
    else:
        if xc_on_g_mesh.ndim == 1:
            n_channels = 1
        elif xc_on_g_mesh.ndim == 2:
            n_channels = xc_on_g_mesh.shape[0]
        elif xc_on_g_mesh.ndim == 3:
            assert xc_on_g_mesh.shape[1] == 1
            n_channels = xc_on_g_mesh.shape[0]
        else:
            raise ValueError("Incorrect shape of xc_on_g_mesh = {xc_on_g_mesh.shape}")
        density_slices = 1

    xc_on_g_mesh = xc_on_g_mesh.reshape(n_channels, density_slices, *mydf.mesh)

    if kpts is None:
        n_k_points = 1
        at_gamma_point = True
    else:
        assert kpts.ndim == 2
        n_k_points = len(kpts)
        at_gamma_point = multigrid.gamma_point(kpts)

    if hermi != 1:
        raise NotImplementedError

    data_type = cp.float64
    if not at_gamma_point:
        data_type = complex_type(cp.float64)

    fock = cp.zeros((n_channels, n_k_points, nao, nao), dtype=data_type)

    for pairs in mydf.sorted_gaussian_pairs:
        interpolated_xc = xc_on_g_mesh[
            :,
            :,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ]
        interpolated_xc = cp.asarray(ifft_in_place(interpolated_xc).real, order="C")

        n_ao_in_localized = len(pairs["ao_indices_in_localized"])
        libgpbc.update_dxyz_dabc(pairs["dxyz_dabc"].ctypes)
        img_phase = image_phase_for_kpts(cell, pairs["neighboring_images"], kpts)
        fock_slice = evaluate_xc_wrapper(pairs, interpolated_xc, img_phase, with_tau=with_tau)
        fock_slice = cp.einsum("nkpq,pi->nkiq", fock_slice, pairs["coeff_in_localized"])
        fock_slice = cp.einsum("nkiq,qj->nkij", fock_slice, pairs["concatenated_coeff"])

        def atomic_add_complex128(dst, idx, src):
            if src.dtype == cp.float64:
                assert dst.dtype == cp.float64
                cp.add.at(dst, idx, src)
            else:
                assert dst.dtype == cp.complex128
                assert src.dtype == cp.complex128
                # Cupy doesn't allow atomic addition for complex128, so we need to add real and imag parts separately.
                cp.add.at(dst.real, idx, src.real)
                cp.add.at(dst.imag, idx, src.imag)

        atomic_add_complex128(fock,
            (slice(None), slice(None), pairs["ao_indices_in_localized"][:, None], pairs["ao_indices_in_localized"][None, :]),
            fock_slice[:, :, :, :n_ao_in_localized])

        atomic_add_complex128(fock,
            (slice(None), slice(None), pairs["ao_indices_in_localized"][:, None], pairs["ao_indices_in_diffused"][None, :]),
            fock_slice[:, :, :, n_ao_in_localized:])

        if hermi == 1:
            atomic_add_complex128(fock,
                (slice(None), slice(None), pairs["ao_indices_in_diffused"][:, None], pairs["ao_indices_in_localized"][None, :]),
                fock_slice[:, :, :, n_ao_in_localized:].transpose(0, 1, 3, 2).conj())
        else:
            raise NotImplementedError

    return fock


def evaluate_xc_gradient_wrapper(
    gradient, pairs_info, xc_weights, dm_slice, img_phase, ignore_imag=True, with_tau=False
):
    if with_tau:
        assert xc_weights.ndim == 3+2 and xc_weights.shape[1] == 2
        n_channels = xc_weights.shape[0]
        # density_slices = 2
    else:
        assert (xc_weights.ndim == 3+2 and xc_weights.shape[1] == 1) or (xc_weights.ndim == 3+1)
        n_channels = xc_weights.shape[0]
        # density_slices = 1

    if with_tau:
        c_driver = libgpbc.evaluate_xc_with_tau_gradient_driver
    else:
        c_driver = libgpbc.evaluate_xc_gradient_driver

    assert gradient.dtype == xc_weights.dtype

    if gradient.dtype == cp.float32:
        use_float_precision = ctypes.c_int(1)
    else:
        use_float_precision = ctypes.c_int(0)

    n_images = pairs_info["neighboring_images"].shape[0]
    phase_diff_among_images, image_pair_difference_index = img_phase
    n_k_points, n_difference_images = phase_diff_among_images.shape

    if n_k_points == 1 and n_difference_images == 1:
        density_matrix_with_translation = dm_slice
    else:
        density_matrix_with_translation = cp.einsum(
            "kt, ikpq->itpq", phase_diff_among_images.conj(), dm_slice
        )

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape
    if ignore_imag is False:
        raise NotImplementedError

    density_matrix_with_translation_real_part = cp.asarray(
        density_matrix_with_translation.real, order="C"
    )

    assert gradient.dtype == density_matrix_with_translation_real_part.dtype

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]

        assert n_i_functions * n_j_functions < np.iinfo(np.int32).max
        # n_channels * n_i_functions * n_j_functions * n_difference_images is allowed to exceed int32
        err = c_driver(
            cast_to_pointer(gradient),
            cast_to_pointer(xc_weights),
            cast_to_pointer(density_matrix_with_translation_real_part),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["screened_shell_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(gaussians_per_angular_pair["pair_indices_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["accumulated_counts_per_block"]),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(image_pair_difference_index),
            ctypes.c_int(n_difference_images),
            cast_to_pointer(pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
            use_float_precision,
        )
        if err != 0:
            raise RuntimeError(f'evaluate_xc_gradient_driver for li={i_angular} lj={j_angular} failed')


def convert_xc_on_g_mesh_to_fock_gradient(
    mydf,
    xc_on_g_mesh,
    dm_kpts,
    hermi=1,
    kpts=None,
    with_tau=False,
):
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    n_atoms = cell.natm

    assert xc_on_g_mesh.ndim == 3
    n_channels = xc_on_g_mesh.shape[0]
    density_slices = xc_on_g_mesh.shape[1]
    xc_on_g_mesh = xc_on_g_mesh.reshape(n_channels, density_slices, *mydf.mesh)

    if hermi != 1:
        raise NotImplementedError

    gradient = cp.zeros((n_atoms, 3))

    for pairs in mydf.sorted_gaussian_pairs:
        interpolated_xc = xc_on_g_mesh[
            :,
            :,
            pairs["fft_grid"][0][:, None, None],
            pairs["fft_grid"][1][:, None],
            pairs["fft_grid"][2],
        ]

        interpolated_xc = cp.asarray(ifft_in_place(interpolated_xc).real, order="C")

        density_matrix_slice = dms[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]
        density_matrix_with_rows_in_diffused = dms[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        n_ao_in_localized = density_matrix_slice.shape[2]
        density_matrix_slice[
            :, :, :, n_ao_in_localized:
        ] += density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2).conj()

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkij,pi->nkpj",
            density_matrix_slice,
            pairs["coeff_in_localized"],
        )

        coeff_sandwiched_density_matrix = cp.einsum(
            "nkpj, qj -> nkpq",
            coeff_sandwiched_density_matrix,
            pairs["concatenated_coeff"],
        )

        libgpbc.update_dxyz_dabc(pairs["dxyz_dabc"].ctypes)

        img_phase = image_phase_for_kpts(cell, pairs["neighboring_images"], kpts)
        evaluate_xc_gradient_wrapper(
            gradient,
            pairs,
            interpolated_xc,
            coeff_sandwiched_density_matrix,
            img_phase,
            ignore_imag=True,
            with_tau=with_tau,
        )

    return gradient

#FIXME: merge to multigrid_v1.get_pp
def get_nuc(ni, kpts=None):
    if ni.sorted_gaussian_pairs is None:
        ni.build()
    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    cell = ni.cell
    mesh = ni.mesh
    vneG = multigrid_v1.eval_nucG(cell, mesh)
    hermi = 1
    vne = convert_xc_on_g_mesh_to_fock(ni, vneG, hermi, kpts)[0]
    if is_single_kpt:
        vne = vne[0]
    return vne

#FIXME: merge to multigrid_v1.get_pp
def get_pp(ni, kpts=None):
    """Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed."""
    if ni.sorted_gaussian_pairs is None:
        ni.build()
    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    cell = ni.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    mesh = ni.mesh
    # Compute the vpplocG as
    # -einsum('ij,ij->j', pseudo.get_vlocG(cell, Gv), cell.get_SI(Gv))
    vpplocG = multigrid_v1.eval_vpplocG(cell, mesh)
    vpp = convert_xc_on_g_mesh_to_fock(ni, vpplocG, hermi=1, kpts=kpts)[0]
    t1 = log.timer_debug1("vpploc", *t0)

    vppnl = get_pp_nl_gpu(cell, kpts)
    for k, kpt in enumerate(kpts):
        if is_single_kpt:
            vpp[k] += cp.asarray(vppnl[k].real)
        else:
            vpp[k] += cp.asarray(vppnl[k])

    if is_single_kpt:
        vpp = vpp[0]
    log.timer_debug1("vppnl", *t1)
    log.timer("get_pp", *t0)
    return vpp

def get_j_kpts(ni, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (*, nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (*, nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    cell = ni.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    mesh = ni.mesh
    ngrids = np.prod(mesh)

    density = evaluate_density_on_g_mesh(ni, dm_kpts, kpts)
    Gv = get_Gv(cell, mesh)
    coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)

    coulomb_on_g_mesh = cp.einsum(
        "ng, g -> ng", density[:, 0], coulomb_kernel_on_g_mesh
    )
    weight = cell.vol / ngrids

    density = density.reshape(-1, *mesh)
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density = ifft_in_place(density).real.reshape(nset, -1, ngrids)
    density /= weight

    #if kpts_band is not None:
    #    ni = ni.copy().reset().build()
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    xc_for_fock = convert_xc_on_g_mesh_to_fock(ni, coulomb_on_g_mesh, hermi, kpts_band)
    t0 = log.timer("vj", *t0)
    return _format_jks(xc_for_fock, dm_kpts, input_band, kpts)

def nr_rks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.
        with_j : bool
            Whether to add the Coulomb matrix into the XC matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    xc_type = ni._xc_type(xc_code)
    if ni.sorted_gaussian_pairs is None:
        ni.build(xc_type)

    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    dms = None
    assert nset == 1

    mesh = ni.mesh
    ngrids = np.prod(mesh)

    density = evaluate_density_on_g_mesh(ni, dm_kpts, kpts, xc_type)
    rho_sf = density[0, 0]

    Gv = get_Gv(cell, mesh)
    coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)
    coulomb_on_g_mesh = rho_sf * coulomb_kernel_on_g_mesh
    coulomb_energy = complex(rho_sf.conj().dot(coulomb_on_g_mesh).get())
    coulomb_energy = (0.5 / cell.vol) * coulomb_energy
    log.debug("Multigrid Coulomb energy %s", coulomb_energy)
    t0 = log.timer("coulomb", *t0)
    weight = cell.vol / ngrids

    density = ifft_in_place(density.reshape(-1, *mesh)).real.reshape(-1, ngrids)
    n_electrons = float(density[0].sum().real.get())
    density /= weight

    # eval_xc_eff supports float64 only
    density = cp.asarray(density, dtype=np.float64, order='C')
    xc_for_energy, xc_for_fock = ni.eval_xc_eff(
        xc_code, density, deriv=1, xctype=xc_type, spin=0
    )[:2]

    rho_sf = density[0].real
    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock *= weight
    xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(-1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == "GGA":
        xc_for_fock[0] -= cp.einsum("gp, pg -> p", xc_for_fock[1:4], Gv) * 1j
        xc_for_fock = xc_for_fock[0].reshape((-1, ngrids))
    elif xc_type == "MGGA":
        xc_for_fock[0] -= cp.einsum("gp, pg -> p", xc_for_fock[1:4], Gv) * 1j
        xc_for_fock = cp.concatenate([
            xc_for_fock[0].reshape((-1, ngrids)),
            xc_for_fock[4].reshape((-1, ngrids)),
        ], axis = 0)
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[0] += coulomb_on_g_mesh

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    veff = convert_xc_on_g_mesh_to_fock(ni, xc_for_fock, hermi, kpts_band, with_tau = (xc_type == "MGGA"))
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(veff, ecoul=coulomb_energy, exc=xc_energy_sum)
    t0 = log.timer("xc", *t0)
    return n_electrons, xc_energy_sum, veff

# Note nr_uks handles only one set of KUKS density matrices (alpha, beta) in
# each call (nr_rks supports multiple sets of KRKS density matrices)
def nr_uks(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=False, verbose=None):
    '''Compute the XC energy and UKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.
        with_j : bool
            Whether to add the Coulomb matrix into the XC matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    xc_type = ni._xc_type(xc_code)
    if ni.sorted_gaussian_pairs is None:
        ni.build(xc_type)

    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    dms = None
    assert nset == 2

    mesh = ni.mesh
    ngrids = np.prod(mesh)

    density = evaluate_density_on_g_mesh(ni, dm_kpts, kpts, xc_type)
    rho_sf = density[0, 0] + density[1, 0]

    Gv = get_Gv(cell, mesh)
    coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)
    coulomb_on_g_mesh = rho_sf * coulomb_kernel_on_g_mesh
    coulomb_energy = rho_sf.conj().dot(coulomb_on_g_mesh).real
    coulomb_energy = 0.5 * float(coulomb_energy.get())
    coulomb_energy /= cell.vol
    log.debug("Multigrid Coulomb energy %s", coulomb_energy)
    t0 = log.timer("coulomb", *t0)
    weight = cell.vol / ngrids

    density = density.reshape(-1, *mesh)
    density = ifft_in_place(density).real.reshape(nset, -1, ngrids)
    n_electrons = density[:, 0].sum(axis=-1).get()
    density /= weight

    # eval_xc_eff supports float64 only
    density = cp.asarray(density, dtype=np.float64, order='C')
    xc_for_energy, xc_for_fock = ni.eval_xc_eff(
        xc_code, density, deriv=1, xctype=xc_type, spin=1
    )[:2]

    rho_sf = (density[0, 0] + density[1, 0]).real
    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).real.get()) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock *= weight
    xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(nset, -1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == "GGA":
        xc_for_fock = (
            xc_for_fock[:, 0] - contract("ngp, pg -> np", xc_for_fock[:, 1:4], Gv) * 1j
        )
        xc_for_fock = xc_for_fock.reshape((nset, -1, ngrids))
    elif xc_type == "MGGA":
        xc_for_fock[:, 0] -= contract("ngp, pg -> np", xc_for_fock[:, 1:4], Gv) * 1j
        xc_for_fock = cp.concatenate([
            xc_for_fock[:, 0].reshape((nset, -1, ngrids)),
            xc_for_fock[:, 4].reshape((nset, -1, ngrids)),
        ], axis = 1)
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    veff = convert_xc_on_g_mesh_to_fock(ni, xc_for_fock, hermi, kpts_band, with_tau = (xc_type == "MGGA"))
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(veff, ecoul=coulomb_energy, exc=xc_energy_sum)
    t0 = log.timer("xc", *t0)
    return n_electrons, xc_energy_sum, veff

def get_rho(ni, dm, kpts=None):
    '''Density in real space

    Args:
        ni:
            MultiGridNumInt instance
        dm:
            density matrix at a single k-point or density matrices for k-sampling

    Kwargs:
        kpts: (N, 3) ndarray
            k points. If not specified, gamma point is assumed
    '''
    cell = ni.cell
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    density = evaluate_density_on_g_mesh(ni, dm, kpts)
    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = ifft_in_place(density.reshape(-1, *mesh)).real / weight
    assert rhoR.size == ngrids
    return rhoR.ravel()

def get_veff_ip1(
    ni,
    xc_code,
    dm_kpts,
    hermi=1,
    kpts=None,
    with_j=True,
    with_pseudo_vloc_orbital_derivative=True,
    verbose=None,
):
    '''Computes the derivatives of the Exc along with additional contributions
    from the Coulomb and pseudopotential terms.

    Note, the current return is the energy per cell scaled by the number of
    k-points. This should return the energy per cell directly and will be
    changed in future.
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    log = logger.new_logger(ni, verbose)
    t0 = log.init_timer()
    cell = ni.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    dms = None

    xc_type = ni._xc_type(xc_code)
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    density = evaluate_density_on_g_mesh(ni, dm_kpts, kpts, xc_type)

    Gv = get_Gv(cell, mesh)
    coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)
    coulomb_on_g_mesh = cp.einsum(
        "ng, g -> g", density[:, 0], coulomb_kernel_on_g_mesh
    )

    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density = (
        cp.asarray(
            ifft_in_place(density.reshape(nset, -1, *mesh)).real,
            order="C",
        ).reshape(nset, -1, ngrids)
        / weight
    )

    if nset == 1: # RHF
        xc_for_fock = ni.eval_xc_eff(
            xc_code, density[0], deriv=1, xctype=xc_type, spin=0
        )[1]
    else: # UHF
        assert nset == 2
        xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type, spin=1
        )[1]

    xc_for_fock = xc_for_fock.reshape(nset, -1, *mesh) * weight
    xc_for_fock = fft_in_place(xc_for_fock).reshape(nset, -1, ngrids)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == "GGA":
        xc_for_fock = (
            xc_for_fock[:, 0] - contract("ngp, pg -> np", xc_for_fock[:, 1:4], Gv) * 1j
        )
        xc_for_fock = xc_for_fock.reshape((nset, -1, ngrids))
    elif xc_type == "MGGA":
        xc_for_fock[:, 0] -= contract("ngp, pg -> np", xc_for_fock[:, 1:4], Gv) * 1j
        xc_for_fock = cp.concatenate([
            xc_for_fock[:, 0].reshape((nset, -1, ngrids)),
            xc_for_fock[:, 4].reshape((nset, -1, ngrids)),
        ], axis = 1)
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    if with_pseudo_vloc_orbital_derivative:
        if cell._pseudo:
            xc_for_fock[:, 0] += multigrid_v1.eval_vpplocG(cell, mesh)
        else:
            xc_for_fock[:, 0] += multigrid_v1.eval_nucG(cell, mesh)

    veff_gradient = convert_xc_on_g_mesh_to_fock_gradient(
        ni, xc_for_fock, dm_kpts, hermi, kpts, with_tau = (xc_type == "MGGA")
    )

    t0 = log.timer("veff_gradient", *t0)

    return veff_gradient

class MultiGridNumInt(lib.StreamObject, numint.LibXCMixin):
    def __init__(self, cell):
        self.cell = cell
        self.mesh = cell.mesh
        self.tasks = None
        self.sorted_gaussian_pairs = None

    build = sort_gaussian_pairs

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.tasks = None
        self.sorted_gaussian_pairs = None
        return self

    def get_j(self, dm, hermi=1, kpts=None, kpts_band=None):
        vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj

    get_nuc = get_nuc
    get_pp = get_pp

    get_rho = get_rho
    nr_rks = nr_rks
    nr_uks = nr_uks
    get_vxc = nr_vxc = NotImplemented

    eval_xc_eff = numint.NumInt.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

    def nr_rks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0, fxc=None,
                   kpts=None, with_j=False):
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts_ibz

        assert kpts.ndim == 2
        assert dms.ndim == 4
        nset, nkpts, nao = dms.shape[:3]
        assert len(kpts) == nkpts

        # The transition density matrices dm1 must be hermitian. The
        # evaluate_density_on_g_mesh function only supports real density.
        assert hermi == 1
        v_hermi = hermi

        xctype = self._xc_type(xc_code)
        if xctype == 'HF':
            return cp.zeros_like(dms)

        assert xctype in ('LDA', 'GGA', 'MGGA')

        if fxc is None:
            spin = 0
            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts, is_rhf=True)[2]

        mesh = self.mesh
        Gv = get_Gv(cell, mesh)
        ngrids = len(Gv)
        rho1 = evaluate_density_on_g_mesh(self, dms, kpts, xctype)
        if with_j:
            coulG = pbc_tools.get_coulG(cell, Gv=Gv)
            coulomb_on_g_mesh = rho1[:,0] * coulG
        rho1 = ifft_in_place(rho1.reshape(-1, *mesh)).real.reshape(nset, -1, ngrids)
        wv = cp.einsum('nxg,xyg->nyg', rho1, fxc)
        wv = fft_in_place(wv.reshape(-1, *mesh)).reshape(wv.shape)

        if with_j:
            wv[:,0] += coulomb_on_g_mesh

        if 'GGA' in xctype:
            wv[:,0] -= contract('nxp,xp->np', wv[:,1:4], Gv.T) * 1j
            if xctype == 'GGA':
                wv = cp.asarray(wv[:,0], order='C')
            elif xctype == 'MGGA':
                wv = cp.asarray(wv[:,[0, 4]], order='C')

        with_tau = (xctype == 'MGGA')
        vmat = convert_xc_on_g_mesh_to_fock(self, wv, v_hermi, kpts, with_tau=with_tau)
        return vmat.reshape(dms.shape)

    def nr_rks_fxc_st(self, cell, grids, xc_code, dm0, dms, hermi=0, singlet=True,
                      fxc=None, kpts=None, with_j=False):
        if fxc is None:
            spin = 1
            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts,
                                      is_rhf=True)[2]
        if singlet:
            fxc = fxc[0,:,0] + fxc[0,:,1]
        else:
            fxc = fxc[0,:,0] - fxc[0,:,1]
        return self.nr_rks_fxc(cell, grids, xc_code, dm0, dms, hermi, fxc, kpts, with_j)

    def nr_uks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0, fxc=None,
                   kpts=None, with_j=False):
        if kpts is None:
            kpts = np.zeros((1,3))
        elif isinstance(kpts, KPoints):
            kpts = kpts.kpts_ibz

        assert kpts.ndim == 2
        assert dms.ndim == 5
        nset, nkpts, nao = dms.shape[1:4]
        assert len(kpts) == nkpts

        # The transition density matrices dm1 must be hermitian. The
        # evaluate_density_on_g_mesh function only supports real density.
        assert hermi == 1
        v_hermi = hermi

        xctype = self._xc_type(xc_code)
        if xctype == 'HF':
            return cp.zeros_like(dms)

        assert xctype in ('LDA', 'GGA', 'MGGA')

        if fxc is None:
            spin = 1
            fxc = self.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts, is_rhf=False)[2]

        mesh = self.mesh
        Gv = get_Gv(cell, mesh)
        ngrids = len(Gv)
        rho1 = evaluate_density_on_g_mesh(self, dms.reshape(-1,nkpts,nao,nao), kpts, xctype)
        if with_j:
            coulG = pbc_tools.get_coulG(cell, Gv=Gv)
            coulomb_on_g_mesh = rho1[:,0].reshape(2, nset, ngrids).sum(axis=0) * coulG
        rho1 = ifft_in_place(rho1.reshape(-1, *mesh)).real.reshape(2, nset, -1, ngrids)
        wv = cp.einsum('anxg,axbyg->bnyg', rho1, fxc)
        wv = fft_in_place(wv.reshape(-1, *mesh)).reshape(wv.shape)

        if with_j:
            wv[:,:,0] += coulomb_on_g_mesh

        if 'GGA' in xctype:
            wv[:,:,0] -= contract('anxp,xp->anp', wv[:,:,1:4], Gv.T) * 1j
            if xctype == 'GGA':
                wv = cp.asarray(wv[:,:,0], order='C')
            elif xctype == 'MGGA':
                wv = cp.asarray(wv[:,:,[0, 4]], order='C')

        wv = wv.reshape(2*nset, -1, ngrids)

        with_tau = (xctype == 'MGGA')
        vmat = convert_xc_on_g_mesh_to_fock(self, wv, v_hermi, kpts, with_tau=with_tau)
        return vmat.reshape(dms.shape)

    def cache_xc_kernel1(self, cell, grids, xc_code, dm, spin=0, kpts=None, is_rhf=None):
        if isinstance(kpts, KPoints):
            raise NotImplementedError

        dms = _format_dms(dm, kpts)
        if is_rhf is None:
            is_rhf = len(dms) == 1
        elif is_rhf:
            assert len(dms) == 1
        else:
            assert spin == 1
            assert len(dms) == 2

        xctype = self._xc_type(xc_code)
        mesh = self.mesh
        ngrids = np.prod(mesh)
        rho = evaluate_density_on_g_mesh(self, dms, kpts, xctype)
        # Remove the grid weights. rho is scaled by the grid weights
        # (vol/ngrids) in evaluate_density_on_g_mesh
        rho *= ngrids / cell.vol
        rho = ifft_in_place(rho.reshape(-1, *mesh)).real.reshape(rho.shape)

        if is_rhf:
            if spin == 1:
                rho *= .5
                rho = cp.repeat(rho, 2, axis=0)
            else:
                rho = rho[0]

        vxc, fxc = self.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
        return rho, vxc, fxc

    cache_xc_kernel = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise RuntimeError('Not available')
