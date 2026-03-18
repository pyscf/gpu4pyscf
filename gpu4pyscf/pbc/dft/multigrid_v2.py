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

from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_gamma_point
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc.tools import pbc as pbc_tools
import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
from gpu4pyscf.lib.cupy_helper import contract, tag_array, load_library

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


def iG_density(density, cell):
    mesh = cell.mesh
    b = cp.asarray(cell.reciprocal_vectors())

    mesh_in_int32 = np.array(mesh, dtype=np.int32)
    block_dim = (8, 8, 8)
    grid_dim = tuple(np.array(np.ceil(mesh_in_int32 / np.array(block_dim)), dtype=np.int32))

    result = cp.zeros((3, *mesh), dtype=cp.complex128).reshape(3, -1)

    custom_kernel = cp.RawKernel(
        r"""
        #include <cupy/complex.cuh>
        extern "C" __global__
        void kernel(complex<double> * result,
                    const complex<double> * density,
                    const double * reciprocal_lattice,
                    const int mesh_a,
                    const int mesh_b,
                    const int mesh_c) {
            int a = blockIdx.x * blockDim.x + threadIdx.x;
            int b = blockIdx.y * blockDim.y + threadIdx.y;
            int c = blockIdx.z * blockDim.z + threadIdx.z;

            if(a >= mesh_a || b >= mesh_b || c >= mesh_c) {
                return;
            }

            const size_t grid_index = a * mesh_b * mesh_c + b * mesh_c + c;
            result += grid_index;
            const complex<double> density_value = density[grid_index] * complex<double>{0.0, 1.0};

            if(a >= (mesh_a + 1) / 2) a -= mesh_a;
            if(b >= (mesh_b + 1) / 2) b -= mesh_b;
            if(c >= (mesh_c + 1) / 2) c -= mesh_c;

            int n_grid = mesh_a * mesh_b * mesh_c;

            *result = (a * reciprocal_lattice[0] +
                       b * reciprocal_lattice[3] +
                       c * reciprocal_lattice[6]) * density_value;
            result += n_grid; 
            *result = (a * reciprocal_lattice[1] +
                       b * reciprocal_lattice[4] +
                       c * reciprocal_lattice[7]) * density_value;
            result += n_grid; 
            *result = (a * reciprocal_lattice[2] +
                       b * reciprocal_lattice[5] +
                       c * reciprocal_lattice[8]) * density_value;
        }
    """,
        'kernel',
    )

    custom_kernel(
        grid_dim,
        block_dim,
        (result, density, b, mesh_in_int32[0], mesh_in_int32[1], mesh_in_int32[2]),
    )

    return result


def contract_iG_potential(gga_potential, cell):
    mesh = cell.mesh
    b = cp.asarray(cell.reciprocal_vectors())

    mesh_in_int32 = np.array(mesh, dtype=np.int32)
    block_dim = (8, 8, 8)
    grid_dim = tuple(np.array(np.ceil(mesh_in_int32 / np.array(block_dim)), dtype=np.int32))

    custom_kernel = cp.RawKernel(
        r"""
        #include <cupy/complex.cuh>
        extern "C" __global__
        void kernel(complex<double> * gga_potential,
                    const double * reciprocal_lattice,
                    const int mesh_a,
                    const int mesh_b,
                    const int mesh_c) {
            int a = blockIdx.x * blockDim.x + threadIdx.x;
            int b = blockIdx.y * blockDim.y + threadIdx.y;
            int c = blockIdx.z * blockDim.z + threadIdx.z;

            if(a >= mesh_a || b >= mesh_b || c >= mesh_c) {
                return;
            }

            int n_grid = mesh_a * mesh_b * mesh_c;

            const size_t grid_index = a * mesh_b * mesh_c + b * mesh_c + c;
            gga_potential += grid_index;

            if(a >= (mesh_a + 1) / 2) a -= mesh_a;
            if(b >= (mesh_b + 1) / 2) b -= mesh_b;
            if(c >= (mesh_c + 1) / 2) c -= mesh_c;
            complex<double> potential_change = 0;
            potential_change += 
                      (a * reciprocal_lattice[0] +
                       b * reciprocal_lattice[3] +
                       c * reciprocal_lattice[6]) * 
                       complex<double>{0.0, 1.0} * 
                       gga_potential[n_grid];

            potential_change += 
                      (a * reciprocal_lattice[1] +
                       b * reciprocal_lattice[4] +
                       c * reciprocal_lattice[7]) * 
                       complex<double>{0.0, 1.0} * 
                       gga_potential[2 * n_grid];

            potential_change += 
                      (a * reciprocal_lattice[2] +
                       b * reciprocal_lattice[5] +
                       c * reciprocal_lattice[8]) * 
                       complex<double>{0.0, 1.0} * 
                       gga_potential[3 * n_grid];

            *gga_potential -= potential_change; 
        }
    """,
        'kernel',
    )

    custom_kernel(
        grid_dim,
        block_dim,
        (gga_potential, b, mesh_in_int32[0], mesh_in_int32[1], mesh_in_int32[2]),
    )


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


def unique_with_multiple_keys(x):
    # This function expands the previous function to handle multiple keys
    # shaped as [ (1, 2), (3, -4), ....]
    assert type(x) is cp.ndarray and (x.dtype == cp.int32 or x.dtype == cp.int64) and x.ndim == 2
    x = x.T
    n = x.shape[-1]

    inverse_sort = cp.zeros(n, dtype=cp.int64)
    if n <= 1:
        return x, inverse_sort

    sort_index = cp.lexsort(x)
    inverse_sort[sort_index] = cp.arange(0, n, dtype=cp.int64)
    x = x[:, sort_index].T

    mask = cp.empty(n, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = cp.any(x[1:] != x[:-1], axis=-1)

    x = x[mask]
    inverse_unique = cp.cumsum(mask, dtype=cp.int64) - 1

    return x, inverse_unique[inverse_sort]


def sort_contraction_coefficients(coeff):
    contraction_shapes = cp.array([i.shape for i in coeff])
    unique_shapes, inverse = unique_with_multiple_keys(contraction_shapes)
    unique_shapes = unique_shapes.get()
    inverse = inverse.get()

    sliced_axis = np.zeros((len(coeff) + 1, 2), dtype=np.int32)
    sliced_axis[1:] = np.cumsum(np.array([i.shape for i in coeff]), axis=0, dtype=np.int32)

    left_basis_function_indices = [cp.arange(begin, end) for begin, end in zip(sliced_axis[:-1, 0], sliced_axis[1:, 0])]
    right_basis_function_indices = [
        cp.arange(begin, end) for begin, end in zip(sliced_axis[:-1, 1], sliced_axis[1:, 1])
    ]

    sorted_coeffs = [{'shape': shape, 'coeffs': [], 'left_indices': [], 'right_indices': []} for shape in unique_shapes]

    for category, coeffs, left, right in zip(inverse, coeff, left_basis_function_indices, right_basis_function_indices):
        sorted_coeffs[category]['coeffs'].append(coeffs)
        sorted_coeffs[category]['left_indices'].append(left)
        sorted_coeffs[category]['right_indices'].append(right)

    for category in sorted_coeffs:
        category['coeffs'] = cp.array(category['coeffs'])
        category['left_indices'] = cp.concatenate(category['left_indices'])
        category['right_indices'] = cp.concatenate(category['right_indices'])

    return sorted_coeffs, sliced_axis[-1]


def contracted_to_primitive(batched_matrices, sorted_coeffs_left, sorted_coeffs_right, primitive_shape):
    assert len(batched_matrices.shape) == 3

    n_slices = batched_matrices.shape[0]
    n_cols = batched_matrices.shape[2]

    n_rows_primitive = primitive_shape[0]
    n_cols_primitive = primitive_shape[1]

    intermediate_shape = (n_slices, n_rows_primitive, n_cols)
    intermediate = cp.zeros(intermediate_shape, dtype=batched_matrices.dtype)

    for i in sorted_coeffs_left:
        subarray_shape = (n_slices, -1, i['shape'][1], n_cols)
        intermediate[:, i['left_indices']] = cp.einsum(
            'naij, api -> napj',
            batched_matrices[:, i['right_indices']].reshape(subarray_shape),
            i['coeffs'],
        ).reshape(n_slices, -1, n_cols)

    intermediate = intermediate.transpose(0, 2, 1)

    result_shape = (n_slices, n_cols_primitive, n_rows_primitive)
    result = cp.zeros(result_shape, dtype=batched_matrices.dtype)

    for i in sorted_coeffs_right:
        subarray_shape = (n_slices, -1, i['shape'][1], n_rows_primitive)
        result[:, i['left_indices']] = cp.einsum(
            'najp, aqj -> naqp',
            intermediate[:, i['right_indices']].reshape(subarray_shape),
            i['coeffs'],
        ).reshape(n_slices, -1, n_rows_primitive)

    return result.transpose(0, 2, 1)


def primitive_to_contracted(batched_matrices, sorted_coeffs_left, sorted_coeffs_right, contracted_shape):
    assert len(batched_matrices.shape) == 3

    n_slices = batched_matrices.shape[0]
    n_cols = batched_matrices.shape[2]

    n_rows_contracted = contracted_shape[0]
    n_cols_contracted = contracted_shape[1]

    intermediate_shape = (n_slices, n_rows_contracted, n_cols)
    intermediate = cp.zeros(intermediate_shape, dtype=batched_matrices.dtype)

    for i in sorted_coeffs_left:
        subarray_shape = (n_slices, -1, i['shape'][0], n_cols)
        intermediate[:, i['right_indices']] = cp.einsum(
            'napq, api -> naiq',
            batched_matrices[:, i['left_indices']].reshape(subarray_shape),
            i['coeffs'],
        ).reshape(n_slices, -1, n_cols)

    intermediate = intermediate.transpose(0, 2, 1)
    result_shape = (n_slices, n_cols_contracted, n_rows_contracted)
    result = cp.zeros(result_shape, dtype=batched_matrices.dtype)

    for i in sorted_coeffs_right:
        subarray_shape = (n_slices, -1, i['shape'][0], n_rows_contracted)
        result[:, i['right_indices']] = cp.einsum(
            'naqi, aqj -> naji',
            intermediate[:, i['left_indices']].reshape(subarray_shape),
            i['coeffs'],
        ).reshape(n_slices, -1, n_rows_contracted)

    return result.transpose(0, 2, 1)


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
    i_angular,
    j_angular,
    non_trivial_pairs,
    i_shells,
    j_shells,
    shell_to_ao_indices,
    image_indices,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    has_warned_instability,
    is_non_orthogonal,
    threshold,
    derivative_order,
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

    libgpbc.tailor_gaussian_pairs(
        cast_to_pointer(pairs_on_blocks),
        cast_to_pointer(n_pairs_on_blocks),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(non_trivial_pairs),
        cast_to_pointer(i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(len(j_shells)),
        cast_to_pointer(shell_to_ao_indices),
        cast_to_pointer(accumulated_n_pairs_per_block),
        cast_to_pointer(sorted_block_index),
        ctypes.c_int(n_contributing_blocks),
        cast_to_pointer(image_indices),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(len(vectors_to_neighboring_images)),
        cast_to_pointer(mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_int(is_non_orthogonal),
        ctypes.c_double(threshold),
        ctypes.c_int(derivative_order),
    )

    pairs_on_blocks = pairs_on_blocks[pairs_on_blocks >= 0]
    sorted_block_index = cp.asarray(cp.argsort(-n_pairs_on_blocks), dtype=cp.int32)
    n_contributing_blocks = cp.count_nonzero(n_pairs_on_blocks)
    accumulated_n_pairs_per_block[1:] = cp.cumsum(n_pairs_on_blocks, dtype=cp.int32)
    sorted_block_index = sorted_block_index[:n_contributing_blocks]

    return (
        pairs_on_blocks,
        accumulated_n_pairs_per_block,
        sorted_block_index,
        has_warned_instability
    )


def sort_gaussian_pairs(mydf, xc_type="LDA"):
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
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
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
            concatenated_coeff = coeff_in_localized
        else:
            subcell_in_diffused_region = grids_diffused.cell
            equivalent_cell_in_diffused, coeff_in_diffused = subcell_in_diffused_region.decontract_basis(to_cart=True)

            grouped_cell = equivalent_cell_in_localized + equivalent_cell_in_diffused

            grouped_cell._bas[n_primitive_gtos_in_localized:, 0] -= len(subcell_in_localized_region._atm)

            concatenated_coeff = coeff_in_localized + coeff_in_diffused

        coeff_in_localized, localized_shape = sort_contraction_coefficients(coeff_in_localized)

        concatenated_coeff, concatenated_shape = sort_contraction_coefficients(concatenated_coeff)

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

        concatenated_ao_indices = cp.concatenate((ao_indices_in_localized, ao_indices_in_diffused))
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
                    i_angular,
                    j_angular,
                    screened_shell_pairs,
                    i_shells,
                    j_shells,
                    shell_to_ao_indices,
                    image_indices,
                    vectors_to_neighboring_images,
                    mesh,
                    atm,
                    bas,
                    env,
                    has_warned_instability,
                    is_non_orthogonal,
                    cell.precision,
                    derivative_order,
                )
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
                'per_angular_pairs': per_angular_pairs,
                'neighboring_images': vectors_to_neighboring_images,
                'grouped_cell': grouped_cell,
                'mesh': mesh,  # this one is on cpu memory
                'fft_grid': fft_grid,
                'ao_indices_in_localized': ao_indices_in_localized,
                'ao_indices_in_diffused': ao_indices_in_diffused,
                'concatenated_ao_indices': concatenated_ao_indices,
                'coeff_in_localized': coeff_in_localized,
                'concatenated_coeff': concatenated_coeff,
                'primitive_shape': (localized_shape[0], concatenated_shape[0]),
                'contracted_shape': (localized_shape[1], concatenated_shape[1]),
                'atm': atm,
                'bas': bas,
                'env': env,
                'dxyz_dabc': dxyz_dabc,
                'is_non_orthogonal': is_non_orthogonal,
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

        density_matrix_with_rows_in_localized[:, :, :, n_ao_in_localized:] += (
            density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2).conj()
        )

        n_sets, n_k_points = density_matrix_with_rows_in_localized.shape[:2]

        density_matrix_with_rows_in_localized = density_matrix_with_rows_in_localized.reshape(
            -1, *pairs['contracted_shape']
        )

        coeff_sandwiched_density_matrix = contracted_to_primitive(
            density_matrix_with_rows_in_localized,
            pairs['coeff_in_localized'],
            pairs['concatenated_coeff'],
            pairs['primitive_shape'],
        ).reshape(n_sets, n_k_points, *pairs['primitive_shape'])

        libgpbc.update_dxyz_dabc(pairs['dxyz_dabc'].ctypes)

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
    if xc_type != 'LDA':
        for i in range(len(density_on_g_mesh)):
            density_on_g_mesh[i, 1:4] = iG_density(density_on_g_mesh[i], cell)

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
    n_i_functions, n_j_functions = pairs_info['primitive_shape']

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
        fock_slice = cp.zeros(
            (n_channels, n_difference_images, n_i_functions, n_j_functions),
            dtype=xc_weights.dtype,
        )
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        err = c_driver(
            cast_to_pointer(fock_slice),
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
        fock += fock_slice
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
        n_sets, n_k_points = fock_slice.shape[:2]

        fock_slice = fock_slice.reshape(-1, *pairs['primitive_shape'])

        fock_slice = primitive_to_contracted(
            fock_slice,
            pairs['coeff_in_localized'],
            pairs['concatenated_coeff'],
            pairs['contracted_shape'],
        ).reshape(n_sets, n_k_points, *pairs['contracted_shape'])

        # While mathematically it is correct to have concatenated
        # ao indices in the addition, but it is possible that the ao
        # indices overlap between localized gaussians and diffused gaussians
        # (imagine two gaussians within a single shell, say, C2s).
        # In this case, the addition to the same place requires atomic
        # operation, while I guess in the cupy code it is assumed that
        # the indices do not overlap, and hence no atomic guard.
        # Anyway, the numerical result will be wrong if we use
        # concatenated ao indices.
        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_localized"],
        ] += fock_slice[:, :, :, :n_ao_in_localized]
        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_diffused"],
        ] += fock_slice[:, :, :, n_ao_in_localized:]
        if hermi == 1:
            fock[
                :,
                :,
                pairs["ao_indices_in_diffused"][:, None],
                pairs["ao_indices_in_localized"],
            ] += (
                fock_slice[:, :, :, n_ao_in_localized:].transpose(0, 1, 3, 2).conj()
            )
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
        density_matrix_slice[:, :, :, n_ao_in_localized:] += density_matrix_with_rows_in_diffused.transpose(
            0, 1, 3, 2
        ).conj()

        n_sets, n_k_points = density_matrix_slice.shape[:2]

        density_matrix_slice = density_matrix_slice.reshape(-1, *pairs['contracted_shape'])

        coeff_sandwiched_density_matrix = contracted_to_primitive(
            density_matrix_slice,
            pairs['coeff_in_localized'],
            pairs['concatenated_coeff'],
            pairs['primitive_shape'],
        ).reshape(n_sets, n_k_points, *pairs['primitive_shape'])

        libgpbc.update_dxyz_dabc(pairs['dxyz_dabc'].ctypes)

        img_phase = image_phase_for_kpts(cell, pairs['neighboring_images'], kpts)
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
    vpplocG = ni.vpplocG
    vpp = convert_xc_on_g_mesh_to_fock(ni, vpplocG, hermi=1, kpts=kpts)[0]
    t1 = log.timer_debug1("vpploc", *t0)

    vppnl = pp_int.get_pp_nl(cell, kpts)
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
    Gv = pbc_tools._get_Gv(cell, mesh)
    # coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)
    coulomb_kernel_on_g_mesh = ni.coulG

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

    coulomb_on_g_mesh = rho_sf * ni.coulG
    coulomb_energy = 0.5 * rho_sf.conj().dot(coulomb_on_g_mesh).real
    coulomb_energy /= cell.vol
    log.debug('Multigrid Coulomb energy %s', coulomb_energy)
    t0 = log.timer('coulomb', *t0)
    weight = cell.vol / ngrids

    density = ifft_in_place(density.reshape(-1, *mesh)).real.reshape(-1, ngrids)
    n_electrons = float(density[0].sum().real.get())
    density /= weight

    # eval_xc_eff supports float64 only
    density = cp.asarray(density, dtype=np.float64, order='C')
    if xc_type == "LDA" or xc_type == 'HF':
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density[0], deriv=1, xctype=xc_type
        )[:2]
    elif xc_type == 'GGA' or xc_type == 'MGGA':
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type
        )[:2]
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    rho_sf = density[0].real
    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock *= weight
    xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(-1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == 'GGA':
        contract_iG_potential(xc_for_fock, cell)
        xc_for_fock = xc_for_fock[0]
        xc_for_fock = xc_for_fock.reshape((-1, ngrids))
    elif xc_type == 'MGGA':
        xc_for_fock = contract_iG_potential(xc_for_fock, cell)
        xc_for_fock = cp.concatenate(
            [
                xc_for_fock[0].reshape((-1, ngrids)),
                xc_for_fock[4].reshape((-1, ngrids)),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[0] += coulomb_on_g_mesh

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    veff = convert_xc_on_g_mesh_to_fock(ni, xc_for_fock, hermi, kpts_band, with_tau = (xc_type == "MGGA"))
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(veff, ecoul=coulomb_energy, exc=xc_energy_sum, vj=None, vk=None)
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

    coulomb_on_g_mesh = rho_sf * ni.coulG
    coulomb_energy = 0.5 * rho_sf.conj().dot(coulomb_on_g_mesh).real
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
    if xc_type == "LDA" or xc_type == 'HF':
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density[:,0], deriv=1, xctype=xc_type
        )[:2]
    elif xc_type == 'GGA' or xc_type == 'MGGA':
        xc_for_energy, xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type
        )[:2]
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    rho_sf = (density[0, 0] + density[1, 0]).real
    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).real.get()) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock *= weight
    xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(nset, -1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == 'GGA':
        for i in xc_for_fock:
            contract_iG_potential(i, cell)
        xc_for_fock = xc_for_fock.reshape((nset, -1, ngrids))
    elif xc_type == 'MGGA':
        for i in xc_for_fock:
            contract_iG_potential(i, cell)
        xc_for_fock = cp.concatenate(
            [
                xc_for_fock[:, 0].reshape((nset, -1, ngrids)),
                xc_for_fock[:, 4].reshape((nset, -1, ngrids)),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    veff = convert_xc_on_g_mesh_to_fock(ni, xc_for_fock, hermi, kpts_band, with_tau = (xc_type == "MGGA"))
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(veff, ecoul=coulomb_energy, exc=xc_energy_sum, vj=None, vk=None)
    t0 = log.timer("xc", *t0)
    return n_electrons, xc_energy_sum, veff

def get_rho(ni, dm, kpts=None):
    '''Density in real space
    '''
    cell = ni.cell
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    density = evaluate_density_on_g_mesh(ni, dm, kpts)
    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = ifft_in_place(density.reshape(-1, *mesh)).real / weight
    return rhoR.reshape(-1, ngrids)

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

    Gv = pbc_tools._get_Gv(cell, mesh)
    coulomb_on_g_mesh = cp.einsum('ng, g -> g', density[:, 0], ni.coulG)

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

    if nset == 1:
        xc_for_fock = ni.eval_xc_eff(
            xc_code, density[0], deriv=1, xctype=xc_type
        )[1]
    else:
        xc_for_fock = ni.eval_xc_eff(
            xc_code, density, deriv=1, xctype=xc_type
        )[1]

    xc_for_fock = xc_for_fock.reshape(nset, -1, *mesh) * weight
    xc_for_fock = fft_in_place(xc_for_fock).reshape(nset, -1, ngrids)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == 'GGA':
        for i in xc_for_fock:
            contract_iG_potential(i, cell)
        xc_for_fock = xc_for_fock[:, 0]
        xc_for_fock = xc_for_fock.reshape((nset, -1, ngrids))
    elif xc_type == 'MGGA':
        xc_for_fock[:, 0] -= contract('ngp, pg -> np', xc_for_fock[:, 1:4], Gv) * 1j
        xc_for_fock = cp.concatenate(
            [
                xc_for_fock[:, 0].reshape((nset, -1, ngrids)),
                xc_for_fock[:, 4].reshape((nset, -1, ngrids)),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    if with_j:
        xc_for_fock[:, 0] += coulomb_on_g_mesh

    if with_pseudo_vloc_orbital_derivative:
        xc_for_fock[:, 0] += ni.vpplocG

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
        Gv = pbc_tools._get_Gv(cell, cell.mesh)
        self.coulG = pbc_tools.get_coulG(cell, Gv=Gv)
        if cell._pseudo:
            self.vpplocG = multigrid_v1.eval_vpplocG(cell, cell.mesh)
        else:
            self.vpplocG = multigrid_v1.eval_nucG(cell, cell.mesh)

        self.build()

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
    get_vxc = nr_vxc = NotImplemented #numint_cpu.KNumInt.nr_vxc

    eval_xc_eff = numint.eval_xc_eff
    _init_xcfuns = numint.NumInt._init_xcfuns

    nr_rks_fxc = NotImplemented
    nr_uks_fxc = NotImplemented
    nr_rks_fxc_st = NotImplemented
    cache_xc_kernel  = NotImplemented
    cache_xc_kernel1 = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise RuntimeError('Not available')
