/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cub/cub.cuh>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>

#include "cartesian.cuh"
#include "constant_objects.cuh"
#include "utils.cuh"

#define BLOCK_DIM_XYZ 4

namespace gpu4pyscf::gpbc::multi_grid::gradient {

template <typename KernelType, int n_channels, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ void evaluate_xc_kernel(
    KernelType *gradient, const KernelType *xc_weights,
    const KernelType *density_matrices, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_ij = n_i_cartesian_functions * n_j_cartesian_functions;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_xy_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_dimensions = 3;

  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride =
      density_matrix_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];

  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const KernelType a_dot_b = dxyz_dabc[0] * dxyz_dabc[3]
                           + dxyz_dabc[1] * dxyz_dabc[4]
                           + dxyz_dabc[2] * dxyz_dabc[5];
  const KernelType a_dot_c = dxyz_dabc[0] * dxyz_dabc[6]
                           + dxyz_dabc[1] * dxyz_dabc[7]
                           + dxyz_dabc[2] * dxyz_dabc[8];
  const KernelType b_dot_c = dxyz_dabc[3] * dxyz_dabc[6]
                           + dxyz_dabc[4] * dxyz_dabc[7]
                           + dxyz_dabc[5] * dxyz_dabc[8];

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  KernelType i_atom_gradient[n_dimensions];
  KernelType j_atom_gradient[n_dimensions];

  KernelType
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  __shared__ KernelType xc_values[n_channels * n_threads];

  int a_index = a_start + threadIdx.z;
  int b_index = b_start + threadIdx.y;
  int c_index = c_start + threadIdx.x;

  const bool out_of_boundary =
      a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c;
  KernelType xc_value = 0;

  const int thread_id =
      threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ + threadIdx.z * n_xy_threads;

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    if (!out_of_boundary) {
      xc_value =
          xc_weights[i_channel * xc_weights_stride + a_index * mesh_b * mesh_c +
                     b_index * mesh_c + c_index];
    }

    xc_values[i_channel * n_threads + thread_id] = xc_value;
  }
  __syncthreads();

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;
    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];
    const int i_atom = bas(ATOM_OF, i_shell);
    const int j_atom = bas(ATOM_OF, j_shell);

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, i_atom);
    const KernelType i_x = env[i_coord_offset] +
                           vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, j_atom);
    const KernelType j_x = env[j_coord_offset] +
                           vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;

#pragma unroll
    for (int xyz_index = 0; xyz_index < n_dimensions; xyz_index++) {
      i_atom_gradient[xyz_index] = 0;
      j_atom_gradient[xyz_index] = 0;
    }

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          const KernelType density_matrix_value =
              density_matrices[density_matrix_channel_stride * i_channel +
                               image_difference_index * density_matrix_stride +
                               (i_function + i_function_index) * n_j_functions +
                               j_function + j_function_index];

          prefactor[i_channel * n_i_cartesian_functions *
                        n_j_cartesian_functions +
                    i_function_index * n_j_cartesian_functions +
                    j_function_index] = pair_prefactor * density_matrix_value;
        }
      }
    }
    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    const KernelType exp_dadb = exp(-2 * ij_exponent * a_dot_b);
    const KernelType exp_dadc = exp(-2 * ij_exponent * a_dot_c);
    const KernelType exp_dbdc = exp(-2 * ij_exponent * b_dot_c);

    KernelType i_cartesian[n_i_cartesian_functions];
    KernelType j_cartesian[n_j_cartesian_functions];
    KernelType i_cartesian_gradient[n_dimensions * n_i_cartesian_functions];
    KernelType j_cartesian_gradient[n_dimensions * n_j_cartesian_functions];
    KernelType x, y, z;
    KernelType gaussian_x, gaussian_y, gaussian_z,
               recursion_factor_a, recursion_factor_b, recursion_factor_c;
    KernelType recursion_factor_ab_pow_a = 1;
    KernelType recursion_factor_ac_pow_a = 1;
    KernelType recursion_factor_bc_pow_b = 1;

    if constexpr (is_non_orthogonal) {
      // recursion_factor_ab_pow_a = 1;
      // recursion_factor_ac_pow_a = 1;
    } else {
      x = start_position_x;
    }
    for (a_index = 0, gaussian_x = 1,
         recursion_factor_a = recursion_factor_a_start;
         a_index < a_upper;
         a_index++, gaussian_x *= recursion_factor_a,
         recursion_factor_a *= exp_da_squared) {

      if constexpr (is_non_orthogonal) {
        recursion_factor_bc_pow_b = 1;
      } else {
        y = start_position_y;
      }
      for (b_index = 0, gaussian_y = 1,
           recursion_factor_b = recursion_factor_b_start;
           b_index < b_upper;
           b_index++, gaussian_y *= recursion_factor_b * recursion_factor_ab_pow_a,
           recursion_factor_b *= exp_db_squared) {

        if constexpr (is_non_orthogonal) {
          x = start_position_x + a_index * dxyz_dabc[0] + b_index * dxyz_dabc[3];
          y = start_position_y + a_index * dxyz_dabc[1] + b_index * dxyz_dabc[4];
          z = start_position_z + a_index * dxyz_dabc[2] + b_index * dxyz_dabc[5];
        } else {
          z = start_position_z;
        }
        for (c_index = 0, gaussian_z = 1,
             recursion_factor_c = recursion_factor_c_start;
             c_index < c_upper;
             c_index++, gaussian_z *= recursion_factor_c
                                    * recursion_factor_ac_pow_a
                                    * recursion_factor_bc_pow_b,
             recursion_factor_c *= exp_dc_squared) {
          multi_grid::gto_cartesian<KernelType, i_angular>(i_cartesian,
                                                           x - i_x, y - i_y, z - i_z);
          multi_grid::gto_cartesian<KernelType, j_angular>(j_cartesian,
                                                           x - j_x, y - j_y, z - j_z);
          gradient::gto_cartesian<KernelType, i_angular>(
              i_cartesian_gradient, i_cartesian, x - i_x, y - i_y, z - i_z,
              i_exponent);
          gradient::gto_cartesian<KernelType, j_angular>(
              j_cartesian_gradient, j_cartesian, x - j_x, y - j_y, z - j_z,
              j_exponent);

          const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            xc_value =
                gaussian *
                xc_values[i_channel * n_threads + a_index * n_xy_threads +
                          b_index * BLOCK_DIM_XYZ + c_index];
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
#pragma unroll
                for (int xyz_index = 0; xyz_index < n_dimensions; xyz_index++) {
                  i_atom_gradient[xyz_index] -=
                      xc_value *
                      i_cartesian_gradient[xyz_index * n_i_cartesian_functions +
                                           i_function_index] *
                      j_cartesian[j_function_index] *
                      prefactor[i_channel * n_ij +
                                i_function_index * n_j_cartesian_functions +
                                j_function_index];

                  j_atom_gradient[xyz_index] -=
                      xc_value *
                      j_cartesian_gradient[xyz_index * n_j_cartesian_functions +
                                           j_function_index] *
                      i_cartesian[i_function_index] *
                      prefactor[i_channel * n_ij +
                                i_function_index * n_j_cartesian_functions +
                                j_function_index];
                }
              }
            }
          }

          if constexpr (is_non_orthogonal) {
            x += dxyz_dabc[6];
            y += dxyz_dabc[7];
            z += dxyz_dabc[8];
          } else {
            z += dxyz_dabc[8];
          }
        }

        if constexpr (is_non_orthogonal) {
          recursion_factor_bc_pow_b *= exp_dbdc;
        } else {
          y += dxyz_dabc[4];
        }
      }

      if constexpr (is_non_orthogonal) {
        recursion_factor_ab_pow_a *= exp_dadb;
        recursion_factor_ac_pow_a *= exp_dadc;
      } else {
        x += dxyz_dabc[0];
      }
    }

    if (is_valid_pair) {
#pragma unroll
      for (int xyz_index = 0; xyz_index < n_dimensions; xyz_index++) {
        atomicAdd(gradient + n_dimensions * i_atom + xyz_index,
                  i_atom_gradient[xyz_index]);
        atomicAdd(gradient + n_dimensions * j_atom + xyz_index,
                  j_atom_gradient[xyz_index]);
      }
    }
  }
}

#define xc_gradient_kernel_macro(li, lj)                                       \
  evaluate_xc_kernel<KernelType, n_channels, li, lj, is_non_orthogonal>        \
      <<<block_grid, block_size>>>(                                            \
          gradient, xc_weights, density_matrices, non_trivial_pairs, i_shells, \
          j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,            \
          n_j_functions, sorted_pairs_per_local_grid,                          \
          accumulated_n_pairs_per_local_grid, sorted_block_index,              \
          image_indices, vectors_to_neighboring_images, n_images,              \
          image_pair_difference_index, n_difference_images, mesh_a, mesh_b,    \
          mesh_c, atm, bas, env)

#define xc_gradient_kernel_case_macro(li, lj)                                  \
  case (li * 10 + lj):                                                         \
    xc_gradient_kernel_macro(li, lj);                                          \
    break

template <typename KernelType, int n_channels, bool is_non_orthogonal>
int evaluate_xc_driver(
    KernelType *gradient, const KernelType *xc_weights,
    const KernelType *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);

  switch (i_angular * 10 + j_angular) {
    xc_gradient_kernel_case_macro(0, 0);
    xc_gradient_kernel_case_macro(0, 1);
    xc_gradient_kernel_case_macro(0, 2);
    xc_gradient_kernel_case_macro(0, 3);
    xc_gradient_kernel_case_macro(0, 4);
    xc_gradient_kernel_case_macro(1, 0);
    xc_gradient_kernel_case_macro(1, 1);
    xc_gradient_kernel_case_macro(1, 2);
    xc_gradient_kernel_case_macro(1, 3);
    xc_gradient_kernel_case_macro(1, 4);
    xc_gradient_kernel_case_macro(2, 0);
    xc_gradient_kernel_case_macro(2, 1);
    xc_gradient_kernel_case_macro(2, 2);
    xc_gradient_kernel_case_macro(2, 3);
    xc_gradient_kernel_case_macro(2, 4);
    xc_gradient_kernel_case_macro(3, 0);
    xc_gradient_kernel_case_macro(3, 1);
    xc_gradient_kernel_case_macro(3, 2);
    xc_gradient_kernel_case_macro(3, 3);
    xc_gradient_kernel_case_macro(3, 4);
    xc_gradient_kernel_case_macro(4, 0);
    xc_gradient_kernel_case_macro(4, 1);
    xc_gradient_kernel_case_macro(4, 2);
    xc_gradient_kernel_case_macro(4, 3);
    xc_gradient_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_xc_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

template <typename KernelType, int n_channels, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ void evaluate_xc_with_tau_kernel(
    KernelType *gradient, const KernelType *xc_weights,
    const KernelType *density_matrices, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *image_pair_difference_index, const int n_difference_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_ij = n_i_cartesian_functions * n_j_cartesian_functions;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_xy_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  constexpr int n_dimensions = 3;

  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride =
      density_matrix_stride * n_difference_images;

  const int block_index = sorted_block_index[blockIdx.x];

  const int n_blocks_b = (mesh_b + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;
  const int n_blocks_c = (mesh_c + BLOCK_DIM_XYZ - 1) / BLOCK_DIM_XYZ;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const KernelType start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const KernelType start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const KernelType start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const KernelType a_dot_b = dxyz_dabc[0] * dxyz_dabc[3]
                           + dxyz_dabc[1] * dxyz_dabc[4]
                           + dxyz_dabc[2] * dxyz_dabc[5];
  const KernelType a_dot_c = dxyz_dabc[0] * dxyz_dabc[6]
                           + dxyz_dabc[1] * dxyz_dabc[7]
                           + dxyz_dabc[2] * dxyz_dabc[8];
  const KernelType b_dot_c = dxyz_dabc[3] * dxyz_dabc[6]
                           + dxyz_dabc[4] * dxyz_dabc[7]
                           + dxyz_dabc[5] * dxyz_dabc[8];

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  KernelType i_atom_gradient[n_dimensions];
  KernelType j_atom_gradient[n_dimensions];

  KernelType
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  __shared__ KernelType xc_values[n_channels * 2 * n_threads];

  int a_index = a_start + threadIdx.z;
  int b_index = b_start + threadIdx.y;
  int c_index = c_start + threadIdx.x;

  const bool out_of_boundary =
      a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c;

  const int thread_id =
      threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ + threadIdx.z * n_xy_threads;

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    KernelType xc_rho_value = 0;
    KernelType xc_tau_value = 0;
    if (!out_of_boundary) {
      xc_rho_value =
          xc_weights[(i_channel * 2 + 0) * xc_weights_stride
                     + a_index * mesh_b * mesh_c + b_index * mesh_c + c_index];
      xc_tau_value =
          xc_weights[(i_channel * 2 + 1) * xc_weights_stride
                     + a_index * mesh_b * mesh_c + b_index * mesh_c + c_index];
    }

    xc_values[(i_channel * 2 + 0) * n_threads + thread_id] = xc_rho_value;
    xc_values[(i_channel * 2 + 1) * n_threads + thread_id] = xc_tau_value;
  }
  __syncthreads();

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;
    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;
    const int image_difference_index = image_pair_difference_index[image_index];

    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];
    const int i_atom = bas(ATOM_OF, i_shell);
    const int j_atom = bas(ATOM_OF, j_shell);

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, i_atom);
    const KernelType i_x = env[i_coord_offset] +
                           vectors_to_neighboring_images[image_index_i * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, j_atom);
    const KernelType j_x = env[j_coord_offset] +
                           vectors_to_neighboring_images[image_index_j * 3];
    const KernelType j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
    const KernelType j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType ij_exponent = i_exponent + j_exponent;
    const KernelType ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const KernelType x0 = start_position_x - pair_x;
    const KernelType y0 = start_position_y - pair_y;
    const KernelType z0 = start_position_z - pair_z;

    const KernelType gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const KernelType pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<KernelType, i_angular>() *
                  common_fac_sp<KernelType, j_angular>()
            : 0;

#pragma unroll
    for (int xyz_index = 0; xyz_index < n_dimensions; xyz_index++) {
      i_atom_gradient[xyz_index] = 0;
      j_atom_gradient[xyz_index] = 0;
    }

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          const KernelType density_matrix_value =
              density_matrices[density_matrix_channel_stride * i_channel +
                               image_difference_index * density_matrix_stride +
                               (i_function + i_function_index) * n_j_functions +
                               j_function + j_function_index];

          prefactor[i_channel * n_i_cartesian_functions * n_j_cartesian_functions +
                    i_function_index * n_j_cartesian_functions +
                    j_function_index] = pair_prefactor * density_matrix_value;
        }
      }
    }
    const KernelType da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const KernelType db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const KernelType dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const KernelType exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const KernelType exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const KernelType exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const KernelType cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const KernelType cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const KernelType cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    const KernelType recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    const KernelType exp_dadb = exp(-2 * ij_exponent * a_dot_b);
    const KernelType exp_dadc = exp(-2 * ij_exponent * a_dot_c);
    const KernelType exp_dbdc = exp(-2 * ij_exponent * b_dot_c);

    KernelType i_cartesian[n_i_cartesian_functions];
    KernelType j_cartesian[n_j_cartesian_functions];
    KernelType i_cartesian_gradient[n_dimensions * n_i_cartesian_functions];
    KernelType j_cartesian_gradient[n_dimensions * n_j_cartesian_functions];
    KernelType i_cartesian_second_derivative[((n_dimensions + 1) * n_dimensions / 2) * n_i_cartesian_functions];
    KernelType j_cartesian_second_derivative[((n_dimensions + 1) * n_dimensions / 2) * n_j_cartesian_functions];
    KernelType x, y, z;
    KernelType gaussian_x, gaussian_y, gaussian_z,
               recursion_factor_a, recursion_factor_b, recursion_factor_c;
    KernelType recursion_factor_ab_pow_a = 1;
    KernelType recursion_factor_ac_pow_a = 1;
    KernelType recursion_factor_bc_pow_b = 1;

    if constexpr (is_non_orthogonal) {
      // recursion_factor_ab_pow_a = 1;
      // recursion_factor_ac_pow_a = 1;
    } else {
      x = start_position_x;
    }
    for (a_index = 0, gaussian_x = 1,
         recursion_factor_a = recursion_factor_a_start;
         a_index < a_upper;
         a_index++, gaussian_x *= recursion_factor_a,
         recursion_factor_a *= exp_da_squared) {

      if constexpr (is_non_orthogonal) {
        recursion_factor_bc_pow_b = 1;
      } else {
        y = start_position_y;
      }
      for (b_index = 0, gaussian_y = 1,
           recursion_factor_b = recursion_factor_b_start;
           b_index < b_upper;
           b_index++, gaussian_y *= recursion_factor_b * recursion_factor_ab_pow_a,
           recursion_factor_b *= exp_db_squared) {

        if constexpr (is_non_orthogonal) {
          x = start_position_x + a_index * dxyz_dabc[0] + b_index * dxyz_dabc[3];
          y = start_position_y + a_index * dxyz_dabc[1] + b_index * dxyz_dabc[4];
          z = start_position_z + a_index * dxyz_dabc[2] + b_index * dxyz_dabc[5];
        } else {
          z = start_position_z;
        }
        for (c_index = 0, gaussian_z = 1,
             recursion_factor_c = recursion_factor_c_start;
             c_index < c_upper;
             c_index++, gaussian_z *= recursion_factor_c
                                    * recursion_factor_ac_pow_a
                                    * recursion_factor_bc_pow_b,
             recursion_factor_c *= exp_dc_squared) {
          multi_grid::gto_cartesian<KernelType, i_angular>(i_cartesian,
                                                           x - i_x, y - i_y, z - i_z);
          multi_grid::gto_cartesian<KernelType, j_angular>(j_cartesian,
                                                           x - j_x, y - j_y, z - j_z);
          gradient::gto_cartesian<KernelType, i_angular>(
              i_cartesian_gradient, i_cartesian, x - i_x, y - i_y, z - i_z,
              i_exponent);
          gradient::gto_cartesian<KernelType, j_angular>(
              j_cartesian_gradient, j_cartesian, x - j_x, y - j_y, z - j_z,
              j_exponent);
          second_derivative::gto_cartesian<KernelType, i_angular>(
            i_cartesian_second_derivative, x - i_x, y - i_y, z - i_z, i_exponent);
          second_derivative::gto_cartesian<KernelType, j_angular>(
            j_cartesian_second_derivative, x - j_x, y - j_y, z - j_z, j_exponent);

          const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            const KernelType xc_rho_value =
                gaussian *
                xc_values[(i_channel * 2 + 0) * n_threads
                          + a_index * n_xy_threads
                          + b_index * BLOCK_DIM_XYZ + c_index];
            const KernelType xc_tau_value =
                gaussian *
                xc_values[(i_channel * 2 + 1) * n_threads
                          + a_index * n_xy_threads
                          + b_index * BLOCK_DIM_XYZ + c_index] / 2;

#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                const KernelType prefactor_ij = prefactor[
                  i_channel * n_ij
                  + i_function_index * n_j_cartesian_functions + j_function_index];

                i_atom_gradient[0] -= prefactor_ij * (
                  xc_rho_value
                  * i_cartesian_gradient[0 * n_i_cartesian_functions + i_function_index]
                  * j_cartesian[j_function_index]
                  + xc_tau_value * (
                      i_cartesian_second_derivative[0 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[0 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[1 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[1 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[2 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[2 * n_j_cartesian_functions + j_function_index]
                  )
                );
                j_atom_gradient[0] -= prefactor_ij * (
                  xc_rho_value
                  * j_cartesian_gradient[0 * n_j_cartesian_functions + j_function_index]
                  * i_cartesian[i_function_index]
                  + xc_tau_value * (
                      j_cartesian_second_derivative[0 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[0 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[1 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[1 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[2 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[2 * n_i_cartesian_functions + i_function_index]
                  )
                );
                i_atom_gradient[1] -= prefactor_ij * (
                  xc_rho_value
                  * i_cartesian_gradient[1 * n_i_cartesian_functions + i_function_index]
                  * j_cartesian[j_function_index]
                  + xc_tau_value * (
                      i_cartesian_second_derivative[1 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[0 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[3 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[1 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[4 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[2 * n_j_cartesian_functions + j_function_index]
                  )
                );
                j_atom_gradient[1] -= prefactor_ij * (
                  xc_rho_value
                  * j_cartesian_gradient[1 * n_j_cartesian_functions + j_function_index]
                  * i_cartesian[i_function_index]
                  + xc_tau_value * (
                      j_cartesian_second_derivative[1 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[0 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[3 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[1 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[4 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[2 * n_i_cartesian_functions + i_function_index]
                  )
                );
                i_atom_gradient[2] -= prefactor_ij * (
                  xc_rho_value
                  * i_cartesian_gradient[2 * n_i_cartesian_functions + i_function_index]
                  * j_cartesian[j_function_index]
                  + xc_tau_value * (
                      i_cartesian_second_derivative[2 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[0 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[4 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[1 * n_j_cartesian_functions + j_function_index]
                    + i_cartesian_second_derivative[5 * n_i_cartesian_functions + i_function_index]
                    * j_cartesian_gradient[2 * n_j_cartesian_functions + j_function_index]
                  )
                );
                j_atom_gradient[2] -= prefactor_ij * (
                  xc_rho_value
                  * j_cartesian_gradient[2 * n_j_cartesian_functions + j_function_index]
                  * i_cartesian[i_function_index]
                  + xc_tau_value * (
                      j_cartesian_second_derivative[2 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[0 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[4 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[1 * n_i_cartesian_functions + i_function_index]
                    + j_cartesian_second_derivative[5 * n_j_cartesian_functions + j_function_index]
                    * i_cartesian_gradient[2 * n_i_cartesian_functions + i_function_index]
                  )
                );
              }
            }
          }

          if constexpr (is_non_orthogonal) {
            x += dxyz_dabc[6];
            y += dxyz_dabc[7];
            z += dxyz_dabc[8];
          } else {
            z += dxyz_dabc[8];
          }
        }

        if constexpr (is_non_orthogonal) {
          recursion_factor_bc_pow_b *= exp_dbdc;
        } else {
          y += dxyz_dabc[4];
        }
      }

      if constexpr (is_non_orthogonal) {
        recursion_factor_ab_pow_a *= exp_dadb;
        recursion_factor_ac_pow_a *= exp_dadc;
      } else {
        x += dxyz_dabc[0];
      }
    }

    if (is_valid_pair) {
#pragma unroll
      for (int xyz_index = 0; xyz_index < n_dimensions; xyz_index++) {
        atomicAdd(gradient + n_dimensions * i_atom + xyz_index,
                  i_atom_gradient[xyz_index]);
        atomicAdd(gradient + n_dimensions * j_atom + xyz_index,
                  j_atom_gradient[xyz_index]);
      }
    }
  }
}

#define xc_with_tau_gradient_kernel_macro(li, lj)                                \
  evaluate_xc_with_tau_kernel<KernelType, n_channels, li, lj, is_non_orthogonal> \
      <<<block_grid, block_size>>>(                                              \
          gradient, xc_weights, density_matrices, non_trivial_pairs, i_shells,   \
          j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,              \
          n_j_functions, sorted_pairs_per_local_grid,                            \
          accumulated_n_pairs_per_local_grid, sorted_block_index,                \
          image_indices, vectors_to_neighboring_images, n_images,                \
          image_pair_difference_index, n_difference_images, mesh_a, mesh_b,      \
          mesh_c, atm, bas, env)

#define xc_with_tau_gradient_kernel_case_macro(li, lj)                           \
  case (li * 10 + lj):                                                           \
    xc_with_tau_gradient_kernel_macro(li, lj);                                   \
    break

template <typename KernelType, int n_channels, bool is_non_orthogonal>
int evaluate_xc_with_tau_driver(
    KernelType *gradient, const KernelType *xc_weights,
    const KernelType *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);

  switch (i_angular * 10 + j_angular) {
    xc_with_tau_gradient_kernel_case_macro(0, 0);
    xc_with_tau_gradient_kernel_case_macro(0, 1);
    xc_with_tau_gradient_kernel_case_macro(0, 2);
    xc_with_tau_gradient_kernel_case_macro(0, 3);
    xc_with_tau_gradient_kernel_case_macro(0, 4);
    xc_with_tau_gradient_kernel_case_macro(1, 0);
    xc_with_tau_gradient_kernel_case_macro(1, 1);
    xc_with_tau_gradient_kernel_case_macro(1, 2);
    xc_with_tau_gradient_kernel_case_macro(1, 3);
    xc_with_tau_gradient_kernel_case_macro(1, 4);
    xc_with_tau_gradient_kernel_case_macro(2, 0);
    xc_with_tau_gradient_kernel_case_macro(2, 1);
    xc_with_tau_gradient_kernel_case_macro(2, 2);
    xc_with_tau_gradient_kernel_case_macro(2, 3);
    xc_with_tau_gradient_kernel_case_macro(2, 4);
    xc_with_tau_gradient_kernel_case_macro(3, 0);
    xc_with_tau_gradient_kernel_case_macro(3, 1);
    xc_with_tau_gradient_kernel_case_macro(3, 2);
    xc_with_tau_gradient_kernel_case_macro(3, 3);
    xc_with_tau_gradient_kernel_case_macro(3, 4);
    xc_with_tau_gradient_kernel_case_macro(4, 0);
    xc_with_tau_gradient_kernel_case_macro(4, 1);
    xc_with_tau_gradient_kernel_case_macro(4, 2);
    xc_with_tau_gradient_kernel_case_macro(4, 3);
    xc_with_tau_gradient_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_xc_with_tau_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

} // namespace gpu4pyscf::gpbc::multi_grid::gradient
