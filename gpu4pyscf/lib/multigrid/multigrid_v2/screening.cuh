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

#include "constant_objects.cuh"
#include "utils.cuh"

#define EIJ_CUTOFF 60
#define BLOCK_DIM_XYZ 4
#define EXP_OVERFLOW 400 

namespace gpu4pyscf::gpbc::multi_grid {

template <int i_angular, int j_angular>
__device__ double
gaussian_pair_cutoff(const double i_exponent, const double j_exponent,
                     const double i_coefficient, const double j_coefficient,
                     const double pair_distance, const double cell_volume,
                     const double precision) {
  constexpr int pair_angular = i_angular + j_angular;
  constexpr double i_norm_constant_part = (2 * i_angular + 1) / (4 * M_PI);
  constexpr double j_norm_constant_part = (2 * j_angular + 1) / (4 * M_PI);

  const double pair_exponent = i_exponent + j_exponent;
  const double fi = i_exponent / pair_exponent;
  const double fj = j_exponent / pair_exponent;
  const double theta = i_exponent * fj;
  const double dri = fj * pair_distance;
  const double drj = fi * pair_distance;
  const double fac_dri =
      pow(i_angular * .5 / pair_exponent + dri * dri, i_angular * 0.5);
  const double fac_drj =
      pow(j_angular * .5 / pair_exponent + drj * drj, j_angular * 0.5);
  const double rad = pow(cell_volume, -1. / 3) * pair_distance + 1;
  double surface = 4 * M_PI * rad * rad;
  if (surface < 1) {
    surface = 1;
  }
  const double i_norm =
      abs(i_coefficient) * sqrt(i_norm_constant_part) * 2 * i_exponent;
  const double j_norm =
      abs(j_coefficient) * sqrt(j_norm_constant_part) * 2 * j_exponent;
  const double prefactor = i_norm * j_norm * pow(M_PI / pair_exponent, 1.5);
  double overlap = prefactor * exp(-theta * pair_distance * pair_distance) *
                   fac_dri * fac_drj * surface;
  if (overlap > 1) {
    overlap = 1;
  }
  const double factor = overlap / precision;

  double radius = 2;
  radius =
      sqrt(log(factor * pow(radius, pair_angular + 1) + 1) / pair_exponent);

  // radius =
  //     sqrt(log(factor * pow(radius, pair_angular + 1) + 1) / pair_exponent);
  return radius;
}

template <int angular>
__device__ double gaussian_summation_cutoff(const double exponent,
                                            const double prefactor_in_log,
                                            const double threshold_in_log) {
  // rho[r-Rp] = ci*cj * exp(-theta*(ri-rj)**2) * r**lij * exp(-aij*r**2)
  //           ~= ovlp * r**lij * exp(-aij*r**2)
  // log(ovlp) ~= log(ci*cj) - theta*(ri-rj)**2
  //           ~= log_cicj + prefactor_in_log
  // radius can be solved using fixed iteration
  // radius = (log(ovlp/precision * radius**(lij+l_inc)) / aij)**.5
  // where l_inc = 0 (LDA), 1 (GGA), 2 (MGGA)
  constexpr double log_r = 2.302585092994046;                // log(10)
  const double log_of_doubled_exponents = log(2 * exponent); // for derivative
  const double log_aij = log(exponent) * 1.5;
  constexpr int l_inc = 1; // TODO: input l_inc for LDA or Coulomb potential
  // approximate log(ci * cj) for primitive Gaussians functions |i> and |j>
  // ci ~= sqrt((2*ai)^((li+3)/2) * \Gamma((li+3)/2) * (2li+1)/4pi)
  //    ~ (2*ai)^((li+3)/4) * ~1
  // approximate log(ci * cj) by the larger normalization coefficient ~= log(ci)
  // TODO: consider the basis contraction coefficients
  double log_cicj = (angular + 3) * .25 * log_of_doubled_exponents;

  double approximated_log_of_sum =
      (angular + l_inc) * log_r + log_of_doubled_exponents;
  approximated_log_of_sum += prefactor_in_log + log_cicj - threshold_in_log;
  if (approximated_log_of_sum < 0) {
    approximated_log_of_sum = 0;
  }
  return sqrt(approximated_log_of_sum / exponent);
}

template <int i_angular, int j_angular>
__global__ void count_non_trivial_pairs_kernel(
    int *n_counts, const int *i_shells, const int n_i_shells,
    const int *j_shells, const int n_j_shells,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env, const double threshold_in_log) {
  const int i_shell_image_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int j_shell_image_index = threadIdx.y + blockDim.y * blockIdx.y;
  bool is_valid_pair = i_shell_image_index < n_i_shells * n_images &&
                       j_shell_image_index < n_j_shells * n_images;

  int i_shell_index = 0, i_image = 0, j_shell_index = 0, j_image = 0;
  if (is_valid_pair) {
    if (n_i_shells > n_images) {
      i_image = i_shell_image_index / n_i_shells;
      i_shell_index = i_shell_image_index - i_image * n_i_shells;
    } else {
      i_shell_index = i_shell_image_index / n_images;
      i_image = i_shell_image_index - i_shell_index * n_images;
    }
    if (n_j_shells > n_images) {
      j_image = j_shell_image_index / n_j_shells;
      j_shell_index = j_shell_image_index - j_image * n_j_shells;
    } else {
      j_shell_index = j_shell_image_index / n_images;
      j_image = j_shell_image_index - j_shell_index * n_images;
    }
  }

  const int i_shell = is_valid_pair ? i_shells[i_shell_index] : 0;
  const int j_shell = is_valid_pair ? j_shells[j_shell_index] : 0;

  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
  const double i_x =
      env[i_coord_offset] + vectors_to_neighboring_images[i_image * 3];
  const double i_y =
      env[i_coord_offset + 1] + vectors_to_neighboring_images[i_image * 3 + 1];
  const double i_z =
      env[i_coord_offset + 2] + vectors_to_neighboring_images[i_image * 3 + 2];

  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
  const double j_x =
      env[j_coord_offset] + vectors_to_neighboring_images[j_image * 3];
  const double j_y =
      env[j_coord_offset + 1] + vectors_to_neighboring_images[j_image * 3 + 1];
  const double j_z =
      env[j_coord_offset + 2] + vectors_to_neighboring_images[j_image * 3 + 2];

  const double i_exponent = env[bas(PTR_EXP, i_shell)];
  const double j_exponent = env[bas(PTR_EXP, j_shell)];

  const double ij_exponent = i_exponent + j_exponent;
  const double ij_exponent_in_prefactor =
      i_exponent * j_exponent / ij_exponent *
      distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

  if (ij_exponent_in_prefactor > EIJ_CUTOFF) {
    is_valid_pair = false;
  }

  const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
  const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
  const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

  const double pair_a = pair_x * reciprocal_lattice_vectors[0] +
                        pair_y * reciprocal_lattice_vectors[1] +
                        pair_z * reciprocal_lattice_vectors[2];
  const double pair_b = pair_x * reciprocal_lattice_vectors[3] +
                        pair_y * reciprocal_lattice_vectors[4] +
                        pair_z * reciprocal_lattice_vectors[5];
  const double pair_c = pair_x * reciprocal_lattice_vectors[6] +
                        pair_y * reciprocal_lattice_vectors[7] +
                        pair_z * reciprocal_lattice_vectors[8];

  const double prefactor_in_log = -ij_exponent_in_prefactor +
                                  log_common_fac_sp<double, i_angular>() +
                                  log_common_fac_sp<double, j_angular>();

  const double cutoff = gaussian_summation_cutoff<i_angular + j_angular>(
      ij_exponent, prefactor_in_log, threshold_in_log);
  const double cutoff_a = cutoff * reciprocal_norm[0];
  const double cutoff_b = cutoff * reciprocal_norm[1];
  const double cutoff_c = cutoff * reciprocal_norm[2];

  int begin_a = ceil((pair_a - cutoff_a) * mesh_a);
  int end_a = floor((pair_a + cutoff_a) * mesh_a);
  int begin_b = ceil((pair_b - cutoff_b) * mesh_b);
  int end_b = floor((pair_b + cutoff_b) * mesh_b);
  int begin_c = ceil((pair_c - cutoff_c) * mesh_c);
  int end_c = floor((pair_c + cutoff_c) * mesh_c);

  if (begin_a > end_a || begin_b > end_b || begin_c > end_c || end_a < 0 ||
      end_b < 0 || end_c < 0 || begin_a >= mesh_a || begin_b >= mesh_b ||
      begin_c >= mesh_c) {
    is_valid_pair = false;
  }
  int count = is_valid_pair ? 1 : 0;
  int sum;
  sum =
      cub::BlockReduce<int, 16, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 16>()
          .Sum(count);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(n_counts, sum);
  }
}

template <int i_angular, int j_angular>
__global__ void screen_gaussian_pairs_kernel(
    int *shell_pair_indices, int *image_indices, int *pairs_to_blocks_begin,
    int *pairs_to_blocks_end, int *written_counts, const int *i_shells,
    const int n_i_shells, const int *j_shells, const int n_j_shells,
    const int n_pairs, const double *vectors_to_neighboring_images,
    const int n_images, const int mesh_a, const int mesh_b, const int mesh_c,
    const int *atm, const int *bas, const double *env,
    const double threshold_in_log) {

  const int i_shell_image_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int j_shell_image_index = threadIdx.y + blockDim.y * blockIdx.y;
  bool is_valid_pair = i_shell_image_index < n_i_shells * n_images &&
                       j_shell_image_index < n_j_shells * n_images;

  int i_shell_index = 0, i_image = 0, j_shell_index = 0, j_image = 0;
  if (is_valid_pair) {
    if (n_i_shells > n_images) {
      i_image = i_shell_image_index / n_i_shells;
      i_shell_index = i_shell_image_index - i_image * n_i_shells;
    } else {
      i_shell_index = i_shell_image_index / n_images;
      i_image = i_shell_image_index - i_shell_index * n_images;
    }
    if (n_j_shells > n_images) {
      j_image = j_shell_image_index / n_j_shells;
      j_shell_index = j_shell_image_index - j_image * n_j_shells;
    } else {
      j_shell_index = j_shell_image_index / n_images;
      j_image = j_shell_image_index - j_shell_index * n_images;
    }
  }

  const int i_shell = is_valid_pair ? i_shells[i_shell_index] : 0;
  const int j_shell = is_valid_pair ? j_shells[j_shell_index] : 0;
  const int shell_pair_index = i_shell_index * n_j_shells + j_shell_index;

  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
  const double i_x =
      env[i_coord_offset] + vectors_to_neighboring_images[i_image * 3];
  const double i_y =
      env[i_coord_offset + 1] + vectors_to_neighboring_images[i_image * 3 + 1];
  const double i_z =
      env[i_coord_offset + 2] + vectors_to_neighboring_images[i_image * 3 + 2];

  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
  const double j_x =
      env[j_coord_offset] + vectors_to_neighboring_images[j_image * 3];
  const double j_y =
      env[j_coord_offset + 1] + vectors_to_neighboring_images[j_image * 3 + 1];
  const double j_z =
      env[j_coord_offset + 2] + vectors_to_neighboring_images[j_image * 3 + 2];

  const double i_exponent = env[bas(PTR_EXP, i_shell)];
  const double j_exponent = env[bas(PTR_EXP, j_shell)];

  const double ij_exponent = i_exponent + j_exponent;
  const double ij_exponent_in_prefactor =
      i_exponent * j_exponent / ij_exponent *
      distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

  if (ij_exponent_in_prefactor > EIJ_CUTOFF) {
    is_valid_pair = false;
  }

  const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
  const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
  const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

  const double pair_a = pair_x * reciprocal_lattice_vectors[0] +
                        pair_y * reciprocal_lattice_vectors[1] +
                        pair_z * reciprocal_lattice_vectors[2];
  const double pair_b = pair_x * reciprocal_lattice_vectors[3] +
                        pair_y * reciprocal_lattice_vectors[4] +
                        pair_z * reciprocal_lattice_vectors[5];
  const double pair_c = pair_x * reciprocal_lattice_vectors[6] +
                        pair_y * reciprocal_lattice_vectors[7] +
                        pair_z * reciprocal_lattice_vectors[8];

  const double prefactor_in_log = -ij_exponent_in_prefactor +
                                  log_common_fac_sp<double, i_angular>() +
                                  log_common_fac_sp<double, j_angular>();

  const double cutoff = gaussian_summation_cutoff<i_angular + j_angular>(
      ij_exponent, prefactor_in_log, threshold_in_log);
  const double cutoff_a = cutoff * reciprocal_norm[0];
  const double cutoff_b = cutoff * reciprocal_norm[1];
  const double cutoff_c = cutoff * reciprocal_norm[2];

  int begin_a = ceil((pair_a - cutoff_a) * mesh_a);
  int end_a = floor((pair_a + cutoff_a) * mesh_a);
  int begin_b = ceil((pair_b - cutoff_b) * mesh_b);
  int end_b = floor((pair_b + cutoff_b) * mesh_b);
  int begin_c = ceil((pair_c - cutoff_c) * mesh_c);
  int end_c = floor((pair_c + cutoff_c) * mesh_c);

  if (begin_a > end_a || begin_b > end_b || begin_c > end_c || end_a < 0 ||
      end_b < 0 || end_c < 0 || begin_a >= mesh_a || begin_b >= mesh_b ||
      begin_c >= mesh_c) {
    is_valid_pair = false;
  }

  begin_a = max(begin_a, 0);
  begin_b = max(begin_b, 0);
  begin_c = max(begin_c, 0);
  end_a = min(end_a, mesh_a - 1);
  end_b = min(end_b, mesh_b - 1);
  end_c = min(end_c, mesh_c - 1);
  begin_a >>= 2;
  end_a >>= 2;
  begin_b >>= 2;
  end_b >>= 2;
  begin_c >>= 2;
  end_c >>= 2;

  int write_pair_index = is_valid_pair ? 1 : 0;
  int aggregated_pairs;
  cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 16>().ExclusiveSum(
      write_pair_index, write_pair_index, aggregated_pairs);
  __shared__ int offset_for_this_block;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    offset_for_this_block = atomicAdd(written_counts, aggregated_pairs);
  }
  __syncthreads();

  const int offset_for_this_thread = offset_for_this_block + write_pair_index;

  if (is_valid_pair) {
    shell_pair_indices[offset_for_this_thread] = shell_pair_index;
    image_indices[offset_for_this_thread] = i_image * n_images + j_image;
    pairs_to_blocks_begin[offset_for_this_thread] = begin_a;
    pairs_to_blocks_begin[offset_for_this_thread + n_pairs] = begin_b;
    pairs_to_blocks_begin[offset_for_this_thread + 2 * n_pairs] = begin_c;
    pairs_to_blocks_end[offset_for_this_thread] = end_a;
    pairs_to_blocks_end[offset_for_this_thread + n_pairs] = end_b;
    pairs_to_blocks_end[offset_for_this_thread + 2 * n_pairs] = end_c;
  }
}

__global__ void count_pairs_on_blocks_kernel(
    int *n_pairs_per_block, int *n_unstable_pairs_per_block,
    const int *pairs_to_blocks_begin, const int *pairs_to_blocks_end,
    const int n_pairs, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {
  const int block_index =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  const int a_start = blockIdx.x * BLOCK_DIM_XYZ;
  const int b_start = blockIdx.y * BLOCK_DIM_XYZ;
  const int c_start = blockIdx.z * BLOCK_DIM_XYZ;

  const double da_squared =
      distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
  const double db_squared =
      distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
  const double dc_squared =
      distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

  const double start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const double start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const double start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  int count = 0;
  int unstable_count = 0;
  constexpr int n_threads = 256;

  for (int i_pair = threadIdx.x; i_pair < n_pairs; i_pair += blockDim.x) {
    const int begin_block_a = pairs_to_blocks_begin[i_pair];
    const int end_block_a = pairs_to_blocks_end[i_pair];
    const int begin_block_b = pairs_to_blocks_begin[n_pairs + i_pair];
    const int end_block_b = pairs_to_blocks_end[n_pairs + i_pair];
    const int begin_block_c = pairs_to_blocks_begin[2 * n_pairs + i_pair];
    const int end_block_c = pairs_to_blocks_end[2 * n_pairs + i_pair];
    if (blockIdx.x >= begin_block_c && blockIdx.x <= end_block_c &&
        blockIdx.y >= begin_block_b && blockIdx.y <= end_block_b &&
        blockIdx.z >= begin_block_a && blockIdx.z <= end_block_a) {

      const int image_index = image_indices[i_pair];
      const int image_index_i = image_index / n_images;
      const int image_index_j = image_index % n_images;

      const int shell_pair_index = non_trivial_pairs[i_pair];
      const int i_shell_index = shell_pair_index / n_j_shells;
      const int j_shell_index = shell_pair_index % n_j_shells;
      const int i_shell = i_shells[i_shell_index];
      const int j_shell = j_shells[j_shell_index];

      const double i_exponent = env[bas(PTR_EXP, i_shell)];
      const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
      const double i_x = env[i_coord_offset] +
                         vectors_to_neighboring_images[image_index_i * 3];
      const double i_y = env[i_coord_offset + 1] +
                         vectors_to_neighboring_images[image_index_i * 3 + 1];
      const double i_z = env[i_coord_offset + 2] +
                         vectors_to_neighboring_images[image_index_i * 3 + 2];

      const double j_exponent = env[bas(PTR_EXP, j_shell)];
      const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
      const double j_x = env[j_coord_offset] +
                         vectors_to_neighboring_images[image_index_j * 3];
      const double j_y = env[j_coord_offset + 1] +
                         vectors_to_neighboring_images[image_index_j * 3 + 1];
      const double j_z = env[j_coord_offset + 2] +
                         vectors_to_neighboring_images[image_index_j * 3 + 2];

      const double ij_exponent = i_exponent + j_exponent;
      const double ij_exponent_in_prefactor =
          i_exponent * j_exponent / ij_exponent *
          distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

      const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
      const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
      const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

      const double x0 = start_position_x - pair_x;
      const double y0 = start_position_y - pair_y;
      const double z0 = start_position_z - pair_z;
      const double cross_term_a_exponent =
          -ij_exponent *
          (2 * (dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0) +
           da_squared);
      const double cross_term_b_exponent =
          -ij_exponent *
          (2 * (dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0) +
           db_squared);
      const double cross_term_c_exponent =
          -ij_exponent *
          (2 * (dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0) +
           dc_squared);

      if (cross_term_a_exponent <= EXP_OVERFLOW &&
          cross_term_b_exponent <= EXP_OVERFLOW &&
          cross_term_c_exponent <= EXP_OVERFLOW) {
        count++;
      } else {
        unstable_count++;
      }
    }
  }
  count = cub::BlockReduce<int, n_threads,
                           cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>()
              .Sum(count);
  __syncthreads();
  unstable_count = cub::BlockReduce<int, n_threads,
                                    cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>()
                       .Sum(unstable_count);
  if (threadIdx.x == 0) {
    n_pairs_per_block[block_index] = count;
    n_unstable_pairs_per_block[block_index] = unstable_count;
    if (count > 0) {
      atomicAdd(n_pairs_per_block + gridDim.x * gridDim.y * gridDim.z, 1);
    }
    if (unstable_count > 0) {
      atomicAdd(n_unstable_pairs_per_block + gridDim.x * gridDim.y * gridDim.z,
                1);
    }
  }
}

__global__ void put_pairs_on_blocks_kernel(
    int *pairs_on_blocks, const int *accumulated_n_pairs_per_block,
    const int *sorted_block_index, const int *pairs_to_blocks_begin,
    const int *pairs_to_blocks_end, const int n_blocks_a, const int n_blocks_b,
    const int n_blocks_c, const int n_pairs, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int mesh_a, const int mesh_b, const int mesh_c,
    const int *atm, const int *bas, const double *env) {
  const int block_index = sorted_block_index[blockIdx.x];
  const int n_blocks_bc = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / n_blocks_bc;
  const int block_bc_index = block_index % n_blocks_bc;
  const int block_b_index = block_bc_index / n_blocks_c;
  const int block_c_index = block_bc_index % n_blocks_c;

  const int a_start = block_a_index * BLOCK_DIM_XYZ;
  const int b_start = block_b_index * BLOCK_DIM_XYZ;
  const int c_start = block_c_index * BLOCK_DIM_XYZ;

  const double start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const double start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const double start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const double da_squared =
      distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
  const double db_squared =
      distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
  const double dc_squared =
      distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

  constexpr int n_threads = 256;
  int stored_pair_index[4];
  int valid_pairs[4];
  int exclusive_sum[4];
  int n_filtered_pairs_on_shared_memory = 0;
  int offset_on_global_memory = accumulated_n_pairs_per_block[block_index];
  constexpr int batch_size = 4 * n_threads;
  constexpr int shared_memory_size = 7 * n_threads;
  __shared__ int filtered_index[shared_memory_size];
  const int n_batches = (n_pairs + batch_size - 1) / batch_size;
  for (int i_batch = 0, i_pair = threadIdx.x; i_batch < n_batches; i_batch++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const bool is_valid_pair = i_pair < n_pairs;
      const int begin_block_a =
          is_valid_pair ? pairs_to_blocks_begin[i_pair] : 0;
      const int end_block_a = is_valid_pair ? pairs_to_blocks_end[i_pair] : -1;
      const int begin_block_b =
          is_valid_pair ? pairs_to_blocks_begin[n_pairs + i_pair] : 0;
      const int end_block_b =
          is_valid_pair ? pairs_to_blocks_end[n_pairs + i_pair] : -1;
      const int begin_block_c =
          is_valid_pair ? pairs_to_blocks_begin[2 * n_pairs + i_pair] : 0;
      const int end_block_c =
          is_valid_pair ? pairs_to_blocks_end[2 * n_pairs + i_pair] : -1;
      if (block_c_index >= begin_block_c && block_c_index <= end_block_c &&
          block_b_index >= begin_block_b && block_b_index <= end_block_b &&
          block_a_index >= begin_block_a && block_a_index <= end_block_a) {
        const int image_index = image_indices[i_pair];
        const int image_index_i = image_index / n_images;
        const int image_index_j = image_index % n_images;

        const int shell_pair_index = non_trivial_pairs[i_pair];
        const int i_shell_index = shell_pair_index / n_j_shells;
        const int j_shell_index = shell_pair_index % n_j_shells;
        const int i_shell = i_shells[i_shell_index];
        const int j_shell = j_shells[j_shell_index];

        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset] +
                           vectors_to_neighboring_images[image_index_i * 3];
        const double i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_i * 3 + 1];
        const double i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_i * 3 + 2];

        const double j_exponent = env[bas(PTR_EXP, j_shell)];
        const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
        const double j_x = env[j_coord_offset] +
                           vectors_to_neighboring_images[image_index_j * 3];
        const double j_y = env[j_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index_j * 3 + 1];
        const double j_z = env[j_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index_j * 3 + 2];

        const double ij_exponent = i_exponent + j_exponent;
        const double ij_exponent_in_prefactor =
            i_exponent * j_exponent / ij_exponent *
            distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

        const double pair_x =
            (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
        const double pair_y =
            (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
        const double pair_z =
            (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

        const double x0 = start_position_x - pair_x;
        const double y0 = start_position_y - pair_y;
        const double z0 = start_position_z - pair_z;

        const double cross_term_a_exponent =
            -ij_exponent *
            (2 * (dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0) +
             da_squared);
        const double cross_term_b_exponent =
            -ij_exponent *
            (2 * (dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0) +
             db_squared);
        const double cross_term_c_exponent =
            -ij_exponent *
            (2 * (dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0) +
             dc_squared);
        if (cross_term_a_exponent <= EXP_OVERFLOW &&
            cross_term_b_exponent <= EXP_OVERFLOW &&
            cross_term_c_exponent <= EXP_OVERFLOW) {
          stored_pair_index[i] = i_pair;
          valid_pairs[i] = 1;
        } else {
          // TODO: store as unstable pair
          stored_pair_index[i] = -2;
          valid_pairs[i] = 0;
        }
      } else {
        stored_pair_index[i] = -2;
        valid_pairs[i] = 0;
      }
      i_pair += n_threads;
    }
    int aggregated_block;
    cub::BlockScan<int, n_threads>().ExclusiveSum(valid_pairs, exclusive_sum,
                                                  aggregated_block);
    if ((aggregated_block + n_filtered_pairs_on_shared_memory) >
        shared_memory_size) {
      for (int i = threadIdx.x; i < n_filtered_pairs_on_shared_memory;
           i += n_threads) {
        pairs_on_blocks[offset_on_global_memory + i] = filtered_index[i];
      }
      offset_on_global_memory += n_filtered_pairs_on_shared_memory;
      n_filtered_pairs_on_shared_memory = 0;
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if (valid_pairs[i] == 1) {
        filtered_index[exclusive_sum[i] + n_filtered_pairs_on_shared_memory] =
            stored_pair_index[i];
      }
    }
    n_filtered_pairs_on_shared_memory += aggregated_block;
  }
  if (n_filtered_pairs_on_shared_memory > 0) {
    __syncthreads();
    for (int i = threadIdx.x; i < n_filtered_pairs_on_shared_memory;
         i += n_threads) {
      pairs_on_blocks[offset_on_global_memory + i] = filtered_index[i];
    }
    offset_on_global_memory += n_filtered_pairs_on_shared_memory;
  }
}

} // namespace gpu4pyscf::gpbc::multi_grid
