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

template <int i_angular, int j_angular, bool is_non_orthogonal>
__global__ static void tailor_gaussian_pairs_kernel(
    int *sorted_pairs_per_local_grid, int *n_pairs_per_local_grid,
    const int *non_trivial_pairs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *shell_to_ao_indices,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env, const double threshold,
    const int derivative_order) {
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

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

  const double start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const double start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const double start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const double a_dot_b = dxyz_dabc[0] * dxyz_dabc[3] +
                         dxyz_dabc[1] * dxyz_dabc[4] +
                         dxyz_dabc[2] * dxyz_dabc[5];
  const double a_dot_c = dxyz_dabc[0] * dxyz_dabc[6] +
                         dxyz_dabc[1] * dxyz_dabc[7] +
                         dxyz_dabc[2] * dxyz_dabc[8];
  const double b_dot_c = dxyz_dabc[3] * dxyz_dabc[6] +
                         dxyz_dabc[4] * dxyz_dabc[7] +
                         dxyz_dabc[5] * dxyz_dabc[8];

  const int a_upper = min(a_start + BLOCK_DIM_XYZ, mesh_a) - a_start;
  const int b_upper = min(b_start + BLOCK_DIM_XYZ, mesh_b) - b_start;
  const int c_upper = min(c_start + BLOCK_DIM_XYZ, mesh_c) - c_start;

  const int thread_id = threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ +
                        threadIdx.z * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

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
    const double i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const double i_y = env[i_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_i * 3 + 1];
    const double i_z = env[i_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_i * 3 + 2];
    const double i_coeff = env[bas(PTR_COEFF, i_shell)];

    const double j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const double j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const double j_y = env[j_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_j * 3 + 1];
    const double j_z = env[j_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_j * 3 + 2];
    const double j_coeff = env[bas(PTR_COEFF, j_shell)];

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

    const double gaussian_exponent_at_reference =
        ij_exponent * distance_squared(x0, y0, z0);

    const double pair_prefactor = i_coeff * j_coeff *
                                  common_fac_sp<double, i_angular>() *
                                  common_fac_sp<double, j_angular>();

    const double gaussian_starting_point =
        is_valid_pair
            ? exp(-(ij_exponent_in_prefactor + gaussian_exponent_at_reference) /
                  3.0)
            : 0;

    const double da_squared =
        distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    const double db_squared =
        distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    const double dc_squared =
        distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);

    const double exp_da_squared = exp(-2 * ij_exponent * da_squared);
    const double exp_db_squared = exp(-2 * ij_exponent * db_squared);
    const double exp_dc_squared = exp(-2 * ij_exponent * dc_squared);

    const double cross_term_a =
        dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
    const double cross_term_b =
        dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
    const double cross_term_c =
        dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

    const double recursion_factor_a_start =
        exp(-ij_exponent * (2 * cross_term_a + da_squared));
    const double recursion_factor_b_start =
        exp(-ij_exponent * (2 * cross_term_b + db_squared));
    const double recursion_factor_c_start =
        exp(-ij_exponent * (2 * cross_term_c + dc_squared));

    const double exp_dadb = exp(-2 * ij_exponent * a_dot_b);
    const double exp_dadc = exp(-2 * ij_exponent * a_dot_c);
    const double exp_dbdc = exp(-2 * ij_exponent * b_dot_c);

    int a_index, b_index, c_index;
    double x, y, z;
    double gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c;
    double recursion_factor_ab_pow_a = 1;
    double recursion_factor_ac_pow_a = 1;
    double recursion_factor_bc_pow_b = 1;

    if constexpr (is_non_orthogonal) {
      // recursion_factor_ab_pow_a = 1;
      // recursion_factor_ac_pow_a = 1;
    } else {
      x = start_position_x;
    }

    double max_gaussian_value = 0;

    for (a_index = 0, gaussian_x = gaussian_starting_point,
        recursion_factor_a = recursion_factor_a_start;
         a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
        recursion_factor_a *= exp_da_squared) {

      if constexpr (is_non_orthogonal) {
        recursion_factor_bc_pow_b = 1;
      } else {
        y = start_position_y;
      }
      for (b_index = 0, gaussian_y = gaussian_starting_point,
          recursion_factor_b = recursion_factor_b_start;
           b_index < b_upper; b_index++,
          gaussian_y *= recursion_factor_b * recursion_factor_ab_pow_a,
          recursion_factor_b *= exp_db_squared) {

        if constexpr (is_non_orthogonal) {
          x = start_position_x + a_index * dxyz_dabc[0] +
              b_index * dxyz_dabc[3];
          y = start_position_y + a_index * dxyz_dabc[1] +
              b_index * dxyz_dabc[4];
          z = start_position_z + a_index * dxyz_dabc[2] +
              b_index * dxyz_dabc[5];
        } else {
          z = start_position_z;
        }
        for (c_index = 0, gaussian_z = gaussian_starting_point,
            recursion_factor_c = recursion_factor_c_start;
             c_index < c_upper; c_index++,
            gaussian_z *= recursion_factor_c * recursion_factor_ac_pow_a *
                                recursion_factor_bc_pow_b,
            recursion_factor_c *= exp_dc_squared) {

          const double r_i = sqrt(distance_squared(x - i_x, y - i_y, z - i_z));
          const double r_j = sqrt(distance_squared(x - j_x, y - j_y, z - j_z));
          const double r_p =
              sqrt(distance_squared(x - pair_x, y - pair_y, z - pair_z));

          const double approxmate_polynomial =
              approximate_polynomial_value<double, i_angular, j_angular>(
                  r_i, r_j, r_p, derivative_order);

          const double gaussian = gaussian_x * gaussian_y * gaussian_z;

          const double approximate_value =
              abs(4.0 * M_PI * r_p * r_p * pair_prefactor *
                  approxmate_polynomial * gaussian);

          max_gaussian_value = max(max_gaussian_value, approximate_value);

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

    if (max_gaussian_value < threshold && is_valid_pair) {
      sorted_pairs_per_local_grid[i_pair_index] = -1;
      atomicAdd(n_pairs_per_local_grid + block_index, -1);
    }
  }
}

#define tailor_gaussian_pairs_kernel_macro(li, lj)                             \
  tailor_gaussian_pairs_kernel<li, lj, is_non_orthogonal>                      \
      <<<block_grid, block_size>>>(                                            \
          sorted_pairs_per_local_grid, n_pairs_per_local_grid,                 \
          non_trivial_pairs, i_shells, j_shells, n_j_shells,                   \
          shell_to_ao_indices, accumulated_n_pairs_per_local_grid,             \
          sorted_block_index, image_indices, vectors_to_neighboring_images,    \
          n_images, mesh_a, mesh_b, mesh_c, atm, bas, env, threshold,          \
          derivative_order);

#define tailor_gaussian_pairs_kernel_case_macro(li, lj)                        \
  case (li * 10 + lj):                                                         \
    tailor_gaussian_pairs_kernel_macro(li, lj);                                \
    break

template <bool is_non_orthogonal>
int tailor_gaussian_pairs_driver(
    int *sorted_pairs_per_local_grid, int *n_pairs_per_local_grid,
    const int i_angular, const int j_angular, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *mesh, const int *atm, const int *bas,
    const double *env, const double threshold, const int derivative_order) {
  dim3 block_size(BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);
  switch (i_angular * 10 + j_angular) {
    tailor_gaussian_pairs_kernel_case_macro(0, 0);
    tailor_gaussian_pairs_kernel_case_macro(0, 1);
    tailor_gaussian_pairs_kernel_case_macro(0, 2);
    tailor_gaussian_pairs_kernel_case_macro(0, 3);
    tailor_gaussian_pairs_kernel_case_macro(0, 4);
    tailor_gaussian_pairs_kernel_case_macro(1, 0);
    tailor_gaussian_pairs_kernel_case_macro(1, 1);
    tailor_gaussian_pairs_kernel_case_macro(1, 2);
    tailor_gaussian_pairs_kernel_case_macro(1, 3);
    tailor_gaussian_pairs_kernel_case_macro(1, 4);
    tailor_gaussian_pairs_kernel_case_macro(2, 0);
    tailor_gaussian_pairs_kernel_case_macro(2, 1);
    tailor_gaussian_pairs_kernel_case_macro(2, 2);
    tailor_gaussian_pairs_kernel_case_macro(2, 3);
    tailor_gaussian_pairs_kernel_case_macro(2, 4);
    tailor_gaussian_pairs_kernel_case_macro(3, 0);
    tailor_gaussian_pairs_kernel_case_macro(3, 1);
    tailor_gaussian_pairs_kernel_case_macro(3, 2);
    tailor_gaussian_pairs_kernel_case_macro(3, 3);
    tailor_gaussian_pairs_kernel_case_macro(3, 4);
    tailor_gaussian_pairs_kernel_case_macro(4, 0);
    tailor_gaussian_pairs_kernel_case_macro(4, 1);
    tailor_gaussian_pairs_kernel_case_macro(4, 2);
    tailor_gaussian_pairs_kernel_case_macro(4, 3);
    tailor_gaussian_pairs_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_density_driver\n",
            i_angular, j_angular);
    return 1;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

} // namespace gpu4pyscf::gpbc::multi_grid
