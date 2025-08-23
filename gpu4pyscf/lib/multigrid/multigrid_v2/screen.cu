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

#include <complex.h>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>

#include "screening.cuh"

extern "C" {

#define count_non_trivial_pairs_kernel_macro(li, lj)                           \
  gpu4pyscf::gpbc::multi_grid::count_non_trivial_pairs_kernel<li, lj>          \
      <<<block_grid, block_size>>>(n_counts, i_shells, n_i_shells, j_shells,   \
                                   n_j_shells, vectors_to_neighboring_images,  \
                                   n_images, mesh_a, mesh_b, mesh_c, atm, bas, \
                                   env, threshold_in_log)

#define count_non_trivial_pairs_kernel_case_macro(li, lj)                      \
  case (li * 10 + lj):                                                         \
    count_non_trivial_pairs_kernel_macro(li, lj);                              \
    break

int count_non_trivial_pairs(int *n_counts, const int i_angular,
                            const int j_angular, const int *i_shells,
                            const int n_i_shells, const int *j_shells,
                            const int n_j_shells,
                            const double *vectors_to_neighboring_images,
                            const int n_images, const int *mesh, const int *atm,
                            const int *bas, const double *env,
                            const double threshold_in_log) {
  dim3 block_size(16, 16);
  dim3 block_grid((n_i_shells * n_images + 15) / 16,
                  (n_j_shells * n_images + 15) / 16);
  const int mesh_a = mesh[0];
  const int mesh_b = mesh[1];
  const int mesh_c = mesh[2];
  switch (i_angular * 10 + j_angular) {
    count_non_trivial_pairs_kernel_case_macro(0, 0);
    count_non_trivial_pairs_kernel_case_macro(0, 1);
    count_non_trivial_pairs_kernel_case_macro(0, 2);
    count_non_trivial_pairs_kernel_case_macro(0, 3);
    count_non_trivial_pairs_kernel_case_macro(0, 4);
    count_non_trivial_pairs_kernel_case_macro(1, 0);
    count_non_trivial_pairs_kernel_case_macro(1, 1);
    count_non_trivial_pairs_kernel_case_macro(1, 2);
    count_non_trivial_pairs_kernel_case_macro(1, 3);
    count_non_trivial_pairs_kernel_case_macro(1, 4);
    count_non_trivial_pairs_kernel_case_macro(2, 0);
    count_non_trivial_pairs_kernel_case_macro(2, 1);
    count_non_trivial_pairs_kernel_case_macro(2, 2);
    count_non_trivial_pairs_kernel_case_macro(2, 3);
    count_non_trivial_pairs_kernel_case_macro(2, 4);
    count_non_trivial_pairs_kernel_case_macro(3, 0);
    count_non_trivial_pairs_kernel_case_macro(3, 1);
    count_non_trivial_pairs_kernel_case_macro(3, 2);
    count_non_trivial_pairs_kernel_case_macro(3, 3);
    count_non_trivial_pairs_kernel_case_macro(3, 4);
    count_non_trivial_pairs_kernel_case_macro(4, 0);
    count_non_trivial_pairs_kernel_case_macro(4, 1);
    count_non_trivial_pairs_kernel_case_macro(4, 2);
    count_non_trivial_pairs_kernel_case_macro(4, 3);
    count_non_trivial_pairs_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "count_non_trivial_pairs\n",
            i_angular, j_angular);
  }

  return checkCudaErrors(cudaPeekAtLastError());
}

#define screen_gaussian_pairs_kernel_macro(li, lj)                             \
  gpu4pyscf::gpbc::multi_grid::screen_gaussian_pairs_kernel<li, lj>            \
      <<<block_grid, block_size>>>(                                            \
          shell_pair_indices, image_indices, pairs_to_blocks_begin,            \
          pairs_to_blocks_end, written_counts, i_shells, n_i_shells, j_shells, \
          n_j_shells, n_pairs, vectors_to_neighboring_images, n_images,        \
          mesh_a, mesh_b, mesh_c, atm, bas, env, threshold_in_log)

#define screen_gaussian_pairs_kernel_case_macro(li, lj)                        \
  case (li * 10 + lj):                                                         \
    screen_gaussian_pairs_kernel_macro(li, lj);                                \
    break

int screen_gaussian_pairs(int *shell_pair_indices, int *image_indices,
                          int *pairs_to_blocks_begin, int *pairs_to_blocks_end,
                          const int i_angular, const int j_angular,
                          const int *i_shells, const int n_i_shells,
                          const int *j_shells, const int n_j_shells,
                          const int n_pairs,
                          const double *vectors_to_neighboring_images,
                          const int n_images, const int *mesh, const int *atm,
                          const int *bas, const double *env,
                          const double threshold_in_log) {
  dim3 block_size(16, 16);
  dim3 block_grid((n_i_shells * n_images + 15) / 16,
                  (n_j_shells * n_images + 15) / 16);
  const int mesh_a = mesh[0];
  const int mesh_b = mesh[1];
  const int mesh_c = mesh[2];
  int *written_counts = nullptr;
  checkCudaErrors(cudaMalloc(&written_counts, sizeof(int)));
  checkCudaErrors(cudaMemset(written_counts, 0, sizeof(int)));
  switch (i_angular * 10 + j_angular) {
    screen_gaussian_pairs_kernel_case_macro(0, 0);
    screen_gaussian_pairs_kernel_case_macro(0, 1);
    screen_gaussian_pairs_kernel_case_macro(0, 2);
    screen_gaussian_pairs_kernel_case_macro(0, 3);
    screen_gaussian_pairs_kernel_case_macro(0, 4);
    screen_gaussian_pairs_kernel_case_macro(1, 0);
    screen_gaussian_pairs_kernel_case_macro(1, 1);
    screen_gaussian_pairs_kernel_case_macro(1, 2);
    screen_gaussian_pairs_kernel_case_macro(1, 3);
    screen_gaussian_pairs_kernel_case_macro(1, 4);
    screen_gaussian_pairs_kernel_case_macro(2, 0);
    screen_gaussian_pairs_kernel_case_macro(2, 1);
    screen_gaussian_pairs_kernel_case_macro(2, 2);
    screen_gaussian_pairs_kernel_case_macro(2, 3);
    screen_gaussian_pairs_kernel_case_macro(2, 4);
    screen_gaussian_pairs_kernel_case_macro(3, 0);
    screen_gaussian_pairs_kernel_case_macro(3, 1);
    screen_gaussian_pairs_kernel_case_macro(3, 2);
    screen_gaussian_pairs_kernel_case_macro(3, 3);
    screen_gaussian_pairs_kernel_case_macro(3, 4);
    screen_gaussian_pairs_kernel_case_macro(4, 0);
    screen_gaussian_pairs_kernel_case_macro(4, 1);
    screen_gaussian_pairs_kernel_case_macro(4, 2);
    screen_gaussian_pairs_kernel_case_macro(4, 3);
    screen_gaussian_pairs_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "screen_gaussian_pairs_kernel\n",
            i_angular, j_angular);
  }
  int err = checkCudaErrors(cudaPeekAtLastError());

  checkCudaErrors(cudaFree(written_counts));
  return err;
}

int count_pairs_on_blocks(int *n_pairs_per_block,
                          int *n_unstable_pairs_per_block,
                          const int *pairs_to_blocks_begin,
                          const int *pairs_to_blocks_end, const int n_blocks[3],
                          const int n_pairs, const int *non_trivial_pairs,
                          const int *i_shells, const int *j_shells,
                          const int n_j_shells, const int *image_indices,
                          const double *vectors_to_neighboring_images,
                          const int n_images, const int mesh[3], const int *atm,
                          const int *bas, const double *env) {
  const int n_blocks_a = n_blocks[0];
  const int n_blocks_b = n_blocks[1];
  const int n_blocks_c = n_blocks[2];
  const int n_threads = 256;
  const dim3 block_size(n_threads, 1, 1);
  const dim3 block_grid(n_blocks_c, n_blocks_b, n_blocks_a);
  gpu4pyscf::gpbc::multi_grid::
      count_pairs_on_blocks_kernel<<<block_grid, block_size>>>(
          n_pairs_per_block, n_unstable_pairs_per_block, pairs_to_blocks_begin,
          pairs_to_blocks_end, n_pairs, non_trivial_pairs, i_shells, j_shells,
          n_j_shells, image_indices, vectors_to_neighboring_images, n_images,
          mesh[0], mesh[1], mesh[2], atm, bas, env);

  return checkCudaErrors(cudaPeekAtLastError());
}

void put_pairs_on_blocks(
    int *pairs_on_blocks, const int *accumulated_n_pairs_per_block,
    const int *sorted_block_index, const int *pairs_to_blocks_begin,
    const int *pairs_to_blocks_end, const int n_blocks[3],
    const int n_contributing_blocks, const int n_pairs,
    const int *non_trivial_pairs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh[3], const int *atm, const int *bas, const double *env) {
  const int n_blocks_a = n_blocks[0];
  const int n_blocks_b = n_blocks[1];
  const int n_blocks_c = n_blocks[2];
  const int n_threads = 256;
  const dim3 block_size(n_threads);
  const dim3 block_grid(n_contributing_blocks);
  gpu4pyscf::gpbc::multi_grid::
      put_pairs_on_blocks_kernel<<<block_grid, block_size>>>(
          pairs_on_blocks, accumulated_n_pairs_per_block, sorted_block_index,
          pairs_to_blocks_begin, pairs_to_blocks_end, n_blocks_a, n_blocks_b,
          n_blocks_c, n_pairs, non_trivial_pairs, i_shells, j_shells,
          n_j_shells, image_indices, vectors_to_neighboring_images, n_images,
          mesh[0], mesh[1], mesh[2], atm, bas, env);

  checkCudaErrors(cudaPeekAtLastError());
}
}
