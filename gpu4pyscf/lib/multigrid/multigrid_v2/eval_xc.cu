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

#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "evaluation.cuh"

extern "C" {

int evaluate_xc_driver(
    double *fock, double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, int n_channels,
    const int is_non_orthogonal, const int use_float_precision) {
  if (use_float_precision) {
#if 0
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 1, true>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 2, true>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        // TODO: general n_channels function has been removed, the compilation of this call will fail.
        return gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<float, true>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env, n_channels);
      }
    } else {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 1, false>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 2, false>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        // TODO: general n_channels function has been removed, the compilation of this call will fail.
        return gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<float, false>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env, n_channels);
      }
    }
#else
    fprintf(stderr, "single precision not available\n");
    return 1;
#endif
  } else {
    size_t size_dm = (size_t)n_i_functions * n_j_functions * n_difference_images;
    size_t ngrids = (size_t)mesh[0] * mesh[1] * mesh[2];
    int err;
    while (n_channels > 0) {
      if (is_non_orthogonal) {
        if (n_channels == 1 ||
            // two channels requires too many registers for high orders.
            i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 1, true>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids;
          fock += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 2, true>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2;
          fock += size_dm * 2;
          n_channels -= 2;
        }
      } else {
        if (n_channels == 1 || i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 1, false>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids;
          fock += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 2, false>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2;
          fock += size_dm * 2;
          n_channels -= 2;
        }
      }
      if (err != 0) {
          return err;
      }
    }
    return 0;
  }
}

int evaluate_xc_with_tau_driver(
    double *fock, double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, int n_channels,
    const int is_non_orthogonal, const int use_float_precision) {
  if (use_float_precision) {
    fprintf(stderr, "single precision not available\n");
    return 1;
  } else {
    size_t size_dm = (size_t)n_i_functions * n_j_functions * n_difference_images;
    size_t ngrids = (size_t)mesh[0] * mesh[1] * mesh[2];
    int err;
    while (n_channels > 0) {
      if (is_non_orthogonal) {
        if (n_channels == 1 ||
            // two channels requires too many registers for high orders.
            i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_with_tau_driver<double, 1, true>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2;
          fock += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_with_tau_driver<double, 2, true>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2 * 2;
          fock += size_dm * 2;
          n_channels -= 2;
        }
      } else {
        if (n_channels == 1 || i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_with_tau_driver<double, 1, false>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2;
          fock += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_xc_with_tau_driver<double, 2, false>(
              (double *)fock, (double *)xc_weights, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          xc_weights += ngrids * 2 * 2;
          fock += size_dm * 2;
          n_channels -= 2;
        }
      }
      if (err != 0) {
          return err;
      }
    }
    return 0;
  }
}
}
