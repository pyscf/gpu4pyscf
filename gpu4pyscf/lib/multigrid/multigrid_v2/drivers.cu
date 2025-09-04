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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "evaluation.cuh"

namespace gpu4pyscf::gpbc::multi_grid {

__constant__ double lattice_vectors[9];
__constant__ double reciprocal_lattice_vectors[9];
__constant__ double dxyz_dabc[9];
__constant__ double reciprocal_norm[3];

} // namespace gpu4pyscf::gpbc::multi_grid

extern "C" {
void update_lattice_vectors(const double *lattice_vectors_on_device,
                            const double *reciprocal_lattice_vectors_on_device,
                            const double *reciprocal_norm_on_device) {
  cudaMemcpyToSymbol(gpu4pyscf::gpbc::multi_grid::lattice_vectors,
                     lattice_vectors_on_device, 9 * sizeof(double));
  cudaMemcpyToSymbol(gpu4pyscf::gpbc::multi_grid::reciprocal_lattice_vectors,
                     reciprocal_lattice_vectors_on_device, 9 * sizeof(double));
  cudaMemcpyToSymbol(gpu4pyscf::gpbc::multi_grid::reciprocal_norm,
                     reciprocal_norm_on_device, 3 * sizeof(double));
}

void update_dxyz_dabc(const double *dxyz_dabc_on_device) {
  cudaMemcpyToSymbol(gpu4pyscf::gpbc::multi_grid::dxyz_dabc,
                     dxyz_dabc_on_device, 9 * sizeof(double));
}

int evaluate_density_driver(
    double *density, double *density_matrices, const int i_angular,
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
        return gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 1, true>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 2, true>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        // TODO: general n_channels function has been removed, the compilation of this call will fail.
        return gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
            float, true>((float *)density, (float *)density_matrices, i_angular,
                         j_angular, non_trivial_pairs, i_shells, j_shells,
                         n_j_shells, shell_to_ao_indices, n_i_functions,
                         n_j_functions, sorted_pairs_per_local_grid,
                         accumulated_n_pairs_per_local_grid, sorted_block_index,
                         n_contributing_blocks, image_indices,
                         vectors_to_neighboring_images, n_images,
                         image_pair_difference_index, n_difference_images, mesh,
                         atm, bas, env, n_channels);
      }
    } else {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 1, false>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 2, false>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        // TODO: general n_channels function has been removed, the compilation of this call will fail.
        return gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
            float, false>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
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
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 1, true>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids;
          density_matrices += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 2, true>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2;
          density_matrices += size_dm * 2;
          n_channels -= 2;
        }
      } else {
        if (n_channels == 1 || i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 1, false>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids;
          density_matrices += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 2, false>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2;
          density_matrices += size_dm * 2;
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

int evaluate_density_tau_driver(
    double *density, double *density_matrices, const int i_angular,
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
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_tau_driver<double, 1, true>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2;
          density_matrices += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_tau_driver<double, 2, true>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2 * 2;
          density_matrices += size_dm * 2;
          n_channels -= 2;
        }
      } else {
        if (n_channels == 1 || i_angular + j_angular >= 6) {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_tau_driver<double, 1, false>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2;
          density_matrices += size_dm;
          n_channels -= 1;
        } else {
          err = gpu4pyscf::gpbc::multi_grid::evaluate_density_tau_driver<double, 2, false>(
              (double *)density, (double *)density_matrices, i_angular, j_angular,
              non_trivial_pairs, i_shells, j_shells, n_j_shells,
              shell_to_ao_indices, n_i_functions, n_j_functions,
              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
              sorted_block_index, n_contributing_blocks, image_indices,
              vectors_to_neighboring_images, n_images,
              image_pair_difference_index, n_difference_images, mesh, atm, bas,
              env);
          density += ngrids * 2 * 2;
          density_matrices += size_dm * 2;
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
