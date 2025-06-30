#include <stdio.h>

#include "evaluation.cuh"
#include "gradient.cuh"
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

void count_non_trivial_pairs(int *n_counts, const int i_angular,
                             const int j_angular, const int *i_shells,
                             const int n_i_shells, const int *j_shells,
                             const int n_j_shells,
                             const double *vectors_to_neighboring_images,
                             const int n_images, const int *mesh,
                             const int *atm, const int *bas, const double *env,
                             const double threshold_in_log) {
  dim3 block_size(16, 16);
  dim3 block_grid(n_i_shells * n_images / 16 + 1,
                  n_j_shells * n_images / 16 + 1);
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

  checkCudaErrors(cudaPeekAtLastError());
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

void screen_gaussian_pairs(int *shell_pair_indices, int *image_indices,
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
  dim3 block_grid(n_i_shells * n_images / 16 + 1,
                  n_j_shells * n_images / 16 + 1);
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
  checkCudaErrors(cudaPeekAtLastError());

  checkCudaErrors(cudaFree(written_counts));
}

void count_pairs_on_blocks(int *n_pairs_per_block,
                           const int *pairs_to_blocks_begin,
                           const int *pairs_to_blocks_end,
                           const int n_blocks[3], const int n_pairs) {
  const int n_blocks_a = n_blocks[0];
  const int n_blocks_b = n_blocks[1];
  const int n_blocks_c = n_blocks[2];
  const int n_threads = 256;
  const dim3 block_size(n_threads, 1, 1);
  const dim3 block_grid(n_blocks_c, n_blocks_b, n_blocks_a);
  gpu4pyscf::gpbc::multi_grid::
      count_pairs_on_blocks_kernel<<<block_grid, block_size>>>(
          n_pairs_per_block, pairs_to_blocks_begin, pairs_to_blocks_end,
          n_pairs);

  checkCudaErrors(cudaPeekAtLastError());
}

void put_pairs_on_blocks(int *pairs_on_blocks,
                         const int *accumulated_n_pairs_per_block,
                         const int *sorted_block_index,
                         const int *pairs_to_blocks_begin,
                         const int *pairs_to_blocks_end, const int n_blocks[3],
                         const int n_contributing_blocks, const int n_pairs) {
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
          n_blocks_c, n_pairs);

  checkCudaErrors(cudaPeekAtLastError());
}

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

void evaluate_density_driver(
    void *density, const void *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, const int n_channels,
    const int is_non_orthogonal, const int use_float_precision) {
  if (use_float_precision) {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 1, true>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 2, true>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
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
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 1, false>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<float, 2, false>(
            (float *)density, (float *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
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
  } else {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 1, true>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 2, true>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
            double, true>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
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
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 1, false>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_density_driver<double, 2, false>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_density_driver<
            double, false>(
            (double *)density, (double *)density_matrices, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env, n_channels);
      }
    }
  }
}

void evaluate_xc_driver(
    void *fock, const void *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, const int n_channels,
    const int is_non_orthogonal, const int use_float_precision) {
  if (use_float_precision) {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 1, true>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 2, true>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<float,
                                                                         true>(
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
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 1, false>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<float, 2, false>(
            (float *)fock, (float *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<float,
                                                                         false>(
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
  } else {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 1, true>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 2, true>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<double,
                                                                         true>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
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
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 1, false>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::evaluate_xc_driver<double, 2, false>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        gpu4pyscf::gpbc::multi_grid::runtime_channel::evaluate_xc_driver<double,
                                                                         false>(
            (double *)fock, (double *)xc_weights, i_angular, j_angular,
            non_trivial_pairs, i_shells, j_shells, n_j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env, n_channels);
      }
    }
  }
}

void evaluate_xc_gradient_driver(
    void *gradient, const void *xc_weights, const void *density_matrices,
    const int i_angular, const int j_angular, const int *non_trivial_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *image_pair_difference_index,
    const int n_difference_images, const int *mesh, const int *atm,
    const int *bas, const double *env, const int n_channels,
    const int is_non_orthogonal, const int use_float_precision) {
  if (use_float_precision) {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 1,
                                                                  true>(
            (float *)gradient, (float *)xc_weights, (float *)density_matrices,
            i_angular, j_angular, non_trivial_pairs, i_shells, j_shells,
            n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 2,
                                                                  true>(
            (float *)gradient, (float *)xc_weights, (float *)density_matrices,
            i_angular, j_angular, non_trivial_pairs, i_shells, j_shells,
            n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else {
        fprintf(stderr,
                "evaluate_xc_gradient_driver: n_channels > 2 not supported");
      }
    } else {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 1,
                                                                  false>(
            (float *)gradient, (float *)xc_weights, (float *)density_matrices,
            i_angular, j_angular, non_trivial_pairs, i_shells, j_shells,
            n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 2,
                                                                  false>(
            (float *)gradient, (float *)xc_weights, (float *)density_matrices,
            i_angular, j_angular, non_trivial_pairs, i_shells, j_shells,
            n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      }
    }
  } else {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 1,
                                                                  true>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 2,
                                                                  true>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else {
        fprintf(stderr,
                "evaluate_xc_gradient_driver: n_channels > 2 not supported");
      }
    } else {
      if (n_channels == 1) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 1,
                                                                  false>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else if (n_channels == 2) {
        gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 2,
                                                                  false>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else {
        fprintf(stderr,
                "evaluate_xc_gradient_driver: n_channels > 2 not supported");
      }
    }
  }
}
}