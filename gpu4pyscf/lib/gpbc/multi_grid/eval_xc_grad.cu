#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "evaluation.cuh"
#include "gradient.cuh"

extern "C" {

int evaluate_xc_gradient_driver(
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
#if 0
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 1,
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
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 2,
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
        return 1;
      }
    } else {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 1, false>(
            (float *)gradient, (float *)xc_weights, (float *)density_matrices,
            i_angular, j_angular, non_trivial_pairs, i_shells, j_shells,
            n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
            sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
            sorted_block_index, n_contributing_blocks, image_indices,
            vectors_to_neighboring_images, n_images,
            image_pair_difference_index, n_difference_images, mesh, atm, bas,
            env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<float, 2, false>(
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
#else
    fprintf(stderr, "single precision not available\n");
    return 1;
#endif
  } else {
    if (is_non_orthogonal) {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 1, true>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 2, true>(
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
        return 1;
      }
    } else {
      if (n_channels == 1) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 1, false>(
            (double *)gradient, (double *)xc_weights,
            (double *)density_matrices, i_angular, j_angular, non_trivial_pairs,
            i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
            n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, sorted_block_index,
            n_contributing_blocks, image_indices, vectors_to_neighboring_images,
            n_images, image_pair_difference_index, n_difference_images, mesh,
            atm, bas, env);
      } else if (n_channels == 2) {
        return gpu4pyscf::gpbc::multi_grid::gradient::evaluate_xc_driver<double, 2, false>(
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
        return 1;
      }
    }
  }
}
}
