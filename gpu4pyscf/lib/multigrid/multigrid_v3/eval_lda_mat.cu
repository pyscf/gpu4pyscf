/*
 * Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "constant_objects.cuh"
#include "cartesian.cuh"
#include "utils.cuh"

#define TILE            4
#define WARP_SIZE       32
#define THREADS         64

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J, bool is_non_orthogonal>
__global__ static
void eval_lda_mat_kernel(double *out, double *vxc_weights, PBCIntEnvVars envs,
                         double *supmol_img_coords,
                         int *shl_pair_offsets, int64_t *dressed_bas_ij_idx,
                         int *grid_tile_index, int ntiles, int tiles_per_block,
                         double a_dot_b, double a_dot_c, double b_dot_c,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c, double negligible)
{
    constexpr int threads = THREADS;
    int thread_id = threadIdx.x;
    int tile_id0 = blockIdx.x * tiles_per_block;
    __shared__ int a_upper, b_upper, c_upper;
    __shared__ double start_position_x, start_position_y, start_position_z;
    __shared__ double vxc_cache[TILE*TILE*TILE];

for (int tile_id = tile_id0; tile_id < min(tile_id0+tiles_per_block, ntiles); tile_id++) {
    int tile_index = grid_tile_index[tile_id];
    int shl_pair0 = shl_pair_offsets[tile_id];
    int shl_pair1 = shl_pair_offsets[tile_id+1];
    int n_tiles_b = (mesh_b + TILE - 1) / TILE;
    int n_tiles_c = (mesh_c + TILE - 1) / TILE;
    int tile_ab_index = tile_index / n_tiles_c;
    int tile_c_index = tile_index % n_tiles_c;
    int tile_a_index = tile_ab_index / n_tiles_b;
    int tile_b_index = tile_ab_index % n_tiles_b;
    int a_start = tile_a_index * TILE;
    int b_start = tile_b_index * TILE;
    int c_start = tile_c_index * TILE;

    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;

    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    int bvk_nbas = envs.bvk_ncells * envs.nbas;

    if (thread_id == 0) {
        start_position_x = c_dxyz_dabc[0] * a_start + c_dxyz_dabc[3] * b_start + c_dxyz_dabc[6] * c_start;
        start_position_y = c_dxyz_dabc[1] * a_start + c_dxyz_dabc[4] * b_start + c_dxyz_dabc[7] * c_start;
        start_position_z = c_dxyz_dabc[2] * a_start + c_dxyz_dabc[5] * b_start + c_dxyz_dabc[8] * c_start;
        a_upper = min(a_start + TILE, mesh_a) - a_start;
        b_upper = min(b_start + TILE, mesh_b) - b_start;
        c_upper = min(c_start + TILE, mesh_c) - c_start;
    }

    int a_idx = a_start + thread_id / (TILE*TILE);
    int b_idx = b_start + thread_id / TILE % TILE;
    int c_idx = c_start + thread_id % TILE;
    if (a_idx < mesh_a && b_idx < mesh_b && c_idx < mesh_c) {
        size_t abc_idx = (a_idx * mesh_b + b_idx) * (size_t)mesh_c + c_idx;
        vxc_cache[thread_id] = vxc_weights[abc_idx];
    }
    __syncthreads();

    for (int pair_id = shl_pair0+thread_id; pair_id < shl_pair1; pair_id += threads) {
        int64_t bas_ij = dressed_bas_ij_idx[pair_id];
        int ish = bas_ij / NBAS_MAX;
        int jsh = bas_ij % NBAS_MAX;
        int latsum_idx = ish / nbas;
        ish = ish - nbas * latsum_idx;
        int jL = jsh / bvk_nbas;
        jsh = jsh - bvk_nbas * jL;
        double Lx = supmol_img_coords[latsum_idx*3+0];
        double Ly = supmol_img_coords[latsum_idx*3+1];
        double Lz = supmol_img_coords[latsum_idx*3+2];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = env[ri+0] - Lx;
        double yi = env[ri+1] - Ly;
        double zi = env[ri+2] - Lz;
        double xj = env[rj+0] - Lx + envs.img_coords[jL*3+0];
        double yj = env[rj+1] - Ly + envs.img_coords[jL*3+1];
        double zj = env[rj+2] - Lz + envs.img_coords[jL*3+2];
        double xjxi = xj - xi;
        double yjyi = yj - yi;
        double zjzi = zj - zi;
        double rr_ij = distance_squared(xjxi, yjyi, zjzi);
        double ai = env[expi];
        double aj = env[expj];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double theta_ij = ai * aj_aij;
        double xij = xjxi * aj_aij + xi;
        double yij = yjyi * aj_aij + yi;
        double zij = zjzi * aj_aij + zi;
        double x0 = start_position_x - xij;
        double y0 = start_position_y - yij;
        double z0 = start_position_z - zij;
        double gaussian_exponent_at_reference = aij * distance_squared(x0, y0, z0);
        double gaussian_starting_exponent = theta_ij * rr_ij + gaussian_exponent_at_reference;
        if (gaussian_starting_exponent > 680.) {
            continue;
        }
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
        double cc = ci * cj;
        if (ish_cell0 == jsh_cell0) {
            cc *= .5;
        }
        double gaussian_starting_point = exp(-gaussian_starting_exponent) * cc;
        double cross_term_a = c_dxyz_dabc[0] * x0 + c_dxyz_dabc[1] * y0 + c_dxyz_dabc[2] * z0;
        double cross_term_b = c_dxyz_dabc[3] * x0 + c_dxyz_dabc[4] * y0 + c_dxyz_dabc[5] * z0;
        double cross_term_c = c_dxyz_dabc[6] * x0 + c_dxyz_dabc[7] * y0 + c_dxyz_dabc[8] * z0;
        double recursion_factor_a_start = exp(-aij * (2 * cross_term_a + da_squared));
        double recursion_factor_b_start = exp(-aij * (2 * cross_term_b + db_squared));
        double recursion_factor_c_start = exp(-aij * (2 * cross_term_c + dc_squared));
        double exp_da_squared = exp(-2 * aij * da_squared);
        double exp_db_squared = exp(-2 * aij * db_squared);
        double exp_dc_squared = exp(-2 * aij * dc_squared);
        double exp_dadb = exp(-2 * aij * a_dot_b);
        double exp_dadc = exp(-2 * aij * a_dot_c);
        double exp_dbdc = exp(-2 * aij * b_dot_c);
#pragma unroll
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
#pragma unroll
        for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
            double vj_cache[SLICE_SIZE_I * SLICE_SIZE_J];
#pragma unroll
            for (int n = 0; n < SLICE_SIZE_I*SLICE_SIZE_J; ++n) {
                vj_cache[n] = 0;
            }

            double x, y, z;
            double recursion_factor_ab_pow_a = 1;
            double recursion_factor_ac_pow_a = 1;
            double recursion_factor_bc_pow_b = 1;

            if constexpr (is_non_orthogonal) {
            } else {
                x = start_position_x;
            }
            double gaussian_x = gaussian_starting_point;
            double recursion_factor_a = recursion_factor_a_start;
            for (int a_index = 0; a_index < a_upper; a_index++,
                 gaussian_x *= recursion_factor_a,
                 recursion_factor_a *= exp_da_squared) {

                if constexpr (is_non_orthogonal) {
                    recursion_factor_bc_pow_b = 1;
                } else {
                    y = start_position_y;
                }
                double gaussian_xy = gaussian_x;
                double recursion_factor_b = recursion_factor_b_start * recursion_factor_ab_pow_a;
                for (int b_index = 0; b_index < b_upper; b_index++,
                     gaussian_xy *= recursion_factor_b,
                     recursion_factor_b *= exp_db_squared) {

                    if constexpr (is_non_orthogonal) {
                        x = start_position_x + a_index * c_dxyz_dabc[0] + b_index * c_dxyz_dabc[3];
                        y = start_position_y + a_index * c_dxyz_dabc[1] + b_index * c_dxyz_dabc[4];
                        z = start_position_z + a_index * c_dxyz_dabc[2] + b_index * c_dxyz_dabc[5];
                    } else {
                        z = start_position_z;
                    }
                    double gaussian_xyz = gaussian_xy;
                    double recursion_factor_c = recursion_factor_c_start *
                            recursion_factor_ac_pow_a * recursion_factor_bc_pow_b;
                    for (int c_index = 0; c_index < c_upper; c_index++,
                         gaussian_xyz *= recursion_factor_c,
                         recursion_factor_c *= exp_dc_squared) {

                        if (fabs(gaussian_xyz) > negligible) {
                            double i_cartesian[nfi];
                            gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                            rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);

                            double j_cartesian[nfj];
                            gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                            rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);

                            int abc_index = a_index * TILE*TILE + b_index*TILE + c_index;
                            double fac = gaussian_xyz * vxc_cache[abc_index];
#pragma unroll
                            for (int i = 0; i < SLICE_SIZE_I; ++i) {
                                if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
                                double s = fac * i_cartesian[i];
#pragma unroll
                                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                    if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                                    vj_cache[i*SLICE_SIZE_J+j] += s * j_cartesian[j];
                                }
                            }
                        }
                        if constexpr (is_non_orthogonal) {
                            x += c_dxyz_dabc[6];
                            y += c_dxyz_dabc[7];
                            z += c_dxyz_dabc[8];
                        } else {
                            z += c_dxyz_dabc[8];
                        }
                    }
                    if constexpr (is_non_orthogonal) {
                        recursion_factor_bc_pow_b *= exp_dbdc;
                    } else {
                        y += c_dxyz_dabc[4];
                    }
                }
                if constexpr (is_non_orthogonal) {
                    recursion_factor_ab_pow_a *= exp_dadb;
                    recursion_factor_ac_pow_a *= exp_dadc;
                } else {
                    x += c_dxyz_dabc[0];
                }
            }

            size_t nao = envs.ao_loc[nbas];
            int i0 = envs.ao_loc[ish_cell0];
            int j0 = envs.ao_loc[jsh_cell0];
            double *pout = out + bvk_cell_id * nao * nao + (dm_i0+i0) * nao + dm_j0+j0;
#pragma unroll
            for (int i = 0; i < SLICE_SIZE_I; ++i) {
                if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
#pragma unroll
            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                atomicAdd(pout + i*nao+j, vj_cache[i*SLICE_SIZE_J+j]);
            } }
        } }
    }
}
}

extern "C" {
#define eval_lda_mat_kernel_case(li, lj, slice_i, slice_j, non_orth) \
    case (li * LMAX1 + lj): \
        eval_lda_mat_kernel<li,lj,slice_i,slice_j,non_orth><<<block_grid, THREADS>>>( \
            out, vxc, *envs, supmol_img_coords, shl_pair_offsets, dressed_bas_ij_idx, \
            grid_tile_index, n_contributing_tiles, tiles_per_block, \
            a_dot_b, a_dot_c, b_dot_c, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c, negligible); \
    break

int evaluate_lda_mat(double *out, double *vxc, double *placeholder, PBCIntEnvVars *envs,
                     double *dxyz_dabc, double *supmol_img_coords,
                     int i_angular, int j_angular, int tiles_per_block,
                     int *shl_pair_offsets, int64_t *dressed_bas_ij_idx,
                     int *grid_tile_index, int n_contributing_tiles, int *mesh,
                     double negligible)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    int block_grid = (n_contributing_tiles + tiles_per_block-1) / tiles_per_block;
    double a_dot_b = dxyz_dabc[0] * dxyz_dabc[3] + dxyz_dabc[1] * dxyz_dabc[4] + dxyz_dabc[2] * dxyz_dabc[5];
    double a_dot_c = dxyz_dabc[0] * dxyz_dabc[6] + dxyz_dabc[1] * dxyz_dabc[7] + dxyz_dabc[2] * dxyz_dabc[8];
    double b_dot_c = dxyz_dabc[3] * dxyz_dabc[6] + dxyz_dabc[4] * dxyz_dabc[7] + dxyz_dabc[5] * dxyz_dabc[8];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    switch (i_angular * LMAX1 + j_angular) {
        eval_lda_mat_kernel_case(0,0, 1, 1, 1);
        eval_lda_mat_kernel_case(1,0, 3, 1, 1);
        eval_lda_mat_kernel_case(1,1, 3, 3, 1);
        eval_lda_mat_kernel_case(2,0, 6, 1, 1);
        eval_lda_mat_kernel_case(2,1, 6, 3, 1);
        eval_lda_mat_kernel_case(2,2, 6, 6, 1);
        eval_lda_mat_kernel_case(3,0,10, 1, 1);
        eval_lda_mat_kernel_case(3,1,10, 3, 1);
        eval_lda_mat_kernel_case(3,2,10, 6, 1);
        eval_lda_mat_kernel_case(3,3, 5,10, 1);
        eval_lda_mat_kernel_case(4,0,15, 1, 1);
        eval_lda_mat_kernel_case(4,1,15, 3, 1);
        eval_lda_mat_kernel_case(4,2, 8, 6, 1);
        eval_lda_mat_kernel_case(4,3,15, 5, 1);
        eval_lda_mat_kernel_case(4,4,15, 5, 1);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
