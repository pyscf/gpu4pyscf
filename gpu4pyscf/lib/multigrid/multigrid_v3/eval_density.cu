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
#include "cartesian.cuh"

__constant__ double reciprocal_lattice_vectors[9];
__constant__ double lattice_vectors[9];
__constant__ double dxyz_dabc[9];
__constant__ double reciprocal_norm[3];

#define TILE            4
#define THREADS         256

template <int LI, int LJ, int DM_SLICE_SIZE, bool is_non_orthogonal>
__global__ static
void eval_density_kernel(double *density, double *dm, PBCIntEnvVars envs,
                         int *shl_pair_offsets, int2 *bas_ij_idx,
                         int *grid_tile_index, int ntiles,
                         int *bas_image_idx, int *Ts_ij_lookup,
                         double a_dot_b, double a_dot_c, double b_dot_c,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c)
{
    int nsp_per_block = blockDim.x;
    int tiles_per_block = blockDim.y;
    int threads = nsp_per_block * tiles_per_block;
    int sp_id = threadIdx.x;
    int tile_id_in_block = threadIdx.y;
    int thread_id = sp_id + nsp_per_block * tile_id_in_block;
    int tile_id = blockIdx.x * tiles_per_block + tile_id_in_block;

    int tile_index = 0;
    if (tile_id < ntiles) {
        tile_index = grid_tile_index[tile_id];
    }
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
    int nimgs = envs.nimgs;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int a_upper, b_upper, c_upper;
    __shared__ double start_position_x, start_position_y, start_position_z;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[tile_index];
        shl_pair1 = shl_pair_offsets[tile_index+1];
        start_position_x = dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
        start_position_y = dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
        start_position_z = dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;
        a_upper = min(a_start + TILE, mesh_a) - a_start;
        b_upper = min(b_start + TILE, mesh_b) - b_start;
        c_upper = min(c_start + TILE, mesh_c) - c_start;
    }

    extern __shared__ double density_value[];
    int valid_tiles = min(tiles_per_block, ntiles - blockIdx.x * tiles_per_block);
    for (int n = thread_id; n < TILE*TILE*TILE*valid_tiles; n += threads) {
        density_value[n] = 0;
    }
    __syncthreads();

    for (int pair_id = shl_pair0+sp_id; pair_id < shl_pair1+sp_id; pair_id += nsp_per_block) {
        int ish = 0;
        int jsh = 0;
        if (pair_id < shl_pair1) {
            int2 bas_ij = bas_ij_idx[pair_id];
            ish = bas_ij.x;
            jsh = bas_ij.y;
        }
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = env[ri+0];
        double yi = env[ri+1];
        double zi = env[ri+2];
        double xj = env[rj+0];
        double yj = env[rj+1];
        double zj = env[rj+2];
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
        double gaussian_starting_point = exp(-(theta_ij * rr_ij + gaussian_exponent_at_reference));
        double exp_da_squared = exp(-2 * aij * da_squared);
        double exp_db_squared = exp(-2 * aij * db_squared);
        double exp_dc_squared = exp(-2 * aij * dc_squared);
        double cross_term_a = dxyz_dabc[0] * x0 + dxyz_dabc[1] * y0 + dxyz_dabc[2] * z0;
        double cross_term_b = dxyz_dabc[3] * x0 + dxyz_dabc[4] * y0 + dxyz_dabc[5] * z0;
        double cross_term_c = dxyz_dabc[6] * x0 + dxyz_dabc[7] * y0 + dxyz_dabc[8] * z0;

        // BUG: when aij gets too large and x0 is negative and large, the
        // exponential can overflow and return inf.
        // ideally recursion should start from the nearest grid point to the pair
        // center, instead of the fixed recursion path
        // (min a, min b, min c) -> (max a, max b, max c)
        // The inf ususally occurs when pseudo-potential is not used,
        // and core electrons appear with large exponents.
        // Potentially another fix is to have a better designed multi-grid
        // structure, where the gaussians with large exponents are evaluated
        // on a more dense grid. Around the boundary the numbers should be
        // within the range of double precision.
        double recursion_factor_a_start = exp(-aij * (2 * cross_term_a + da_squared));
        double recursion_factor_b_start = exp(-aij * (2 * cross_term_b + db_squared));
        double recursion_factor_c_start = exp(-aij * (2 * cross_term_c + dc_squared));
        double exp_dadb = exp(-2 * aij * a_dot_b);
        double exp_dadc = exp(-2 * aij * a_dot_c);
        double exp_dbdc = exp(-2 * aij * b_dot_c);
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += DM_SLICE_SIZE) {
            double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
            double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
            double cc = ci * cj;
            size_t nao = envs.ao_loc[nbas];
            size_t nao2 = nao * nao;
            int iL = bas_image_idx[ish];
            int jL = bas_image_idx[jsh];
            int i0 = envs.ao_loc[ish];
            int j0 = envs.ao_loc[jsh];
            double *dm_image_shift = dm + Ts_ij_lookup[jL*nimgs+iL] * nao2;
            double dm_cache[DM_SLICE_SIZE * nfj];
#pragma unroll
            for (int i = 0; i < min(nfi, DM_SLICE_SIZE); ++i) {
                if (dm_i0 + i > nfi) break;
#pragma unroll
            for (int j = 0; j < nfj; ++j) {
                dm_cache[i*nfj+j] = 0;
                if (pair_id < shl_pair1) {
                    dm_cache[i*nfj+j] = dm_image_shift[(i0+dm_i0+i)*nao+j0+j] * cc;
                }
            } }

            double x, y, z;
            double recursion_factor_ab_pow_a = 1;
            double recursion_factor_ac_pow_a = 1;
            double recursion_factor_bc_pow_b = 1;

            if constexpr (is_non_orthogonal) {
                // recursion_factor_ab_pow_a = 1;
                // recursion_factor_ac_pow_a = 1;
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
                        x = start_position_x + a_index * dxyz_dabc[0] + b_index * dxyz_dabc[3];
                        y = start_position_y + a_index * dxyz_dabc[1] + b_index * dxyz_dabc[4];
                        z = start_position_z + a_index * dxyz_dabc[2] + b_index * dxyz_dabc[5];
                    } else {
                        z = start_position_z;
                    }
                    double gaussian_xyz = gaussian_xy;
                    double recursion_factor_c = recursion_factor_c_start *
                            recursion_factor_ac_pow_a * recursion_factor_bc_pow_b;
                    for (int c_index = 0; c_index < c_upper; c_index++,
                         gaussian_xyz *= recursion_factor_c,
                         recursion_factor_c *= exp_dc_squared) {

                        double i_cartesian[nfi];
                        double j_cartesian[nfj];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);

                        double val = 0;
#pragma unroll
                        for (int i = 0; i < min(nfi, DM_SLICE_SIZE); ++i) {
                            if (dm_i0 + i > nfi) break;
                            double s = 0;
#pragma unroll
                            for (int j = 0; j < nfj; j++) {
                                s += dm_cache[i * nfj + j] * j_cartesian[j];
                            }
                            val += s * i_cartesian[i];
                        }
                        for (int offset = nsp_per_block/2; offset > 0; offset >>= 1) {
                            val += __shfl_down_sync(0xffffffff, val, offset);
                        }
                        if (sp_id == 0) {
                            int abc_index = a_index * TILE*TILE + b_index*TILE + c_index;
                            density_value[tile_id_in_block*TILE*TILE*TILE+abc_index] += val * gaussian_xyz;
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
    }
    __syncthreads();

    for (int n = thread_id; n < TILE*TILE*TILE*valid_tiles; n += threads) {
        int tile_id = blockIdx.x * tiles_per_block + n / (TILE*TILE*TILE);
        int tile_index = grid_tile_index[tile_id];
        int tile_ab_index = tile_index / n_tiles_c;
        int tile_c_index = tile_index % n_tiles_c;
        int tile_a_index = tile_ab_index / n_tiles_b;
        int tile_b_index = tile_ab_index % n_tiles_b;
        int a_start = tile_a_index * TILE;
        int b_start = tile_b_index * TILE;
        int c_start = tile_c_index * TILE;
        int a_idx = a_start + n / (TILE*TILE) % TILE;
        int b_idx = b_start + n / TILE % TILE;
        int c_idx = c_start + n % TILE;
        atomicAdd(density + (a_idx * mesh_b + b_idx) * mesh_c + c_idx, density_value[n]);
    }
}

extern "C" {
#define eval_density_kernel_case(li, lj, dm_slice) \
    case (li * 10 + lj): \
        eval_density_kernel<li,lj,dm_slice,0><<<block_grid, threads, shm_size>>>( \
            density, dm, *envs, shl_pair_offsets, bas_ij_idx, \
            grid_tile_index, n_contributing_tiles, bas_image_idx, Ts_ij_lookup, \
            a_dot_b, a_dot_c, b_dot_c, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c); \
    break

int evaluate_density(double *density, double *dm, PBCIntEnvVars *envs,
                     double *dxyz_dabc,
                     int shm_size, int i_angular, int j_angular,
                     int *shl_pair_offsets, int2 *bas_ij_idx,
                     int *grid_tile_index, int n_contributing_tiles,
                     int tiles_per_block, int nsp_per_block,
                     int *bas_image_idx, int *Ts_ij_lookup, int *mesh)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    dim3 threads(TILE, tiles_per_block, nsp_per_block);
    int block_grid = (n_contributing_tiles + tiles_per_block-1) / tiles_per_block;
    double a_dot_b = dxyz_dabc[0] * dxyz_dabc[3] + dxyz_dabc[1] * dxyz_dabc[4] + dxyz_dabc[2] * dxyz_dabc[5];
    double a_dot_c = dxyz_dabc[0] * dxyz_dabc[6] + dxyz_dabc[1] * dxyz_dabc[7] + dxyz_dabc[2] * dxyz_dabc[8];
    double b_dot_c = dxyz_dabc[3] * dxyz_dabc[6] + dxyz_dabc[4] * dxyz_dabc[7] + dxyz_dabc[5] * dxyz_dabc[8];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    switch (i_angular * 10 + j_angular) {
        eval_density_kernel_case(0,0, 1);
        eval_density_kernel_case(1,0, 3);
        eval_density_kernel_case(1,1, 3);
        eval_density_kernel_case(2,0, 6);
        eval_density_kernel_case(2,1, 6);
        eval_density_kernel_case(2,2, 6);
        eval_density_kernel_case(3,0,10);
        eval_density_kernel_case(3,1,10);
        eval_density_kernel_case(3,2,10);
        eval_density_kernel_case(3,3, 5);
        eval_density_kernel_case(4,0,15);
        eval_density_kernel_case(4,1,15);
        eval_density_kernel_case(4,2,15);
        eval_density_kernel_case(4,3, 8);
        eval_density_kernel_case(4,4, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_density_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
