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
void eval_density_kernel(double *density, double *dm, PBCIntEnvVars envs,
                         double *supmol_img_coords, double factor,
                         int *shl_pair_offsets, int64_t *dressed_bas_ij_idx,
                         int *grid_tile_index, int ntiles, int tiles_per_block,
                         double a_dot_b, double a_dot_c, double b_dot_c,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c, double negligible)
{
    constexpr int threads = THREADS;
    constexpr int WARPS = THREADS / WARP_SIZE;
    int thread_id = threadIdx.x;
    int tile_id0 = blockIdx.x * tiles_per_block;
    __shared__ int a_upper, b_upper, c_upper;
    __shared__ double start_position_x, start_position_y, start_position_z;
    __shared__ double density_value[TILE*TILE*TILE*WARPS];

    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;

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

    int lane = thread_id % WARP_SIZE;
    int warp = thread_id / WARP_SIZE;
    for (int n = thread_id; n < TILE*TILE*TILE*WARPS; n += threads) {
        density_value[n] = 0;
    }
    __syncthreads();

    for (int pair_id = shl_pair0+thread_id; pair_id < shl_pair1+thread_id; pair_id += threads) {
        int ish = 0;
        int jsh = 0;
        if (pair_id < shl_pair1) {
            int64_t bas_ij = dressed_bas_ij_idx[pair_id];
            ish = bas_ij / NBAS_MAX;
            jsh = bas_ij % NBAS_MAX;
        }
        int latsum_idx = ish / nbas;
        ish = ish - nbas * latsum_idx;
        int jL = jsh / bvk_nbas;
        jsh = jsh - bvk_nbas * jL;
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        double Lx = supmol_img_coords[latsum_idx*3+0];
        double Ly = supmol_img_coords[latsum_idx*3+1];
        double Lz = supmol_img_coords[latsum_idx*3+2];
        int expi = bas[ish_cell0*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh_cell0*BAS_SLOTS+PTR_EXP];
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
        double gaussian_starting_point = 0.;
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
        double recursion_factor_a_start = 0.;
        double recursion_factor_b_start = 0.;
        double recursion_factor_c_start = 0.;
        double exp_da_squared = 0.;
        double exp_db_squared = 0.;
        double exp_dc_squared = 0.;
        double exp_dadb = 0.;
        double exp_dadc = 0.;
        double exp_dbdc = 0.;
        if (gaussian_starting_exponent < 680.) {
            double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
            double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
            double cc = ci * cj;
            gaussian_starting_point = exp(-gaussian_starting_exponent);
            gaussian_starting_point *= factor * cc;
            if (ish == jsh_cell0) {
                gaussian_starting_point *= 0.5;
            }
            double cross_term_a = c_dxyz_dabc[0] * x0 + c_dxyz_dabc[1] * y0 + c_dxyz_dabc[2] * z0;
            double cross_term_b = c_dxyz_dabc[3] * x0 + c_dxyz_dabc[4] * y0 + c_dxyz_dabc[5] * z0;
            double cross_term_c = c_dxyz_dabc[6] * x0 + c_dxyz_dabc[7] * y0 + c_dxyz_dabc[8] * z0;
            recursion_factor_a_start = exp(-aij * (2 * cross_term_a + da_squared));
            recursion_factor_b_start = exp(-aij * (2 * cross_term_b + db_squared));
            recursion_factor_c_start = exp(-aij * (2 * cross_term_c + dc_squared));
            exp_da_squared = exp(-2 * aij * da_squared);
            exp_db_squared = exp(-2 * aij * db_squared);
            exp_dc_squared = exp(-2 * aij * dc_squared);
            exp_dadb = exp(-2 * aij * a_dot_b);
            exp_dadc = exp(-2 * aij * a_dot_c);
            exp_dbdc = exp(-2 * aij * b_dot_c);
        }
#pragma unroll
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
#pragma unroll
        for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
            uint32_t nao = envs.ao_loc[nbas];
            int i0 = envs.ao_loc[ish_cell0];
            int j0 = envs.ao_loc[jsh_cell0];
            uint32_t ij_offset = bvk_cell_id * nao * nao + (dm_i0+i0) * nao + dm_j0+j0;
            double dm_cache[SLICE_SIZE_I * SLICE_SIZE_J];
            if (pair_id < shl_pair1) {
#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
#pragma unroll
                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                    if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                    dm_cache[i*SLICE_SIZE_J+j] = dm[ij_offset + i*nao+j];
                } }
            }

            double x, y, z;
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
                double recursion_factor_b = recursion_factor_b_start;
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
                    double recursion_factor_c = recursion_factor_c_start * recursion_factor_bc_pow_b;
                    for (int c_index = 0; c_index < c_upper; c_index++,
                         gaussian_xyz *= recursion_factor_c,
                         recursion_factor_c *= exp_dc_squared) {

                        double val = 0;
                        if (pair_id < shl_pair1 && fabs(gaussian_xyz) > negligible) {
                            double i_cartesian[nfi];
                            gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                            rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);

                            double j_cartesian[nfj];
                            gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                            rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);
#pragma unroll
                            for (int i = 0; i < SLICE_SIZE_I; ++i) {
                                if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
                                double s = 0;
#pragma unroll
                                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                    if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                                    s += dm_cache[i * SLICE_SIZE_J + j] * j_cartesian[j];
                                }
                                val += s * i_cartesian[i];
                            }
                            val *= gaussian_xyz;
                        }
                        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                            val += __shfl_down_sync(0xffffffff, val, offset);
                        }
                        if (lane == 0) {
                            int abc_index = a_index * TILE*TILE + b_index*TILE + c_index;
                            density_value[abc_index+TILE*TILE*TILE*warp] += val;
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
                    recursion_factor_b_start *= exp_dadb;
                    recursion_factor_c_start *= exp_dadc;
                } else {
                    x += c_dxyz_dabc[0];
                }
            }
        } }
    }
    __syncthreads();

    int a_idx = a_start + thread_id / (TILE*TILE);
    int b_idx = b_start + thread_id / TILE % TILE;
    int c_idx = c_start + thread_id % TILE;
    if (a_idx < mesh_a && b_idx < mesh_b && c_idx < mesh_c) {
        double val = density_value[thread_id];
        for (int i = 1; i < WARPS; i++) {
            val += density_value[thread_id+i*TILE*TILE*TILE];
        }
        size_t abc_idx = (a_idx * mesh_b + b_idx) * (size_t)mesh_c + c_idx;
        // update the density.real, skip the imaginary part
        atomicAdd(density + abc_idx*2, val);
    }
    __syncthreads();
}
}

extern "C" {
#define eval_density_kernel_case(li, lj, slice_i, slice_j, non_orth) \
    case (li * LMAX1 + lj): \
        eval_density_kernel<li,lj,slice_i,slice_j,non_orth><<<block_grid, THREADS>>>( \
            density, dm, *envs, supmol_img_coords, factor, \
            shl_pair_offsets, dressed_bas_ij_idx, \
            grid_tile_index, n_contributing_tiles, tiles_per_block, \
            a_dot_b, a_dot_c, b_dot_c, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c, negligible); \
    break

int evaluate_density(double *density, double *placeholder,
                     double *dm, PBCIntEnvVars *envs,
                     double *dxyz_dabc, double *supmol_img_coords,
                     int i_angular, int j_angular, int tiles_per_block,
                     int *shl_pair_offsets, int64_t *dressed_bas_ij_idx,
                     int *grid_tile_index, int n_contributing_tiles, int *mesh,
                     double factor, double negligible)
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
        eval_density_kernel_case(0,0, 1, 1, 1);
        eval_density_kernel_case(1,0, 3, 1, 1);
        eval_density_kernel_case(1,1, 3, 3, 1);
        eval_density_kernel_case(2,0, 6, 1, 1);
        eval_density_kernel_case(2,1, 6, 3, 1);
        eval_density_kernel_case(2,2, 6, 6, 1);
        eval_density_kernel_case(3,0,10, 1, 1);
        eval_density_kernel_case(3,1,10, 3, 1);
        eval_density_kernel_case(3,2,10, 6, 1);
        eval_density_kernel_case(3,3,10, 5, 1);
        eval_density_kernel_case(4,0,15, 1, 1);
        eval_density_kernel_case(4,1,15, 3, 1);
        eval_density_kernel_case(4,2, 8, 6, 1);
        eval_density_kernel_case(4,3,15, 5, 1);
        eval_density_kernel_case(4,4,15, 5, 1);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_density_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
