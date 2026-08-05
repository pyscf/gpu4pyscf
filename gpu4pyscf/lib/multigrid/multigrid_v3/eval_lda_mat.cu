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

#define TILE            4
#define WARP_SIZE       32
#define WARPS           8
#define THREADS         256

__device__ __forceinline__
double reduce(double val, double *swap, int thread_id)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = thread_id % WARP_SIZE;
    int warp = thread_id / WARP_SIZE;
    if (lane == 0) {
        swap[warp] = val;
    }
    __syncthreads();

    val = (thread_id < 8) ? swap[lane] : 0.;
    for (int offset = 4; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J, bool is_non_orthogonal>
__global__ static
void eval_lda_mat_kernel(double *out, double *vxc_weights, PBCIntEnvVars envs,
                         int *shl_pair_offsets, int64_t *bas_ij_idx,
                         int *grid_tile_idx, int ntiles, int *supmol_to_bvk_mapping,
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
        tile_index = grid_tile_idx[tile_id];
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
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int a_upper, b_upper, c_upper;
    __shared__ double start_position_x, start_position_y, start_position_z;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[tile_index];
        shl_pair1 = shl_pair_offsets[tile_index+1];
        start_position_x = c_dxyz_dabc[0] * a_start + c_dxyz_dabc[3] * b_start + c_dxyz_dabc[6] * c_start;
        start_position_y = c_dxyz_dabc[1] * a_start + c_dxyz_dabc[4] * b_start + c_dxyz_dabc[7] * c_start;
        start_position_z = c_dxyz_dabc[2] * a_start + c_dxyz_dabc[5] * b_start + c_dxyz_dabc[8] * c_start;
        a_upper = min(a_start + TILE, mesh_a) - a_start;
        b_upper = min(b_start + TILE, mesh_b) - b_start;
        c_upper = min(c_start + TILE, mesh_c) - c_start;
    }

    extern __shared__ double vxc_cache[];
    int valid_tiles = min(tiles_per_block, ntiles - blockIdx.x * tiles_per_block);
    for (int n = thread_id; n < TILE*TILE*TILE*valid_tiles; n += threads) {
        int tile_id = blockIdx.x * tiles_per_block + n / (TILE*TILE*TILE);
        int tile_index = grid_tile_idx[tile_id];
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
        vxc_cache[n] = vxc_weights[(a_idx * mesh_b + b_idx) * mesh_c + c_idx];
    }
    __syncthreads();

    for (int pair_id = shl_pair0+sp_id; pair_id < shl_pair1+sp_id; pair_id += nsp_per_block) {
        int ish = 0;
        int jsh = 0;
        if (pair_id < shl_pair1) {
            int64_t bas_ij = bas_ij_idx[pair_id];
            ish = bas_ij / NBAS_MAX;
            jsh = bas_ij % NBAS_MAX;
        }
        int latsum_idx = ish / nbas;
        int ish_cell0 = ish - nbas * latsum_idx;
        int jL = jsh / nbas;
        int jsh_cell0 = jsh - nbas * jL;
        double Lx = envs.img_coords[latsum_idx*3+0];
        double Ly = envs.img_coords[latsum_idx*3+1];
        double Lz = envs.img_coords[latsum_idx*3+2];
        int expi = bas[ish_cell0*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh_cell0*BAS_SLOTS+PTR_EXP];
        int ri = bas[ish_cell0*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh_cell0*BAS_SLOTS+PTR_BAS_COORD];
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
        double gaussian_starting_point = exp(-(theta_ij * rr_ij + gaussian_exponent_at_reference));
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
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
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

                        double i_cartesian[nfi];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi, dm_i0);
                        if (SLICE_SIZE_I < nfi) {
                            if (dm_i0 == 1) {
#pragma unroll
                                for (int i = 0; i < min(SLICE_SIZE_I, nfi-SLICE_SIZE_I); i++) {
                                    i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I];
                                }
                            } else if (dm_i0 == 2) {
#pragma unroll
                                for (int i = 0; i < nfi-SLICE_SIZE_I*2; i++) {
                                    i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I*2];
                                }
                            }
                        }

                        double j_cartesian[nfj];
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj, dm_j0);
                        if (SLICE_SIZE_J < nfj) {
                            if (dm_j0 == 1) {
#pragma unroll
                                for (int i = 0; i < min(SLICE_SIZE_J, nfj-SLICE_SIZE_J); i++) {
                                    j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J];
                                }
                            } else if (dm_j0 == 2) {
#pragma unroll
                                for (int i = 0; i < nfj-SLICE_SIZE_J*2; i++) {
                                    j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J*2];
                                }
                            }
                        }

                        int abc_index = a_index * TILE*TILE + b_index*TILE + c_index;
                        double fac = gaussian_xyz * vxc_cache[abc_index];
#pragma unroll
                        for (int i = 0; i < SLICE_SIZE_I; ++i) {
                            if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
                            double s = fac * i_cartesian[i];
#pragma unroll
                            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                                vj_cache[i*nfj+j] += s * j_cartesian[j];
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

            if (pair_id < shl_pair1) {
                double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
                double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
                double cc = ci * cj;
                size_t nao = envs.ao_loc[nbas];
                size_t nao2 = nao * nao;
                int i0 = envs.ao_loc[ish_cell0];
                int j0 = envs.ao_loc[jsh_cell0];
                double *vj = out + supmol_to_bvk_mapping[jL] * nao2;
#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
#pragma unroll
                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                    if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                    atomicAdd(vj + (i0+dm_i0+i)*nao+j0+j, vj_cache[i*nfj+j] * cc);
                } }
            }
        } }
    }
}

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J>
__global__ static
void eval_lda_mat_kernel_v2(double *out, double *vxc_weights, PBCIntEnvVars envs,
                         int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                         int *supmol_to_bvk_mapping,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c, int npairs)
{
    constexpr int tile = 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * tile + tx;
    int pair_id = blockIdx.x;

    __shared__ int a_start, a_stop, a_center;
    __shared__ int b_start, b_stop;
    __shared__ int c_start, c_stop;
    __shared__ double exp_da_squared, inv_exp_da_squared;
    __shared__ double xi, yi, zi;
    __shared__ double xj, yj, zj;
    __shared__ double xij, yij, zij, aij, theta_rr;
    __shared__ double swap[WARPS];

    int *bas = envs.bas;
    int nbas = envs.nbas;
    double *env = envs.env;
    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int jL = jsh / nbas;
    int ish_cell0 = ish;
    int jsh_cell0 = jsh - nbas * jL;
    if (thread_id == 0) {
        int expi = bas[ish_cell0*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh_cell0*BAS_SLOTS+PTR_EXP];
        int ri = bas[ish_cell0*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh_cell0*BAS_SLOTS+PTR_BAS_COORD];
        double ai = env[expi];
        double aj = env[expj];
        aij = ai + aj;
        double aj_aij = aj / aij;
        xi = env[ri+0];
        yi = env[ri+1];
        zi = env[ri+2];
        xj = env[rj+0] + envs.img_coords[jL*3+0];
        yj = env[rj+1] + envs.img_coords[jL*3+1];
        zj = env[rj+2] + envs.img_coords[jL*3+2];
        double xjxi = xj - xi;
        double yjyi = yj - yi;
        double zjzi = zj - zi;
        double rr = distance_squared(xjxi, yjyi, zjzi);
        xij = xjxi * aj_aij + xi;
        yij = yjyi * aj_aij + yi;
        zij = zjzi * aj_aij + zi;
        theta_rr = ai * aj_aij * rr;

        float2 range = grid_frac_ranges[pair_id];
        float xfrac_lower = range.x;
        float xfrac_upper = range.y;
        range = grid_frac_ranges[npairs+pair_id];
        float yfrac_lower = range.x;
        float yfrac_upper = range.y;
        range = grid_frac_ranges[npairs*2+pair_id];
        float zfrac_lower = range.x;
        float zfrac_upper = range.y;
        a_start = ceil (xfrac_lower * mesh_a);
        a_stop  = floor(xfrac_upper * mesh_a);
        b_start = ceil (yfrac_lower * mesh_b);
        b_stop  = floor(yfrac_upper * mesh_b);
        c_start = ceil (zfrac_lower * mesh_c);
        c_stop  = floor(zfrac_upper * mesh_c);
        a_center = (a_start + a_stop) / 2;
        exp_da_squared = exp(-2 * aij * da_squared);
        inv_exp_da_squared = 1./exp_da_squared;
    }
    __syncthreads();

    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;

    for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
    for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
        double vj_cache[SLICE_SIZE_I * SLICE_SIZE_J];
#pragma unroll
        for (int n = 0; n < SLICE_SIZE_I*SLICE_SIZE_J; ++n) {
            vj_cache[n] = 0;
        }

        for (int b_index = b_start+ty; b_index < b_stop; b_index += tile) {
        for (int c_index = c_start+tx; c_index < c_stop; c_index += tile) {
            double x = a_center * c_dxyz_dabc[0] + b_index * c_dxyz_dabc[3] + c_index * c_dxyz_dabc[6];
            double y = a_center * c_dxyz_dabc[1] + b_index * c_dxyz_dabc[4] + c_index * c_dxyz_dabc[7];
            double z = a_center * c_dxyz_dabc[2] + b_index * c_dxyz_dabc[5] + c_index * c_dxyz_dabc[8];
            double x_xij = x - xij;
            double y_yij = y - yij;
            double z_zij = z - zij;
            double gaussian_starting_point = exp(-theta_rr - aij * distance_squared(x_xij, y_yij, z_zij));
            double cross_term_a = c_dxyz_dabc[0] * x_xij + c_dxyz_dabc[1] * y_yij + c_dxyz_dabc[2] * z_zij;
            double recursion_factor_a = exp(-aij * (2 * cross_term_a + da_squared));

            int bc_idx = (b_index % mesh_b) * mesh_c + (c_index % mesh_c);
            int mesh_bc = mesh_b * mesh_c;
            double gaussian_xyz = gaussian_starting_point;
            for (int a_index = a_center; a_index < a_stop; a_index++,
                 gaussian_xyz *= recursion_factor_a,
                 recursion_factor_a *= exp_da_squared) {
                if (gaussian_xyz < 1e-18) break;

                double v = vxc_weights[(a_index % mesh_a) * mesh_bc + bc_idx];
                double i_cartesian[nfi];
                gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi, dm_i0);
                if (SLICE_SIZE_I < nfi) {
                    if (dm_i0 == 1) {
#pragma unroll
                        for (int i = 0; i < min(SLICE_SIZE_I, nfi-SLICE_SIZE_I); i++) {
                            i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I];
                        }
                    } else if (dm_i0 == 2) {
#pragma unroll
                        for (int i = 0; i < nfi-SLICE_SIZE_I*2; i++) {
                            i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I*2];
                        }
                    }
                }

                double j_cartesian[nfj];
                gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj, dm_j0);
                if (SLICE_SIZE_J < nfj) {
                    if (dm_j0 == 1) {
#pragma unroll
                        for (int i = 0; i < min(SLICE_SIZE_J, nfj-SLICE_SIZE_J); i++) {
                            j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J];
                        }
                    } else if (dm_j0 == 2) {
#pragma unroll
                        for (int i = 0; i < nfj-SLICE_SIZE_J*2; i++) {
                            j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J*2];
                        }
                    }
                }

#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
                    double s = v * i_cartesian[i];
#pragma unroll
                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                    if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                        vj_cache[i*nfj+j] += s * j_cartesian[j];
                    }
                }
                x += c_dxyz_dabc[0];
                y += c_dxyz_dabc[1];
                z += c_dxyz_dabc[2];
            }

            x -= c_dxyz_dabc[0] * (a_stop - a_center);
            y -= c_dxyz_dabc[1] * (a_stop - a_center);
            z -= c_dxyz_dabc[2] * (a_stop - a_center);
            recursion_factor_a = 1./recursion_factor_a;
            gaussian_xyz = gaussian_starting_point;
            for (int a_index = a_center - 1; a_index >= a_start; a_index--,
                 recursion_factor_a *= inv_exp_da_squared) {
                gaussian_xyz *= recursion_factor_a;
                if (gaussian_xyz < 1e-18) break;

                double v = vxc_weights[(a_index % mesh_a) * mesh_bc + bc_idx];
                double i_cartesian[nfi];
                gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi, dm_i0);
                if (SLICE_SIZE_I < nfi) {
                    if (dm_i0 == 1) {
#pragma unroll
                        for (int i = 0; i < min(SLICE_SIZE_I, nfi-SLICE_SIZE_I); i++) {
                            i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I];
                        }
                    } else if (dm_i0 == 2) {
#pragma unroll
                        for (int i = 0; i < nfi-SLICE_SIZE_I*2; i++) {
                            i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I*2];
                        }
                    }
                }

                double j_cartesian[nfj];
                gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj, dm_j0);
                if (SLICE_SIZE_J < nfj) {
                    if (dm_j0 == 1) {
#pragma unroll
                        for (int i = 0; i < min(SLICE_SIZE_J, nfj-SLICE_SIZE_J); i++) {
                            j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J];
                        }
                    } else if (dm_j0 == 2) {
#pragma unroll
                        for (int i = 0; i < nfj-SLICE_SIZE_J*2; i++) {
                            j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J*2];
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
                    double s = v * i_cartesian[i];
#pragma unroll
                    for (int j = 0; j < SLICE_SIZE_J; ++j) {
                        if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                        vj_cache[i*nfj+j] += s * j_cartesian[j];
                    }
                }
                x += c_dxyz_dabc[0];
                y += c_dxyz_dabc[1];
                z += c_dxyz_dabc[2];
            }
        } }

        double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
        double cc = ci * cj;
        size_t nao = envs.ao_loc[nbas];
        size_t nao2 = nao * nao;
        int i0 = envs.ao_loc[ish_cell0];
        int j0 = envs.ao_loc[jsh_cell0];
        double *vj = out + supmol_to_bvk_mapping[jL] * nao2;
#pragma unroll
        for (int i = 0; i < SLICE_SIZE_I; ++i) {
            if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
#pragma unroll
            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                double val = reduce(vj_cache[i*nfj+j], swap, thread_id);
                if (thread_id == 0) {
                    atomicAdd(vj + (i0+dm_i0+i)*nao+j0+j, vj_cache[i*nfj+j] * cc);
                }
            }
        }
    } }
}

extern "C" {
#define eval_lda_mat_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_lda_mat_kernel<li,lj,slice_i,slice_j,0><<<block_grid, threads, shm_size>>>( \
            out, vxc_weights, *envs, shl_pair_offsets, bas_ij_idx, \
            grid_tile_idx, n_contributing_tiles, supmol_to_bvk_mapping, \
            a_dot_b, a_dot_c, b_dot_c, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c); \
    break

int evaluate_lda_mat(double *out, double *vxc_weights, PBCIntEnvVars *envs,
                     double *dxyz_dabc,
                     int tiles_per_block, int nsp_per_block,
                     int shm_size, int i_angular, int j_angular,
                     int *shl_pair_offsets, int64_t *bas_ij_idx,
                     int *grid_tile_idx, int *supmol_to_bvk_mapping,
                     int n_contributing_tiles, int *mesh)
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
    switch (i_angular * LMAX1 + j_angular) {
        eval_lda_mat_kernel_case(0,0, 1, 1);
        eval_lda_mat_kernel_case(1,0, 3, 1);
        eval_lda_mat_kernel_case(1,1, 3, 3);
        eval_lda_mat_kernel_case(2,0, 6, 1);
        eval_lda_mat_kernel_case(2,1, 6, 3);
        eval_lda_mat_kernel_case(2,2, 6, 6);
        eval_lda_mat_kernel_case(3,0,10, 1);
        eval_lda_mat_kernel_case(3,1,10, 3);
        eval_lda_mat_kernel_case(3,2,10, 6);
        eval_lda_mat_kernel_case(3,3, 5,10);
        eval_lda_mat_kernel_case(4,0,15, 1);
        eval_lda_mat_kernel_case(4,1,15, 3);
        eval_lda_mat_kernel_case(4,2,15, 6);
        eval_lda_mat_kernel_case(4,3,15, 5);
        eval_lda_mat_kernel_case(4,4, 5,15);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

#define eval_lda_mat_kernel_v2_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_lda_mat_kernel_v2<li,lj,slice_i,slice_j><<<npairs, threads>>>( \
            out, vxc_weights, *envs, bas_ij_idx, grid_frac_ranges, \
            supmol_to_bvk_mapping, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c, npairs); \
    break

int evaluate_lda_mat_v2(double *out, double *vxc_weights, PBCIntEnvVars *envs,
                     double *dxyz_dabc, int li, int lj, int64_t *bas_ij_idx,
                     float2 *grid_frac_ranges, int *supmol_to_bvk_mapping,
                     int *mesh, int npairs)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    dim3 threads(16, 16);
    switch (li * LMAX1 + lj) {
        eval_lda_mat_kernel_v2_case(0,0, 1, 1);
        eval_lda_mat_kernel_v2_case(1,0, 3, 1);
        eval_lda_mat_kernel_v2_case(1,1, 3, 3);
        eval_lda_mat_kernel_v2_case(2,0, 6, 1);
        eval_lda_mat_kernel_v2_case(2,1, 6, 3);
        eval_lda_mat_kernel_v2_case(2,2, 6, 6);
        eval_lda_mat_kernel_v2_case(3,0,10, 1);
        eval_lda_mat_kernel_v2_case(3,1,10, 3);
        eval_lda_mat_kernel_v2_case(3,2,10, 6);
        eval_lda_mat_kernel_v2_case(3,3, 5,10);
        eval_lda_mat_kernel_v2_case(4,0,15, 1);
        eval_lda_mat_kernel_v2_case(4,1,15, 3);
        eval_lda_mat_kernel_v2_case(4,2,15, 6);
        eval_lda_mat_kernel_v2_case(4,3,15, 5);
        eval_lda_mat_kernel_v2_case(4,4, 5,15);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
