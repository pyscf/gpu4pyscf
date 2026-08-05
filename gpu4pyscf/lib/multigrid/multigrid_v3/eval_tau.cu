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
#define THREADS         256

template <int ANG> __forceinline__ __device__
void gto_deriv1(double gradient_values[], double original_values[],
                double fx, double fy, double fz, double exponent, int start)
{
    double a2 = -2 * exponent;
    double minus_2afx = a2 * fx;
    double minus_2afy = a2 * fy;
    double minus_2afz = a2 * fz;
    if constexpr (ANG == 0) {
        // For s orbital (ANG=0), f(x,y,z) = 1
        // f'_x = 0, so g_x = -2 * exponent * fx
        gradient_values[0] = minus_2afx; // x gradient
        gradient_values[1] = minus_2afy; // y gradient
        gradient_values[2] = minus_2afz; // z gradient
    } else if constexpr (ANG == 1) {
        // For p orbitals (ANG=1), f(x,y,z) = {x, y, z}
        // First row: x gradient
        gradient_values[0] = 1 + minus_2afx * fx; // d/dx(x) - 2*exponent*x*fx
        gradient_values[1] = minus_2afx * fy;     // d/dx(y) - 2*exponent*y*fx
        gradient_values[2] = minus_2afx * fz;     // d/dx(z) - 2*exponent*z*fx
        // Second row: y gradient
        gradient_values[3] = minus_2afy * fx;     // d/dy(x) - 2*exponent*x*fy
        gradient_values[4] = 1 + minus_2afy * fy; // d/dy(y) - 2*exponent*y*fy
        gradient_values[5] = minus_2afy * fz;     // d/dy(z) - 2*exponent*z*fy
        // Third row: z gradient
        gradient_values[6] = minus_2afz * fx;     // d/dz(x) - 2*exponent*x*fz
        gradient_values[7] = minus_2afz * fy;     // d/dz(y) - 2*exponent*y*fz
        gradient_values[8] = 1 + minus_2afz * fz; // d/dz(z) - 2*exponent*z*fz
    } else if constexpr (ANG == 2) {
        // For d orbitals (ANG=2), f(x,y,z) = {xx, xy, xz, yy, yz, zz}
        // First row: x gradient
        gradient_values[0] =
            2 * fx + minus_2afx * original_values[0]; // d/dx(xx) - 2*exponent*xx*fx
        gradient_values[1] =
            fy + minus_2afx * original_values[1]; // d/dx(xy) - 2*exponent*xy*fx
        gradient_values[2] =
            fz + minus_2afx * original_values[2]; // d/dx(xz) - 2*exponent*xz*fx
        gradient_values[3] =
            minus_2afx * original_values[3]; // d/dx(yy) - 2*exponent*yy*fx
        gradient_values[4] =
            minus_2afx * original_values[4]; // d/dx(yz) - 2*exponent*yz*fx
        gradient_values[5] =
            minus_2afx * original_values[5]; // d/dx(zz) - 2*exponent*zz*fx
        // Second row: y gradient
        gradient_values[6] =
            minus_2afy * original_values[0]; // d/dy(xx) - 2*exponent*xx*fy
        gradient_values[7] =
            fx + minus_2afy * original_values[1]; // d/dy(xy) - 2*exponent*xy*fy
        gradient_values[8] =
            minus_2afy * original_values[2]; // d/dy(xz) - 2*exponent*xz*fy
        gradient_values[9] =
            2 * fy + minus_2afy * original_values[3]; // d/dy(yy) - 2*exponent*yy*fy
        gradient_values[10] =
            fz + minus_2afy * original_values[4]; // d/dy(yz) - 2*exponent*yz*fy
        gradient_values[11] =
            minus_2afy * original_values[5]; // d/dy(zz) - 2*exponent*zz*fy
        // Third row: z gradient
        gradient_values[12] =
            minus_2afz * original_values[0]; // d/dz(xx) - 2*exponent*xx*fz
        gradient_values[13] =
            minus_2afz * original_values[1]; // d/dz(xy) - 2*exponent*xy*fz
        gradient_values[14] =
            fx + minus_2afz * original_values[2]; // d/dz(xz) - 2*exponent*xz*fz
        gradient_values[15] =
            minus_2afz * original_values[3]; // d/dz(yy) - 2*exponent*yy*fz
        gradient_values[16] =
            fy + minus_2afz * original_values[4]; // d/dz(yz) - 2*exponent*yz*fz
        gradient_values[17] =
            2 * fz + minus_2afz * original_values[5]; // d/dz(zz) - 2*exponent*zz*fz
    } else if constexpr (ANG == 3) {
        // For f orbitals (ANG=3), f(x,y,z) = {xxx, xxy, xxz, xyy, xyz, xzz, yyy,
        // yyz, yzz, zzz}
        if (start < 5) {
            // First row: x gradient
            gradient_values[0] =
                3 * fx * fx +
                minus_2afx * original_values[0]; // d/dx(xxx) - 2*exponent*xxx*fx
            gradient_values[1] =
                2 * fx * fy +
                minus_2afx * original_values[1]; // d/dx(xxy) - 2*exponent*xxy*fx
            gradient_values[2] =
                2 * fx * fz +
                minus_2afx * original_values[2]; // d/dx(xxz) - 2*exponent*xxz*fx
            gradient_values[3] =
                fy * fy +
                minus_2afx * original_values[3]; // d/dx(xyy) - 2*exponent*xyy*fx
            gradient_values[4] =
                fy * fz +
                minus_2afx * original_values[4]; // d/dx(xyz) - 2*exponent*xyz*fx
            // Second row: y gradient
            gradient_values[10] =
                minus_2afy * original_values[0]; // d/dy(xxx) - 2*exponent*xxx*fy
            gradient_values[11] =
                fx * fx +
                minus_2afy * original_values[1]; // d/dy(xxy) - 2*exponent*xxy*fy
            gradient_values[12] =
                minus_2afy * original_values[2]; // d/dy(xxz) - 2*exponent*xxz*fy
            gradient_values[13] =
                2 * fx * fy +
                minus_2afy * original_values[3]; // d/dy(xyy) - 2*exponent*xyy*fy
            gradient_values[14] =
                fx * fz +
                minus_2afy * original_values[4]; // d/dy(xyz) - 2*exponent*xyz*fy
            // Third row: z gradient
            gradient_values[20] =
                minus_2afz * original_values[0]; // d/dz(xxx) - 2*exponent*xxx*fz
            gradient_values[21] =
                minus_2afz * original_values[1]; // d/dz(xxy) - 2*exponent*xxy*fz
            gradient_values[22] =
                fx * fx +
                minus_2afz * original_values[2]; // d/dz(xxz) - 2*exponent*xxz*fz
            gradient_values[23] =
                minus_2afz * original_values[3]; // d/dz(xyy) - 2*exponent*xyy*fz
            gradient_values[24] =
                fx * fy +
                minus_2afz * original_values[4]; // d/dz(xyz) - 2*exponent*xyz*fz
        }
        // First row: x gradient
        gradient_values[5] =
            fz * fz +
            minus_2afx * original_values[5]; // d/dx(xzz) - 2*exponent*xzz*fx
        gradient_values[6] =
            minus_2afx * original_values[6]; // d/dx(yyy) - 2*exponent*yyy*fx
        gradient_values[7] =
            minus_2afx * original_values[7]; // d/dx(yyz) - 2*exponent*yyz*fx
        gradient_values[8] =
            minus_2afx * original_values[8]; // d/dx(yzz) - 2*exponent*yzz*fx
        gradient_values[9] =
            minus_2afx * original_values[9]; // d/dx(zzz) - 2*exponent*zzz*fx
        // Second row: y gradient
        gradient_values[15] =
            minus_2afy * original_values[5]; // d/dy(xzz) - 2*exponent*xzz*fy
        gradient_values[16] =
            3 * fy * fy +
            minus_2afy * original_values[6]; // d/dy(yyy) - 2*exponent*yyy*fy
        gradient_values[17] =
            2 * fy * fz +
            minus_2afy * original_values[7]; // d/dy(yyz) - 2*exponent*yyz*fy
        gradient_values[18] =
            fz * fz +
            minus_2afy * original_values[8]; // d/dy(yzz) - 2*exponent*yzz*fy
        gradient_values[19] =
            minus_2afy * original_values[9]; // d/dy(zzz) - 2*exponent*zzz*fy
        // Third row: z gradient
        gradient_values[25] =
            2 * fx * fz +
            minus_2afz * original_values[5]; // d/dz(xzz) - 2*exponent*xzz*fz
        gradient_values[26] =
            minus_2afz * original_values[6]; // d/dz(yyy) - 2*exponent*yyy*fz
        gradient_values[27] =
            fy * fy +
            minus_2afz * original_values[7]; // d/dz(yyz) - 2*exponent*yyz*fz
        gradient_values[28] =
            2 * fy * fz +
            minus_2afz * original_values[8]; // d/dz(yzz) - 2*exponent*yzz*fz
        gradient_values[29] =
            3 * fz * fz +
            minus_2afz * original_values[9]; // d/dz(zzz) - 2*exponent*zzz*fz
    } else if constexpr (ANG == 4) {
        // For g orbitals (ANG=4), f(x,y,z) = {xxxx, xxxy, xxxz, xxyy, xxyz, xxzz,
        // xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz}
        if (start < 5) {
            // First row: x gradient
            gradient_values[0] =
                4 * fx * fx * fx +
                minus_2afx * original_values[0]; // d/dx(xxxx) - 2*exponent*xxxx*fx
            gradient_values[1] =
                3 * fx * fx * fy +
                minus_2afx * original_values[1]; // d/dx(xxxy) - 2*exponent*xxxy*fx
            gradient_values[2] =
                3 * fx * fx * fz +
                minus_2afx * original_values[2]; // d/dx(xxxz) - 2*exponent*xxxz*fx
            gradient_values[3] =
                2 * fx * fy * fy +
                minus_2afx * original_values[3]; // d/dx(xxyy) - 2*exponent*xxyy*fx
            gradient_values[4] =
                2 * fx * fy * fz +
                minus_2afx * original_values[4]; // d/dx(xxyz) - 2*exponent*xxyz*fx
            // Second row: y gradient
            gradient_values[15] =
                minus_2afy * original_values[0]; // d/dy(xxxx) - 2*exponent*xxxx*fy
            gradient_values[16] =
                fx * fx * fx +
                minus_2afy * original_values[1]; // d/dy(xxxy) - 2*exponent*xxxy*fy
            gradient_values[17] =
                minus_2afy * original_values[2]; // d/dy(xxxz) - 2*exponent*xxxz*fy
            gradient_values[18] =
                2 * fx * fx * fy +
                minus_2afy * original_values[3]; // d/dy(xxyy) - 2*exponent*xxyy*fy
            gradient_values[19] =
                fx * fx * fz +
                minus_2afy * original_values[4]; // d/dy(xxyz) - 2*exponent*xxyz*fy
            // Third row: z gradient
            gradient_values[30] =
                minus_2afz * original_values[0]; // d/dz(xxxx) - 2*exponent*xxxx*fz
            gradient_values[31] =
                minus_2afz * original_values[1]; // d/dz(xxxy) - 2*exponent*xxxy*fz
            gradient_values[32] =
                fx * fx * fx +
                minus_2afz * original_values[2]; // d/dz(xxxz) - 2*exponent*xxxz*fz
            gradient_values[33] =
                minus_2afz * original_values[3]; // d/dz(xxyy) - 2*exponent*xxyy*fz
            gradient_values[34] =
                fx * fx * fy +
                minus_2afz * original_values[4]; // d/dz(xxyz) - 2*exponent*xxyz*fz
        }
        if (start < 10) {
            // First row: x gradient
            gradient_values[5] =
                2 * fx * fz * fz +
                minus_2afx * original_values[5]; // d/dx(xxzz) - 2*exponent*xxzz*fx
            gradient_values[6] =
                fy * fy * fy +
                minus_2afx * original_values[6]; // d/dx(xyyy) - 2*exponent*xyyy*fx
            gradient_values[7] =
                fy * fy * fz +
                minus_2afx * original_values[7]; // d/dx(xyyz) - 2*exponent*xyyz*fx
            gradient_values[8] =
                fy * fz * fz +
                minus_2afx * original_values[8]; // d/dx(xyzz) - 2*exponent*xyzz*fx
            gradient_values[9] =
                fz * fz * fz +
                minus_2afx * original_values[9]; // d/dx(xzzz) - 2*exponent*xzzz*fx
            // Second row: y gradient
            gradient_values[20] =
                minus_2afy * original_values[5]; // d/dy(xxzz) - 2*exponent*xxzz*fy
            gradient_values[21] =
                3 * fx * fy * fy +
                minus_2afy * original_values[6]; // d/dy(xyyy) - 2*exponent*xyyy*fy
            gradient_values[22] =
                2 * fx * fy * fz +
                minus_2afy * original_values[7]; // d/dy(xyyz) - 2*exponent*xyyz*fy
            gradient_values[23] =
                fx * fz * fz +
                minus_2afy * original_values[8]; // d/dy(xyzz) - 2*exponent*xyzz*fy
            gradient_values[24] =
                minus_2afy * original_values[9]; // d/dy(xzzz) - 2*exponent*xzzz*fy
            // Third row: z gradient
            gradient_values[35] =
                2 * fx * fx * fz +
                minus_2afz * original_values[5]; // d/dz(xxzz) - 2*exponent*xxzz*fz
            gradient_values[36] =
                minus_2afz * original_values[6]; // d/dz(xyyy) - 2*exponent*xyyy*fz
            gradient_values[37] =
                fx * fy * fy +
                minus_2afz * original_values[7]; // d/dz(xyyz) - 2*exponent*xyyz*fz
            gradient_values[38] =
                2 * fx * fy * fz +
                minus_2afz * original_values[8]; // d/dz(xyzz) - 2*exponent*xyzz*fz
            gradient_values[39] =
                3 * fx * fz * fz +
                minus_2afz * original_values[9]; // d/dz(xzzz) - 2*exponent*xzzz*fz
        }
        // First row: x gradient
        gradient_values[10] =
            minus_2afx * original_values[10]; // d/dx(yyyy) - 2*exponent*yyyy*fx
        gradient_values[11] =
            minus_2afx * original_values[11]; // d/dx(yyyz) - 2*exponent*yyyz*fx
        gradient_values[12] =
            minus_2afx * original_values[12]; // d/dx(yyzz) - 2*exponent*yyzz*fx
        gradient_values[13] =
            minus_2afx * original_values[13]; // d/dx(yzzz) - 2*exponent*yzzz*fx
        gradient_values[14] =
            minus_2afx * original_values[14]; // d/dx(zzzz) - 2*exponent*zzzz*fx
        // Second row: y gradient
        gradient_values[25] =
            4 * fy * fy * fy +
            minus_2afy * original_values[10]; // d/dy(yyyy) - 2*exponent*yyyy*fy
        gradient_values[26] =
            3 * fy * fy * fz +
            minus_2afy * original_values[11]; // d/dy(yyyz) - 2*exponent*yyyz*fy
        gradient_values[27] =
            2 * fy * fz * fz +
            minus_2afy * original_values[12]; // d/dy(yyzz) - 2*exponent*yyzz*fy
        gradient_values[28] =
            fz * fz * fz +
            minus_2afy * original_values[13]; // d/dy(yzzz) - 2*exponent*yzzz*fy
        gradient_values[29] =
            minus_2afy * original_values[14]; // d/dy(zzzz) - 2*exponent*zzzz*fy
        // Third row: z gradient
        gradient_values[40] =
            minus_2afz * original_values[10]; // d/dz(yyyy) - 2*exponent*yyyy*fz
        gradient_values[41] =
            fy * fy * fy +
            minus_2afz * original_values[11]; // d/dz(yyyz) - 2*exponent*yyyz*fz
        gradient_values[42] =
            2 * fy * fy * fz +
            minus_2afz * original_values[12]; // d/dz(yyzz) - 2*exponent*yyzz*fz
        gradient_values[43] =
            3 * fy * fz * fz +
            minus_2afz * original_values[13]; // d/dz(yzzz) - 2*exponent*yzzz*fz
        gradient_values[44] =
            4 * fz * fz * fz +
            minus_2afz * original_values[14]; // d/dz(zzzz) - 2*exponent*zzzz*fz
    }
}

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J, bool is_non_orthogonal>
__global__ static
void eval_tau_kernel(double *tau, double *dm, PBCIntEnvVars envs,
                         int *shl_pair_offsets, int64_t *bas_ij_idx,
                         int *grid_tile_index, int ntiles, int *supmol_to_bvk_mapping,
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

    extern __shared__ double tau_value[];
    int valid_tiles = min(tiles_per_block, ntiles - blockIdx.x * tiles_per_block);
    for (int n = thread_id; n < TILE*TILE*TILE*valid_tiles; n += threads) {
        tau_value[n] = 0;
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
        double exp_da_squared = exp(-2 * aij * da_squared);
        double exp_db_squared = exp(-2 * aij * db_squared);
        double exp_dc_squared = exp(-2 * aij * dc_squared);
        double cross_term_a = c_dxyz_dabc[0] * x0 + c_dxyz_dabc[1] * y0 + c_dxyz_dabc[2] * z0;
        double cross_term_b = c_dxyz_dabc[3] * x0 + c_dxyz_dabc[4] * y0 + c_dxyz_dabc[5] * z0;
        double cross_term_c = c_dxyz_dabc[6] * x0 + c_dxyz_dabc[7] * y0 + c_dxyz_dabc[8] * z0;

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
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
        for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
            double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
            double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
            double cc = ci * cj;
            size_t nao = envs.ao_loc[nbas];
            size_t nao2 = nao * nao;
            int i0 = envs.ao_loc[ish_cell0];
            int j0 = envs.ao_loc[jsh_cell0];
            double *dm_image_shift = dm + supmol_to_bvk_mapping[jL] * nao2;
            double dm_cache[SLICE_SIZE_I * SLICE_SIZE_J];
            if (pair_id < shl_pair1) {
#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
#pragma unroll
                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                    if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                    dm_cache[i*nfj+j] = dm_image_shift[(i0+dm_i0+i)*nao+j0+j] * cc;
                } }
            } else {
                for (int n = 0; n < SLICE_SIZE_I*SLICE_SIZE_J; ++n) {
                    dm_cache[n] = 0.;
                }
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
                        double i_gradient[3*nfi];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi, dm_i0);
                        gto_deriv1<LI>(i_gradient, i_cartesian, x - xi, y - yi, z - zi, ai, dm_i0);
                        if (SLICE_SIZE_I < nfi) {
                            if (dm_i0 == 1) {
#pragma unroll
                                for (int i = 0; i < min(SLICE_SIZE_I, nfi-SLICE_SIZE_I); i++) {
                                    i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I];
                                    i_gradient[i] = i_gradient[i+SLICE_SIZE_I            ];
                                    i_gradient[i] = i_gradient[i+SLICE_SIZE_I+nfi  +nfi  ];
                                    i_gradient[i] = i_gradient[i+SLICE_SIZE_I+nfi*2+nfi*2];
                                }
                            } else if (dm_i0 == 2) {
#pragma unroll
                                for (int i = 0; i < nfi-SLICE_SIZE_I*2; i++) {
                                    i_cartesian[i] = i_cartesian[i+SLICE_SIZE_I*2];
                                    i_gradient[i      ] = i_gradient[i+SLICE_SIZE_I*2      ];
                                    i_gradient[i+nfi  ] = i_gradient[i+SLICE_SIZE_I*2+nfi  ];
                                    i_gradient[i+nfi*2] = i_gradient[i+SLICE_SIZE_I*2+nfi*2];
                                }
                            }
                        }

                        double j_cartesian[nfj];
                        double j_gradient[3*nfj];
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj, dm_j0);
                        gto_deriv1<LJ>(j_gradient, j_cartesian, x - xj, y - yj, z - zj, aj, dm_j0);
                        if (SLICE_SIZE_J < nfj) {
                            if (dm_j0 == 1) {
#pragma unroll
                                for (int i = 0; i < min(SLICE_SIZE_J, nfj-SLICE_SIZE_J); i++) {
                                    j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J];
                                    j_gradient[i      ] = j_gradient[i+SLICE_SIZE_J      ];
                                    j_gradient[i+nfj  ] = j_gradient[i+SLICE_SIZE_J+nfj  ];
                                    j_gradient[i+nfj*2] = j_gradient[i+SLICE_SIZE_J+nfj*2];
                                }
                            } else if (dm_j0 == 2) {
#pragma unroll
                                for (int i = 0; i < nfj-SLICE_SIZE_J*2; i++) {
                                    j_cartesian[i] = j_cartesian[i+SLICE_SIZE_J*2];
                                    j_gradient[i      ] = j_gradient[i+SLICE_SIZE_J*2      ];
                                    j_gradient[i+nfj  ] = j_gradient[i+SLICE_SIZE_J*2+nfj  ];
                                    j_gradient[i+nfj*2] = j_gradient[i+SLICE_SIZE_J*2+nfj*2];
                                }
                            }
                        }

                        //double rho = 0;
                        double val = 0;
#pragma unroll
                        for (int i = 0; i < SLICE_SIZE_I; ++i) {
                            if (SLICE_SIZE_I < nfi && dm_i0 + i > nfi) break;
                            //double s0 = 0;
                            double s1 = 0;
                            double s2 = 0;
                            double s3 = 0;
#pragma unroll
                            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                if (SLICE_SIZE_J < nfj && dm_j0 + j > nfj) break;
                                double dm_val = dm_cache[i * nfj + j];
                                //s0 += dm_val * j_cartesian[j];
                                s1 += dm_val * j_gradient[j      ];
                                s2 += dm_val * j_gradient[j+nfj  ];
                                s3 += dm_val * j_gradient[j+nfj*2];
                            }
                            //rho += s0 * i_cartesian[i];
                            val += s1 * i_gradient[i      ];
                            val += s2 * i_gradient[i+nfi  ];
                            val += s3 * i_gradient[i+nfi*2];
                        }
                        for (int offset = nsp_per_block/2; offset > 0; offset >>= 1) {
                            //rho += __shfl_down_sync(0xffffffff, rho, offset);
                            val += __shfl_down_sync(0xffffffff, val, offset);
                        }
                        if (sp_id == 0) {
                            int abc_index = a_index * TILE*TILE + b_index*TILE + c_index;
                            //density_value[tile_id_in_block*TILE*TILE*TILE+abc_index] += rho * gaussian_xyz;
                            tau_value[tile_id_in_block*TILE*TILE*TILE+abc_index] += val * gaussian_xyz/2;
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
        } }
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
        atomicAdd(tau + (a_idx * mesh_b + b_idx) * mesh_c + c_idx, tau_value[n]);
    }
}

extern "C" {
#define eval_tau_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_tau_kernel<li,lj,slice_i,slice_j,0><<<block_grid, threads, shm_size>>>( \
            tau, dm, *envs, shl_pair_offsets, bas_ij_idx, \
            grid_tile_index, n_contributing_tiles, supmol_to_bvk_mapping, \
            a_dot_b, a_dot_c, b_dot_c, da_squared, db_squared, dc_squared, \
            mesh_a, mesh_b, mesh_c); \
    break

int evaluate_tau(double *tau, double *dm, PBCIntEnvVars *envs,
                 double *dxyz_dabc,
                 int tiles_per_block, int nsp_per_block,
                 int shm_size, int i_angular, int j_angular,
                 int *shl_pair_offsets, int64_t *bas_ij_idx,
                 int *grid_tile_index, int *supmol_to_bvk_mapping,
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
        eval_tau_kernel_case(0,0, 1, 1);
        eval_tau_kernel_case(1,0, 3, 1);
        eval_tau_kernel_case(1,1, 3, 3);
        eval_tau_kernel_case(2,0, 6, 1);
        eval_tau_kernel_case(2,1, 6, 3);
        eval_tau_kernel_case(2,2, 6, 6);
        eval_tau_kernel_case(3,0,10, 1);
        eval_tau_kernel_case(3,1,10, 3);
        eval_tau_kernel_case(3,2,10, 6);
        eval_tau_kernel_case(3,3,10, 5);
        eval_tau_kernel_case(4,0,15, 1);
        eval_tau_kernel_case(4,1,15, 3);
        eval_tau_kernel_case(4,2, 8, 6);
        eval_tau_kernel_case(4,3, 5,10);
        eval_tau_kernel_case(4,4, 8, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_tau_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
