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

#define TILE    4

template <typename T>
__device__ static
T estimate_rcut(int li, int lj, T x, T aij, T xpi, T xpj, T log_factor)
{
    // let s = r - Rp
    // rho[r-Rp] ~ ci*cj * exp(-theta*(Ri-Rj)**2) * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    //           ~= ovlp * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    // radius can be solved using fixed iteration
    // radius = (log(ovlp/precision * (s+Rpi)**li * (s+Rpj)**lj) / aij)**.5
    T aij_ss = log_factor + li * log(x + xpi) + lj * log(x + xpj);
    return sqrt(max(aij_ss, static_cast<T>(0)) / aij);
}

template <typename T>
__device__ inline
void accumulate(T lower, T upper, T c, T& min_val, T& max_val)
{
    T a = c * lower;
    T b = c * upper;
    min_val += min(a, b);
    max_val += max(a, b);
}

__global__ static
void grid_ranges_kernel(float2 *xfrac_range, float2 *yfrac_range, float2 *zfrac_range,
                        PBCIntEnvVars envs, int64_t *bas_ij_idx, int li_inc, int lj_inc,
                        int npairs, float log_threshold)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) return;

    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int *bas = envs.bas;
    double *env = envs.env;
    // li_inc and lj_inc to account for derivatives
    int li = bas[ish*BAS_SLOTS+ANG_OF] + li_inc;
    int lj = bas[jsh*BAS_SLOTS+ANG_OF] + lj_inc;
    int expi = bas[ish*BAS_SLOTS+PTR_EXP];
    int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = env[expi];
    float aj = env[expj];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = env[ri+0];
    float yi = env[ri+1];
    float zi = env[ri+2];
    float xj = env[rj+0];
    float yj = env[rj+1];
    float zj = env[rj+2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float rr = distance_squared(xjxi, yjyi, zjzi);
    float xp = xjxi * aj_aij + xi;
    float yp = yjyi * aj_aij + yi;
    float zp = zjzi * aj_aij + zi;
    float theta_rr = ai * aj_aij * rr;
    float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    float log_cicj = logf(fabsf(ci * cj));
    float derivative_penalty = li_inc * logf(2*ai) + lj_inc * logf(2*aj);
    float xpi = xp - xi;
    float xpj = xp - xj;
    float ypi = yp - yi;
    float ypj = yp - yj;
    float zpi = zp - zi;
    float zpj = zp - zj;

    // let s = r - Rp
    // rho[r-Rp] ~ ci*cj * exp(-theta*(Ri-Rj)**2) * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    //           ~= ovlp * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    // radius can be solved using fixed iteration
    // radius = (log(ovlp/precision * radius**(lij+l_inc)) / aij)**.5
    // where l_inc = 0 (LDA), 1 (GGA), 2 (MGGA)
    float log_r = 2.302585092994046; // log(10)
    float log_factor = log_cicj + derivative_penalty - log_threshold - theta_rr;
    // initial guess
    float radius = sqrt((log_factor + (li+lj)*log_r) / aij);
    float x_cut = estimate_rcut(li, lj, -radius, aij, xpi, xpj, log_factor);
    float y_cut = estimate_rcut(li, lj, -radius, aij, ypi, ypj, log_factor);
    float z_cut = estimate_rcut(li, lj, -radius, aij, zpi, zpj, log_factor);

    float b00 = c_reciprocal_lattice_vectors[0];
    float b01 = c_reciprocal_lattice_vectors[1];
    float b02 = c_reciprocal_lattice_vectors[2];
    float b10 = c_reciprocal_lattice_vectors[3];
    float b11 = c_reciprocal_lattice_vectors[4];
    float b12 = c_reciprocal_lattice_vectors[5];
    float b20 = c_reciprocal_lattice_vectors[6];
    float b21 = c_reciprocal_lattice_vectors[7];
    float b22 = c_reciprocal_lattice_vectors[8];

    float xp_frac = xp * b00 + yp * b01 + zp * b02;
    float yp_frac = xp * b10 + yp * b11 + zp * b12;
    float zp_frac = xp * b20 + yp * b21 + zp * b22;

    float xcut_frac = x_cut * abs(b00) + y_cut * abs(b01) + z_cut * abs(b02);
    float ycut_frac = x_cut * abs(b10) + y_cut * abs(b11) + z_cut * abs(b12);
    float zcut_frac = x_cut * abs(b20) + y_cut * abs(b21) + z_cut * abs(b22);

    xfrac_range[pair_id] = {xp_frac - xcut_frac, xp_frac + xcut_frac};
    yfrac_range[pair_id] = {yp_frac - ycut_frac, yp_frac + ycut_frac};
    zfrac_range[pair_id] = {zp_frac - zcut_frac, zp_frac + zcut_frac};
}

__global__ static
void grid_range_to_tiles_kernel(int *grid_tile_idx, int64_t *supmol_bas_ij, int64_t *bas_ij_idx,
                                float2 *xfrac_range, float2 *yfrac_range, float2 *zfrac_range,
                                int nimgs_x, int nimgs_y, int nimgs_z,
                                int mesh_x, int mesh_y, int mesh_z, int npairs,
                                int nbas, int *head)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) return;

    int64_t bas_ij = bas_ij_idx[pair_id];
    int64_t nbas_stride = nbas * NBAS_MAX;

    float2 range = xfrac_range[pair_id];
    float xfrac_lower = range.x;
    float xfrac_upper = range.y;
    range = yfrac_range[pair_id];
    float yfrac_lower = range.x;
    float yfrac_upper = range.y;
    range = zfrac_range[pair_id];
    float zfrac_lower = range.x;
    float zfrac_upper = range.y;

    int tiles_x = (mesh_x + TILE - 1) / TILE;
    int tiles_y = (mesh_y + TILE - 1) / TILE;
    int tiles_z = (mesh_z + TILE - 1) / TILE;
    int tile_size_x = min(TILE, mesh_x);
    int tile_size_y = min(TILE, mesh_y);
    int tile_size_z = min(TILE, mesh_z);

    int img_x_lower = xfrac_lower;
    int img_y_lower = yfrac_lower;
    int img_z_lower = zfrac_lower;
    int img_x_upper = xfrac_upper;
    int img_y_upper = yfrac_upper;
    int img_z_upper = zfrac_upper;
    img_x_lower = max(img_x_lower, -nimgs_x);
    img_y_lower = max(img_y_lower, -nimgs_y);
    img_z_lower = max(img_z_lower, -nimgs_z);
    img_x_upper = min(img_x_upper,  nimgs_x);
    img_y_upper = min(img_y_upper,  nimgs_y);
    img_z_upper = min(img_z_upper,  nimgs_z);
    int tile_x_lower = floor(max(xfrac_lower - img_x_lower, 0.f) * mesh_x / tile_size_x);
    int tile_y_lower = floor(max(yfrac_lower - img_y_lower, 0.f) * mesh_y / tile_size_y);
    int tile_z_lower = floor(max(zfrac_lower - img_z_lower, 0.f) * mesh_z / tile_size_z);
    int tile_x_upper = ceil (min(xfrac_upper - img_x_upper, 1.f) * mesh_x / tile_size_x);
    int tile_y_upper = ceil (min(yfrac_upper - img_y_upper, 1.f) * mesh_y / tile_size_y);
    int tile_z_upper = ceil (min(zfrac_upper - img_z_upper, 1.f) * mesh_z / tile_size_z);
    int count_x = tile_x_upper - tile_x_lower + (img_x_upper - img_x_lower) * tiles_x;
    int count_y = tile_y_upper - tile_y_lower + (img_y_upper - img_y_lower) * tiles_y;
    int count_z = tile_z_upper - tile_z_lower + (img_z_upper - img_z_lower) * tiles_z;
    // TODO: tiles in the corners sometimes are out of the cutoff radius.
    // They can be discarded and counts can be reduced
    int counts = count_x * count_y * count_z;
    int n = atomicAdd(head, counts);
    // lattice sum spans over [-nimgs_x, nimgs_x], [-nimgs_y, nimgs_y], [-nimgs_z, nimgs_z], 
    // Add img_offset to avoid negative indexing
    int img_offset = nimgs_x * (nimgs_y*2+1) * (nimgs_z*2+1) + nimgs_y * (nimgs_z*2+1) + nimgs_z;
    for (int x = tile_x_lower, img_x = img_x_lower; x < tile_x_upper || img_x < img_x_upper;) {
        for (int y = tile_y_lower, img_y = img_y_lower; y < tile_y_upper || img_y < img_y_upper;) {
            for (int z = tile_z_lower, img_z = img_z_lower; z < tile_z_upper || img_z < img_z_upper;) {
                // when (x, y, z) lies out of the unit cell, they can be repositioned
                // by shifting the lattice sum index on bra
                int64_t latsum_idx = img_offset + (img_x * nimgs_y + img_y) * nimgs_z + img_z;
                // supmol_bas_ij stores (latsum_idx*nbas+ish, jL*nbas+jsh).
                // latsum_idx is the image index to reposition bra whereas jL is
                // the image index relative to bra.
                supmol_bas_ij[n] = latsum_idx * nbas_stride + bas_ij;
                grid_tile_idx[n] = (x * tiles_y + y) * tiles_z + z;

                n++;
                z++;
                if (z >= tiles_z) {
                    z = 0;
                    img_z++;
                }
            }
            y++;
            if (y >= tiles_y) {
                y = 0;
                img_y++;
            }
        }
        x++;
        if (x >= tiles_x) {
            x = 0;
            img_x++;
        }
    }
}

extern "C" {
int gaussian_prod_grid_ranges(float2 *grid_frac_ranges, PBCIntEnvVars *envs,
                              int64_t *bas_ij_idx, int npairs,
                              int li_inc, int lj_inc, float log_threshold)
{
    int batches = (npairs + 255) / 256;
    float2 *xfrac_range = grid_frac_ranges;
    float2 *yfrac_range = grid_frac_ranges + npairs;
    float2 *zfrac_range = grid_frac_ranges + npairs * 2;
    grid_ranges_kernel<<<batches, 256>>>(
        xfrac_range, yfrac_range, zfrac_range, *envs, bas_ij_idx,
        li_inc, lj_inc, npairs, log_threshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in gaussian_prod_grid_ranges: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int grid_range_to_tiles(int *grid_tile_idx, int64_t *supmol_bas_ij,
                        int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                        int *nimgs, int *mesh, int npairs, int nbas, int *head)
{
    cudaMemset(head, 0, sizeof(int));
    int nimgs_x = nimgs[0];
    int nimgs_y = nimgs[1];
    int nimgs_z = nimgs[2];
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int batches = (npairs + 255) / 256;
    float2 *xfrac_range = grid_frac_ranges;
    float2 *yfrac_range = grid_frac_ranges + npairs;
    float2 *zfrac_range = grid_frac_ranges + npairs * 2;
    grid_range_to_tiles_kernel<<<batches, 256>>>(
        grid_tile_idx, supmol_bas_ij, bas_ij_idx, xfrac_range, yfrac_range, zfrac_range,
        nimgs_x, nimgs_y, nimgs_z, mesh_x, mesh_y, mesh_z, npairs, nbas, head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in grid_range_to_tiles: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
