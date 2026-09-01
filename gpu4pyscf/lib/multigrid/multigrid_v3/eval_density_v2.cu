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

#define TILE            8
#define WARP_SIZE       32

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J>
__global__ static
void eval_density_kernel(double *rho_c, double *dm, PBCIntEnvVars envs,
                         double factor, int *shl_pair_offsets, int64_t *bas_ij_idx,
                         float2 *atom_frac_ranges, int atom_mesh_a_max, int nseg, int c_stride,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c, double negligible)
{
    constexpr int nsp_per_block = WARP_SIZE;
    constexpr int threads = WARP_SIZE * TILE;
    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * nsp_per_block + tx;
    int segment_id = blockIdx.x / atom_mesh_a_max;
    int a_index_id = blockIdx.x - atom_mesh_a_max * segment_id;
    __shared__ int a_index;
    __shared__ int b_start, b_stop;
    __shared__ int c_start, c_stop, c_index1;
    extern __shared__ double rho_cache[];

    float2 x_range = atom_frac_ranges[       segment_id];
    float2 y_range = atom_frac_ranges[nseg  +segment_id];
    float2 z_range = atom_frac_ranges[nseg*2+segment_id];
    if (thread_id == 0) {
        b_start = ceil(y_range.x * mesh_b);
        c_start = ceil(z_range.x * mesh_c);
        b_stop = ceil(y_range.y * mesh_b);
        c_stop = ceil(z_range.y * mesh_c);
        int a_start = ceil(x_range.x * mesh_a);
        a_index = a_start + a_index_id;
    }
    __syncthreads();
    int a_stop = ceil(x_range.y * mesh_a);
    if (a_index >= a_stop) {
        return;
    }

    int shl_pair0 = shl_pair_offsets[segment_id];
    int shl_pair1 = shl_pair_offsets[segment_id+1];
    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    int bvk_nbas = envs.bvk_ncells * envs.nbas;
    int atom_mesh_b = b_stop - b_start;

for (int c_index0 = c_start; c_index0 < c_stop; c_index0 += c_stride) {
    __syncthreads();
    if (thread_id == 0) {
        c_index1 = min(c_stop, c_index0 + c_stride);
    }
    int b_center = (b_start + b_stop) / 2;
    int bc_offset = b_start * c_stride + c_start;

    int atom_mesh_bc = atom_mesh_b * c_stride;
    for (int n = thread_id; n < atom_mesh_bc; n += threads) {
        rho_cache[n] = 0;
    }
    __syncthreads();

    for (int pair_id = shl_pair0+tx; pair_id < shl_pair1+tx; pair_id += nsp_per_block) {
        int64_t bas_ij = 0;
        if (pair_id < shl_pair1) {
            bas_ij = bas_ij_idx[pair_id];
        }
        int ish = bas_ij / NBAS_MAX;
        int jsh = bas_ij % NBAS_MAX;
        int jL = jsh / bvk_nbas;
        jsh = jsh - bvk_nbas * jL;
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = env[bas[ish_cell0*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh_cell0*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
        double cc = ci * cj * factor;
        if (ish_cell0 == jsh_cell0) {
            cc *= .5;
        }
        if (pair_id >= shl_pair1) {
            cc = 0;
        }
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xi = env[ri+0];
        double yi = env[ri+1];
        double zi = env[ri+2];
        double xj = env[rj+0] + envs.img_coords[jL*3+0];
        double yj = env[rj+1] + envs.img_coords[jL*3+1];
        double zj = env[rj+2] + envs.img_coords[jL*3+2];
        double xjxi = xj - xi;
        double yjyi = yj - yi;
        double zjzi = zj - zi;
        double rr = distance_squared(xjxi, yjyi, zjzi);
        double xij = xjxi * aj_aij + xi;
        double yij = yjyi * aj_aij + yi;
        double zij = zjzi * aj_aij + zi;
        double theta_rr = ai * aj_aij * rr;
        double exp_db_squared = exp(-2 * aij * db_squared);
        // TODO: adjust b_start and b_stop for each individual shl_pair batch
#pragma unroll
        for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
#pragma unroll
        for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
            size_t nao = envs.ao_loc[nbas];
            int i0 = envs.ao_loc[ish_cell0];
            int j0 = envs.ao_loc[jsh_cell0];
            size_t ij_offset = bvk_cell_id * nao * nao + (dm_i0+i0) * nao + dm_j0+j0;
            double dm_cache[SLICE_SIZE_I * SLICE_SIZE_J];
            if (pair_id < shl_pair1) {
#pragma unroll
                for (int i = 0; i < SLICE_SIZE_I; ++i) {
                    if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
#pragma unroll
                for (int j = 0; j < SLICE_SIZE_J; ++j) {
                    if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                    dm_cache[i*SLICE_SIZE_J+j] = dm[ij_offset + i*nao+j];
                } }
            }
            for (int c_index = c_index0+ty; c_index < c_index1; c_index += TILE) {
                double x_start = a_index * c_dxyz_dabc[0] + b_center * c_dxyz_dabc[3] + c_index * c_dxyz_dabc[6];
                double y_start = a_index * c_dxyz_dabc[1] + b_center * c_dxyz_dabc[4] + c_index * c_dxyz_dabc[7];
                double z_start = a_index * c_dxyz_dabc[2] + b_center * c_dxyz_dabc[5] + c_index * c_dxyz_dabc[8];
                double x = x_start;
                double y = y_start;
                double z = z_start;
                double x_xij = x - xij;
                double y_yij = y - yij;
                double z_zij = z - zij;
                double e = theta_rr + aij * distance_squared(x_xij, y_yij, z_zij);
                //if (e > 50.) continue; // ~1e-22
                double gaussian_starting_point = exp(-e) * cc;
                double cross_term_b = c_dxyz_dabc[3] * x_xij + c_dxyz_dabc[4] * y_yij + c_dxyz_dabc[5] * z_zij;
                double recursion_factor_b_start = exp(-aij * (2 * cross_term_b + db_squared));

                double gaussian_xyz = gaussian_starting_point;
                double recursion_factor_b = recursion_factor_b_start;
                for (int b_index = b_center; b_index < b_stop; b_index++,
                     gaussian_xyz *= recursion_factor_b,
                     recursion_factor_b *= exp_db_squared) {
                    double rho = 0;
                    if (pair_id < shl_pair1 && fabs(gaussian_xyz) > negligible) {
                        double i_cartesian[nfi];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                        rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);

                        double j_cartesian[nfj];
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                        rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);
#pragma unroll
                        for (int i = 0; i < SLICE_SIZE_I; ++i) {
                            if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
                            double s = 0;
#pragma unroll
                            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                                s += dm_cache[i * SLICE_SIZE_J + j] * j_cartesian[j];
                            }
                            rho += s * i_cartesian[i];
                        }
                        rho *= gaussian_xyz;
                    }
                    for (int offset = nsp_per_block/2; offset > 0; offset >>= 1) {
                        rho += __shfl_down_sync(0xffffffff, rho, offset);
                    }
                    if (tx == 0) {
                        rho_cache[b_index*c_stride+c_index - bc_offset] += rho;
                    }
                    x += c_dxyz_dabc[3];
                    y += c_dxyz_dabc[4];
                    z += c_dxyz_dabc[5];
                }
                x = x_start;
                y = y_start;
                z = z_start;
                gaussian_xyz = gaussian_starting_point;
                double inv_recursion_factor_b = exp_db_squared / recursion_factor_b_start;
                for (int b_index = b_center - 1; b_index >= b_start; b_index--,
                    inv_recursion_factor_b *= exp_db_squared) {
                    gaussian_xyz *= inv_recursion_factor_b;
                    //if (fabs(gaussian_xyz) < negligible) break;
                    x -= c_dxyz_dabc[3];
                    y -= c_dxyz_dabc[4];
                    z -= c_dxyz_dabc[5];
                    double rho = 0;
                    if (pair_id < shl_pair1 && fabs(gaussian_xyz) > negligible) {
                        double i_cartesian[nfi];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                        rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);

                        double j_cartesian[nfj];
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                        rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);
#pragma unroll
                        for (int i = 0; i < SLICE_SIZE_I; ++i) {
                            if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
                            double s = 0;
#pragma unroll
                            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                                s += dm_cache[i * SLICE_SIZE_J + j] * j_cartesian[j];
                            }
                            rho += s * i_cartesian[i];
                        }
                        rho *= gaussian_xyz;
                    }
                    for (int offset = nsp_per_block/2; offset > 0; offset >>= 1) {
                        rho += __shfl_down_sync(0xffffffff, rho, offset);
                    }
                    if (tx == 0) {
                        rho_cache[b_index*c_stride+c_index - bc_offset] += rho;
                    }
                }
            }
        } }
    }
    __syncthreads();
    int64_t mesh_bc = mesh_b * mesh_c;
    int64_t abc_idx_start = (a_index + 100 * mesh_a) % mesh_a * mesh_bc;
    for (int n = thread_id; n < atom_mesh_bc; n += threads) {
        int b_index = n / c_stride;
        int c_index = n - c_stride * b_index + c_index0;
        if (c_index >= c_stop) continue;
        int64_t abc_idx = abc_idx_start +
            (b_start + b_index + 100 * mesh_b) % mesh_b * mesh_c +
            (c_index + 100 * mesh_c) % mesh_c;
        atomicAdd(rho_c + abc_idx*2, rho_cache[n]);
    }
    __syncthreads();
}
}

extern "C" {
#define eval_density_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_density_kernel<li,lj,slice_i,slice_j><<<ntasks, threads, shmsize>>>( \
            rho_c, dm, *envs, factor, shl_pair_offsets, bas_ij_idx, \
            atom_frac_ranges, atom_mesh[0], nseg, c_stride, \
            da_squared, db_squared, dc_squared, mesh_a, mesh_b, mesh_c, negligible); \
    break

int evaluate_density_v2(double *rho_c, double *placeholder, double *dm,
                     PBCIntEnvVars *envs, double *dxyz_dabc,
                     int i_angular, int j_angular,
                     int *shl_pair_offsets, int64_t *bas_ij_idx,
                     float2 *atom_frac_ranges, int nseg, int *atom_mesh,
                     int *mesh, double factor, double negligible)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    int ntasks = nseg * atom_mesh[0];
    int c_stride = (6000 / atom_mesh[1] / TILE) * TILE;
    int shmsize = atom_mesh[1] * c_stride * sizeof(double);
    dim3 threads(WARP_SIZE, TILE);
    double a_dot_b = dxyz_dabc[0] * dxyz_dabc[3] + dxyz_dabc[1] * dxyz_dabc[4] + dxyz_dabc[2] * dxyz_dabc[5];
    double a_dot_c = dxyz_dabc[0] * dxyz_dabc[6] + dxyz_dabc[1] * dxyz_dabc[7] + dxyz_dabc[2] * dxyz_dabc[8];
    double b_dot_c = dxyz_dabc[3] * dxyz_dabc[6] + dxyz_dabc[4] * dxyz_dabc[7] + dxyz_dabc[5] * dxyz_dabc[8];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    switch (i_angular * LMAX1 + j_angular) {
        eval_density_kernel_case(0,0, 1, 1);
        eval_density_kernel_case(1,0, 3, 1);
        eval_density_kernel_case(1,1, 3, 3);
        eval_density_kernel_case(2,0, 6, 1);
        eval_density_kernel_case(2,1, 6, 3);
        eval_density_kernel_case(2,2, 6, 6);
        eval_density_kernel_case(3,0,10, 1);
        eval_density_kernel_case(3,1,10, 3);
        eval_density_kernel_case(3,2,10, 6);
        eval_density_kernel_case(3,3,10, 5);
        eval_density_kernel_case(4,0,15, 1);
        eval_density_kernel_case(4,1,15, 3);
        eval_density_kernel_case(4,2, 8, 6);
        eval_density_kernel_case(4,3,15, 5);
        eval_density_kernel_case(4,4,15, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_density_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
