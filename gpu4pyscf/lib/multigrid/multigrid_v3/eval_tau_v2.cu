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
void eval_tau_kernel(double *rho_c, double *tau_c, double *dm, PBCIntEnvVars envs,
                     double factor, int *shl_pair_offsets, int64_t *bas_ij_idx,
                     float2 *atom_frac_ranges, int atom_mesh_a_max, int nseg, int c_stride,
                     double da_squared, double db_squared, double dc_squared,
                     int mesh_a, int mesh_b, int mesh_c, double negligible)
{
    constexpr int nsp_per_block = WARP_SIZE;
    constexpr int threads = WARP_SIZE * TILE;
    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;
    setup_context();
    int tx = threadIdx_x;
    int ty = threadIdx_y;
    int thread_id = ty * nsp_per_block + tx;
    int segment_id = blockIdx_x / atom_mesh_a_max;
    int a_index_id = blockIdx_x - atom_mesh_a_max * segment_id;
    SHARED_ARRAY(int, a_index);
    SHARED_ARRAY(int, b_start);
    SHARED_ARRAY(int, b_stop);
    SHARED_ARRAY(int, c_start);
    SHARED_ARRAY(int, c_stop);
    SHARED_ARRAY(int, c_index1);
    DYNAMIC_SHARED_PTR(double, rho_cache);

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
    // Use float32 for boundary estimates; evaluation stays in float64.
    float db_x = (float)c_dxyz_dabc[3];
    float db_y = (float)c_dxyz_dabc[4];
    float db_z = (float)c_dxyz_dabc[5];
    float db_squared_f = (float)db_squared;

for (int c_index0 = c_start; c_index0 < c_stop; c_index0 += c_stride) {
    __syncthreads();
    if (thread_id == 0) {
        c_index1 = min(c_stop, c_index0 + c_stride);
    }
    int bc_offset = b_start * c_stride + c_index0;

    int atom_mesh_bc = atom_mesh_b * c_stride;
    double *tau_cache = rho_cache + atom_mesh_bc;
    for (int n = thread_id; n < atom_mesh_bc; n += threads) {
        rho_cache[n] = 0;
        tau_cache[n] = 0;
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
        float aij_f = (float)aij;
        float theta_rr_f = (float)theta_rr;
        float e_cutoff = (float)(log(fabs(cc)) - log(negligible));
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
                double x = a_index * c_dxyz_dabc[0] + b_start * c_dxyz_dabc[3] + c_index * c_dxyz_dabc[6];
                double y = a_index * c_dxyz_dabc[1] + b_start * c_dxyz_dabc[4] + c_index * c_dxyz_dabc[7];
                double z = a_index * c_dxyz_dabc[2] + b_start * c_dxyz_dabc[5] + c_index * c_dxyz_dabc[8];
                float x_xij = (float)(x - xij);
                float y_yij = (float)(y - yij);
                float z_zij = (float)(z - zij);
                float cross_term_b = db_x * x_xij + db_y * y_yij + db_z * z_zij;

                // db_peak: how many steps to move from b_start to reach the
                // closest point to the Gaussian center on this fixed-(a,c) line.
                // b_start + db_peak may lie outside [b_start, b_stop).
                float db_peak = -cross_term_b / db_squared_f;
                // Displacement from the Gaussian center to that closest point.
                float dx = x_xij + db_peak * db_x;
                float dy = y_yij + db_peak * db_y;
                float dz = z_zij + db_peak * db_z;
                float e_min = theta_rr_f + aij_f * distance_squared(dx, dy, dz);
                // An empty interval keeps inactive lanes in the warp reductions.
                int b_active_start = b_stop;
                int b_active_stop = b_stop - 1;
                if (pair_id < shl_pair1 && e_min <= e_cutoff) {
                    float b_peak = b_start + db_peak;
                    float b_radius = sqrtf((e_cutoff - e_min) / (aij_f * db_squared_f));
                    // Active bounds are inclusive; the traversal stops at b_stop-1.
                    int lower = ceilf(b_peak - b_radius);
                    int upper = floorf(b_peak + b_radius);
                    b_active_start = max(b_start, lower);
                    b_active_stop = upper;
                }

                double gaussian_xyz = 0;
                double recursion_factor_b = 0;
                // Initialize at the active start to avoid underflow, while all
                // lanes retain the common b_start traversal for warp reductions.
                if (b_active_start <= b_active_stop && b_active_start < b_stop) {
                    double db_active = b_active_start - b_start;
                    double dx_active = x - xij + db_active * c_dxyz_dabc[3];
                    double dy_active = y - yij + db_active * c_dxyz_dabc[4];
                    double dz_active = z - zij + db_active * c_dxyz_dabc[5];
                    double e = theta_rr + aij * distance_squared(dx_active, dy_active, dz_active);
                    double cross_term_b_start = c_dxyz_dabc[3] * dx_active + c_dxyz_dabc[4] * dy_active + c_dxyz_dabc[5] * dz_active;
                    gaussian_xyz = exp(-e) * cc;
                    recursion_factor_b = exp(-aij * (2 * cross_term_b_start + db_squared));
                }
                for (int b_index = b_start; b_index < b_stop; ++b_index) {
                    double rho = 0;
                    double tau = 0;
                    if (b_index >= b_active_start && b_index <= b_active_stop) {
                        double i_cartesian[nfi];
                        double i_gradient[3*nfi];
                        gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                        gto_deriv1<LI>(i_gradient, i_cartesian, x - xi, y - yi, z - zi, ai);
                        rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);
                        rename_registers(i_gradient      , dm_i0, nfi, SLICE_SIZE_I);
                        rename_registers(i_gradient+nfi  , dm_i0, nfi, SLICE_SIZE_I);
                        rename_registers(i_gradient+nfi*2, dm_i0, nfi, SLICE_SIZE_I);

                        double j_cartesian[nfj];
                        double j_gradient[3*nfj];
                        gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                        gto_deriv1<LJ>(j_gradient, j_cartesian, x - xj, y - yj, z - zj, aj);
                        rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);
                        rename_registers(j_gradient      , dm_j0, nfj, SLICE_SIZE_J);
                        rename_registers(j_gradient+nfj  , dm_j0, nfj, SLICE_SIZE_J);
                        rename_registers(j_gradient+nfj*2, dm_j0, nfj, SLICE_SIZE_J);
#pragma unroll
                        for (int i = 0; i < SLICE_SIZE_I; ++i) {
                            if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
                            double s0 = 0;
                            double s1 = 0;
                            double s2 = 0;
                            double s3 = 0;
#pragma unroll
                            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                                if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                                double dm_val = dm_cache[i * SLICE_SIZE_J + j];
                                s0 += dm_val * j_cartesian[j];
                                s1 += dm_val * j_gradient[j      ];
                                s2 += dm_val * j_gradient[j+nfj  ];
                                s3 += dm_val * j_gradient[j+nfj*2];
                            }
                            rho += s0 * i_cartesian[i];
                            tau += s1 * i_gradient[i      ];
                            tau += s2 * i_gradient[i+nfi  ];
                            tau += s3 * i_gradient[i+nfi*2];
                        }
                        rho *= gaussian_xyz;
                        tau *= gaussian_xyz;
                        gaussian_xyz *= recursion_factor_b;
                        recursion_factor_b *= exp_db_squared;
                    }
                    // Every lane participates, including inactive shell pairs.
                    for (int offset = nsp_per_block/2; offset > 0; offset >>= 1) {
                        rho += __shfl_down_sync(0xffffffff, rho, offset);
                        tau += __shfl_down_sync(0xffffffff, tau, offset);
                    }
                    if (tx == 0) {
                        rho_cache[b_index*c_stride+c_index - bc_offset] += rho;
                        tau_cache[b_index*c_stride+c_index - bc_offset] += tau;
                    }
                    x += c_dxyz_dabc[3];
                    y += c_dxyz_dabc[4];
                    z += c_dxyz_dabc[5];
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
        atomicAdd(tau_c + abc_idx*2, tau_cache[n]/2);
    }
    __syncthreads();
}
}

extern "C" {
#define eval_tau_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): { \
        auto dev_envs = *envs; \
        auto _threads = make_block(WARP_SIZE, TILE); \
        auto _grids = make_grid(ntasks); \
        LAUNCH_KERNEL( (eval_tau_kernel<li,lj,slice_i,slice_j>), _grids, _threads, shmsize, \
            rho_c, tau_c, dm, dev_envs, factor, shl_pair_offsets, bas_ij_idx, \
                        atom_frac_ranges, atom_mesh[0], nseg, c_stride, \
                        da_squared, db_squared, dc_squared, mesh_a, mesh_b, mesh_c, negligible); \
    } \
    break

int evaluate_tau_v2(double *rho_c, double *tau_c, double *dm,
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
    int c_stride = (3000 / atom_mesh[1] / TILE) * TILE;
    int shmsize = atom_mesh[1] * c_stride * 2 * sizeof(double);
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
        eval_tau_kernel_case(4,3, 8, 5);
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
