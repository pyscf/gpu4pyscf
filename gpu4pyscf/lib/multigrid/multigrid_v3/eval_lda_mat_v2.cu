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
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "constant_objects.cuh"
#include "cartesian.cuh"
#include "utils.cuh"

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J>
__global__ static
void eval_lda_mat_kernel_v2(double *out, double *vxc_weights,
                         int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                         double da_squared, double db_squared, double dc_squared,
                         int mesh_a, int mesh_b, int mesh_c, int npairs,
                         double negligible, PBCIntEnvVars envs)
{
    constexpr int tile = 16;
    setup_context();
    int tx = threadIdx_x;
    int ty = threadIdx_y;
    int thread_id = ty * tile + tx;
    int pair_id = blockIdx_x;

    SHARED_ARRAY(int, a_start);
    SHARED_ARRAY(int, a_stop);
    SHARED_ARRAY(int, a_center);
    SHARED_ARRAY(int, b_start);
    SHARED_ARRAY(int, b_stop);
    SHARED_ARRAY(int, c_start);
    SHARED_ARRAY(int, c_stop);
    SHARED_ARRAY(uint32_t, ij_offset);
    SHARED_ARRAY(double, cc);
    SHARED_ARRAY(double, exp_da_squared);
    SHARED_ARRAY(float, e_cutoff);
    SHARED_ARRAY(double, xi);
    SHARED_ARRAY(double, yi);
    SHARED_ARRAY(double, zi);
    SHARED_ARRAY(double, xj);
    SHARED_ARRAY(double, yj);
    SHARED_ARRAY(double, zj);
    SHARED_ARRAY(double, xij);
    SHARED_ARRAY(double, yij);
    SHARED_ARRAY(double, zij);
    SHARED_ARRAY(double, aij);
    SHARED_ARRAY(double, theta_rr);
    SHARED_ARRAY(double, swap, [8]);

    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    int bvk_nbas = envs.bvk_ncells * envs.nbas;

    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int jL = jsh / bvk_nbas;
    jsh = jsh - bvk_nbas * jL;
    if (thread_id == 0) {
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        uint32_t nao = envs.ao_loc[nbas];
        uint32_t i0 = envs.ao_loc[ish_cell0];
        uint32_t j0 = envs.ao_loc[jsh_cell0];
        ij_offset = bvk_cell_id * nao * nao + i0 * nao + j0;

        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = env[bas[ish_cell0*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh_cell0*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish_cell0*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh_cell0*BAS_SLOTS+PTR_COEFF]];
        cc = ci * cj;
        if (ish_cell0 == jsh_cell0) {
            cc *= .5;
        }
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
        a_start = ceil(xfrac_lower * mesh_a);
        b_start = ceil(yfrac_lower * mesh_b);
        c_start = ceil(zfrac_lower * mesh_c);
        a_stop = floor(xfrac_upper * mesh_a);
        b_stop = floor(yfrac_upper * mesh_b);
        c_stop = floor(zfrac_upper * mesh_c);
        exp_da_squared = exp(-2 * aij * da_squared);
        e_cutoff = (float)(log(fabs(cc)) - log(negligible));
    }
    __syncthreads();

    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;

    // Use float32 for boundary estimates; integration stays in float64.
    float da_x = (float)c_dxyz_dabc[0];
    float da_y = (float)c_dxyz_dabc[1];
    float da_z = (float)c_dxyz_dabc[2];
    float da_squared_f = (float)da_squared;
    float aij_f = (float)aij;
    float theta_rr_f = (float)theta_rr;

#pragma unroll
    for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
#pragma unroll
    for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {
        double vj_cache[SLICE_SIZE_I * SLICE_SIZE_J];
#pragma unroll
        for (int n = 0; n < SLICE_SIZE_I*SLICE_SIZE_J; ++n) {
            vj_cache[n] = 0;
        }

        for (int b_index = b_start+ty; b_index <= b_stop; b_index += tile) {
        for (int c_index = c_start+tx; c_index <= c_stop; c_index += tile) {
            double x = a_start * c_dxyz_dabc[0] + b_index * c_dxyz_dabc[3] + c_index * c_dxyz_dabc[6];
            double y = a_start * c_dxyz_dabc[1] + b_index * c_dxyz_dabc[4] + c_index * c_dxyz_dabc[7];
            double z = a_start * c_dxyz_dabc[2] + b_index * c_dxyz_dabc[5] + c_index * c_dxyz_dabc[8];
            float x_xij = (float)(x - xij);
            float y_yij = (float)(y - yij);
            float z_zij = (float)(z - zij);
            float cross_term_a = da_x * x_xij + da_y * y_yij + da_z * z_zij;

            // da_peak: how many steps to move from a_start to reach the
            // closest point to the Gaussian center.
            // a_start + da_peak may lie outside [a_start, a_stop].
            float da_peak = -cross_term_a / da_squared_f;
            // (dx,dy,dz) is the displacement from the Gaussian center to that
            // closest point.
            float dx = x_xij + da_peak * da_x;
            float dy = y_yij + da_peak * da_y;
            float dz = z_zij + da_peak * da_z;
            // Minimum exponent on the line, including the shell-pair prefactor.
            float e_min = theta_rr_f + aij_f * distance_squared(dx, dy, dz);
            if (e_min > e_cutoff) continue;
            // Absolute continuous a-grid index of the peak, and the half-width
            // in grid steps over which |cc| * exp(-e) >= negligible.
            float a_peak = a_start + da_peak;
            float a_radius = sqrtf((e_cutoff - e_min) / (aij_f * da_squared_f));
            // Intersect with the original bounds and retain integer grid points.
            // a_active_start and a_active_stop are the inclusive first and last indices.
            int lower = ceilf(a_peak - a_radius);
            int upper = floorf(a_peak + a_radius);
            int a_active_start = max(a_start, lower);
            int a_active_stop = min(a_stop, upper);
            if (a_active_start > a_active_stop) continue;

            int mesh_bc = mesh_b * mesh_c;
            int mesh_abc = mesh_a * mesh_bc;
            // mod(negative_number, N) leads to a negative value. Adding a large
            // multiplier to a_index, b_index, c_index to avoid the negative modulo.
            // Images spreads in supmol are typically < 10. A shift of 100* images
            // should be enough.
            int64_t abc_idx = (a_start + 100 * mesh_a) % mesh_a * (int64_t)mesh_bc +
                (b_index + 100 * mesh_b) % mesh_b * mesh_c +
                (c_index + 100 * mesh_c) % mesh_c;
            // Initialize at the active start to avoid underflow, but traverse
            // from a_start to keep neighboring threads' grid accesses aligned.
            double da_active = a_active_start - a_start;
            double dx_active = x - xij + da_active * c_dxyz_dabc[0];
            double dy_active = y - yij + da_active * c_dxyz_dabc[1];
            double dz_active = z - zij + da_active * c_dxyz_dabc[2];
            double e = theta_rr + aij * distance_squared(dx_active, dy_active, dz_active);
            double cross_term_a_start = c_dxyz_dabc[0] * dx_active + c_dxyz_dabc[1] * dy_active + c_dxyz_dabc[2] * dz_active;
            double gaussian_xyz = exp(-e) * cc;
            double recursion_factor_a = exp(-aij * (2 * cross_term_a_start + da_squared));
            for (int a_index = a_start; a_index <= a_active_stop; ++a_index) {
                if (a_index >= a_active_start) {
                    double v = vxc_weights[abc_idx] * gaussian_xyz;
                    double i_cartesian[nfi];
                    gto_cartesian<LI>(i_cartesian, x - xi, y - yi, z - zi);
                    rename_registers(i_cartesian, dm_i0, nfi, SLICE_SIZE_I);

                    double j_cartesian[nfj];
                    gto_cartesian<LJ>(j_cartesian, x - xj, y - yj, z - zj);
                    rename_registers(j_cartesian, dm_j0, nfj, SLICE_SIZE_J);
#pragma unroll
                    for (int i = 0; i < SLICE_SIZE_I; ++i) {
                        if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
                        double s = v * i_cartesian[i];
#pragma unroll
                        for (int j = 0; j < SLICE_SIZE_J; ++j) {
                            if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                            vj_cache[i*SLICE_SIZE_J+j] += s * j_cartesian[j];
                        }
                    }
                    gaussian_xyz *= recursion_factor_a;
                    recursion_factor_a *= exp_da_squared;
                }
                x += c_dxyz_dabc[0];
                y += c_dxyz_dabc[1];
                z += c_dxyz_dabc[2];
                abc_idx += mesh_bc;
                if (abc_idx >= mesh_abc) {
                    abc_idx -= mesh_abc;
                }
            }
        } }

        uint32_t nao = envs.ao_loc[nbas];
        double *pout = out + ij_offset + dm_i0 * nao + dm_j0;
#pragma unroll
        for (int i = 0; i < SLICE_SIZE_I; ++i) {
            if (SLICE_SIZE_I < nfi && dm_i0 + i >= nfi) break;
#pragma unroll
            for (int j = 0; j < SLICE_SIZE_J; ++j) {
                if (SLICE_SIZE_J < nfj && dm_j0 + j >= nfj) break;
                double val = reduce(vj_cache[i*SLICE_SIZE_J+j], swap, thread_id);
                if (thread_id == 0) {
                    atomicAdd(pout + i*nao+j, val);
                }
            }
        }
    } }
}

extern "C" {
#define eval_lda_mat_kernel_v2_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): { \
        auto dev_envs = *envs; \
        LAUNCH_KERNEL((eval_lda_mat_kernel_v2<li,lj,slice_i,slice_j>), grids, threads, 0, \
            out, vxc, bas_ij_idx, grid_frac_ranges, \
            da_squared, db_squared, dc_squared, mesh_a, mesh_b, mesh_c, npairs, \
            negligible, dev_envs); \
    } \
    break

int evaluate_lda_mat_v2(double *out, double *vxc, double *placeholder, PBCIntEnvVars *envs,
                     double *dxyz_dabc, int li, int lj, int64_t *bas_ij_idx,
                     float2 *grid_frac_ranges, int *mesh, int npairs,
                     double negligible)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    auto threads = make_block(16, 16);
    auto grids = make_grid(npairs, 1);
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
        eval_lda_mat_kernel_v2_case(3,3,10, 5);
        eval_lda_mat_kernel_v2_case(4,0,15, 1);
        eval_lda_mat_kernel_v2_case(4,1,15, 3);
        eval_lda_mat_kernel_v2_case(4,2, 8, 6);
        eval_lda_mat_kernel_v2_case(4,3,15, 5);
        eval_lda_mat_kernel_v2_case(4,4,15, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
