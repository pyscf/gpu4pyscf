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

template <int LI, int LJ, int SLICE_SIZE_I, int SLICE_SIZE_J>
__global__ static
void eval_lda_strain_kernel(double *out, double *dm,
                            double *vxc_weights, PBCIntEnvVars envs,
                            int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                            double da_squared, double db_squared, double dc_squared,
                            int mesh_a, int mesh_b, int mesh_c, int npairs,
                            double factor, double negligible)
{
    constexpr int tile = 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * tile + tx;
    int pair_id = blockIdx.x;

    constexpr int nfi = (LI + 1) * (LI + 2) / 2;
    constexpr int nfj = (LJ + 1) * (LJ + 2) / 2;

    __shared__ int a_start, a_stop, a_center;
    __shared__ int b_start, b_stop;
    __shared__ int c_start, c_stop;
    __shared__ double cc, exp_da_squared;
    __shared__ double xi, yi, zi;
    __shared__ double xj, yj, zj;
    __shared__ double xij, yij, zij, ai, aj, aij, theta_rr;
    __shared__ double xjxi, yjyi, zjzi;
    __shared__ double dm_cache[nfi*nfj];

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
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ai = env[bas[ish_cell0*BAS_SLOTS+PTR_EXP]];
        aj = env[bas[jsh_cell0*BAS_SLOTS+PTR_EXP]];
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
        xjxi = xj - xi;
        yjyi = yj - yi;
        zjzi = zj - zi;
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
        a_center = (a_start + a_stop) / 2;
        exp_da_squared = exp(-2 * aij * da_squared);
    }
    __syncthreads();

    for (int n = thread_id; n < nfi * nfj; n += THREADS) {
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        uint32_t nao = envs.ao_loc[nbas];
        int i0 = envs.ao_loc[ish_cell0];
        int j0 = envs.ao_loc[jsh_cell0];
        int i = n * c_div_nf[LJ];
        int j = n - nfj * i;
        dm_cache[n] = dm[bvk_cell_id*nao*nao + (i0+i)*nao + j0+j] * factor;
    }

    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_xz = 0;
    double sigma_yx = 0;
    double sigma_yy = 0;
    double sigma_yz = 0;
    double sigma_zx = 0;
    double sigma_zy = 0;
    double sigma_zz = 0;

#pragma unroll
    for (int dm_i0 = 0; dm_i0 < nfi; dm_i0 += SLICE_SIZE_I) {
#pragma unroll
    for (int dm_j0 = 0; dm_j0 < nfj; dm_j0 += SLICE_SIZE_J) {

        for (int b_index = b_start+ty; b_index <= b_stop; b_index += tile) {
        for (int c_index = c_start+tx; c_index <= c_stop; c_index += tile) {
            double x_start = a_center * c_dxyz_dabc[0] + b_index * c_dxyz_dabc[3] + c_index * c_dxyz_dabc[6];
            double y_start = a_center * c_dxyz_dabc[1] + b_index * c_dxyz_dabc[4] + c_index * c_dxyz_dabc[7];
            double z_start = a_center * c_dxyz_dabc[2] + b_index * c_dxyz_dabc[5] + c_index * c_dxyz_dabc[8];
            double x = x_start;
            double y = y_start;
            double z = z_start;
            double x_xij = x - xij;
            double y_yij = y - yij;
            double z_zij = z - zij;
            double e = theta_rr + aij * distance_squared(x_xij, y_yij, z_zij);
            if (e > 42.) continue; // ~1e-18
            double gaussian_starting_point = exp(-e) * cc;
            double cross_term_a = c_dxyz_dabc[0] * x_xij + c_dxyz_dabc[1] * y_yij + c_dxyz_dabc[2] * z_zij;
            double recursion_factor_a_start = exp(-aij * (2 * cross_term_a + da_squared));

            int mesh_bc = mesh_b * mesh_c;
            int mesh_abc = mesh_a * mesh_bc;
            // mod(negative_number, N) leads to a negative value. Adding a large
            // multiplier to a_index, b_index, c_index to avoid the negative modulo.
            // Images spreads in supmol are typically < 10. A shift of 100* images
            // should be enough.
            int64_t abc_idx_start = (a_center + 100 * mesh_a) % mesh_a * (int64_t)mesh_bc +
                (b_index + 100 * mesh_b) % mesh_b * mesh_c +
                (c_index + 100 * mesh_c) % mesh_c;
            int64_t abc_idx = abc_idx_start;
            double gaussian_xyz = gaussian_starting_point;
            double recursion_factor_a = recursion_factor_a_start;
            for (int a_index = a_center; a_index <= a_stop; a_index++,
                 gaussian_xyz *= recursion_factor_a,
                 recursion_factor_a *= exp_da_squared) {
                if (fabs(gaussian_xyz) < negligible) break;

                double i_deriv0[nfi];
                double i_deriv1[3*nfi];
                gto_cartesian<LI>(i_deriv0, x - xi, y - yi, z - zi);
                gto_deriv1<LI>(i_deriv1, i_deriv0, x - xi, y - yi, z - zi, ai);

                double j_deriv0[nfj];
                gto_cartesian<LJ>(j_deriv0, x - xj, y - yj, z - zj);
                double rhox = 0;
                double rhoy = 0;
                double rhoz = 0;
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double s = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        s += dm_cache[i*nfj+j] * j_deriv0[j];
                    }
                    rhox += s * i_deriv1[i      ];
                    rhoy += s * i_deriv1[i+nfi  ];
                    rhoz += s * i_deriv1[i+nfi*2];
                }
                double v = vxc_weights[abc_idx] * gaussian_xyz;
                rhox *= v;
                rhoy *= v;
                rhoz *= v;
                sigma_xx -= rhox * xjxi;
                sigma_xy -= rhox * yjyi;
                sigma_xz -= rhox * zjzi;
                sigma_yx -= rhoy * xjxi;
                sigma_yy -= rhoy * yjyi;
                sigma_yz -= rhoy * zjzi;
                sigma_zx -= rhoz * xjxi;
                sigma_zy -= rhoz * yjyi;
                sigma_zz -= rhoz * zjzi;

                x += c_dxyz_dabc[0];
                y += c_dxyz_dabc[1];
                z += c_dxyz_dabc[2];
                abc_idx += mesh_bc;
                if (abc_idx >= mesh_abc) {
                    abc_idx -= mesh_abc;
                }
            }

            x = x_start;
            y = y_start;
            z = z_start;
            gaussian_xyz = gaussian_starting_point;
            double inv_recursion_factor_a = exp_da_squared / recursion_factor_a_start;
            abc_idx = abc_idx_start;
            for (int a_index = a_center - 1; a_index >= a_start; a_index--,
                inv_recursion_factor_a *= exp_da_squared) {
                gaussian_xyz *= inv_recursion_factor_a;
                if (fabs(gaussian_xyz) < negligible) break;
                x -= c_dxyz_dabc[0];
                y -= c_dxyz_dabc[1];
                z -= c_dxyz_dabc[2];
                abc_idx -= mesh_bc;
                if (abc_idx < 0) {
                    abc_idx += mesh_abc;
                }
                double i_deriv0[nfi];
                double i_deriv1[3*nfi];
                gto_cartesian<LI>(i_deriv0, x - xi, y - yi, z - zi);
                gto_deriv1<LI>(i_deriv1, i_deriv0, x - xi, y - yi, z - zi, ai);

                double j_deriv0[nfj];
                gto_cartesian<LJ>(j_deriv0, x - xj, y - yj, z - zj);
                double rhox = 0;
                double rhoy = 0;
                double rhoz = 0;
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double s = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        s += dm_cache[i*nfj+j] * j_deriv0[j];
                    }
                    rhox += s * i_deriv1[i      ];
                    rhoy += s * i_deriv1[i+nfi  ];
                    rhoz += s * i_deriv1[i+nfi*2];
                }
                double v = vxc_weights[abc_idx] * gaussian_xyz;
                rhox *= v;
                rhoy *= v;
                rhoz *= v;
                sigma_xx -= rhox * xjxi;
                sigma_xy -= rhox * yjyi;
                sigma_xz -= rhox * zjzi;
                sigma_yx -= rhoy * xjxi;
                sigma_yy -= rhoy * yjyi;
                sigma_yz -= rhoy * zjzi;
                sigma_zx -= rhoz * xjxi;
                sigma_zy -= rhoz * yjyi;
                sigma_zz -= rhoz * zjzi;
            }
        } }
    } }

    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        sigma_xx += __shfl_down_sync(0xffffffff, sigma_xx, offset);
        sigma_xy += __shfl_down_sync(0xffffffff, sigma_xy, offset);
        sigma_xz += __shfl_down_sync(0xffffffff, sigma_xz, offset);
        sigma_yx += __shfl_down_sync(0xffffffff, sigma_yx, offset);
        sigma_yy += __shfl_down_sync(0xffffffff, sigma_yy, offset);
        sigma_yz += __shfl_down_sync(0xffffffff, sigma_yz, offset);
        sigma_zx += __shfl_down_sync(0xffffffff, sigma_zx, offset);
        sigma_zy += __shfl_down_sync(0xffffffff, sigma_zy, offset);
        sigma_zz += __shfl_down_sync(0xffffffff, sigma_zz, offset);
    }
    int lane = thread_id % WARP_SIZE;
    if (lane == 0) {
        atomicAdd(out+0, sigma_xx);
        atomicAdd(out+1, sigma_xy);
        atomicAdd(out+2, sigma_xz);
        atomicAdd(out+3, sigma_yx);
        atomicAdd(out+4, sigma_yy);
        atomicAdd(out+5, sigma_yz);
        atomicAdd(out+6, sigma_zx);
        atomicAdd(out+7, sigma_zy);
        atomicAdd(out+8, sigma_zz);
    }
}

extern "C" {
#define eval_lda_strain_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_lda_strain_kernel<li,lj,slice_i,slice_j><<<npairs, threads>>>( \
            out, dm, vxc, *envs, bas_ij_idx, grid_frac_ranges, \
            da_squared, db_squared, dc_squared, mesh_a, mesh_b, mesh_c, npairs, \
            factor, negligible); \
    break

int evaluate_lda_strain(double *out, double *dm,
                        double *vxc, double *placeholder, PBCIntEnvVars *envs,
                        double *dxyz_dabc, int li, int lj, int64_t *bas_ij_idx,
                        float2 *grid_frac_ranges, int *mesh, int npairs,
                        double factor, double negligible)
{
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    double da_squared = distance_squared(dxyz_dabc[0], dxyz_dabc[1], dxyz_dabc[2]);
    double db_squared = distance_squared(dxyz_dabc[3], dxyz_dabc[4], dxyz_dabc[5]);
    double dc_squared = distance_squared(dxyz_dabc[6], dxyz_dabc[7], dxyz_dabc[8]);
    dim3 threads(16, 16);
    switch (li * LMAX1 + lj) {
        eval_lda_strain_kernel_case(0,0, 1, 1);
        eval_lda_strain_kernel_case(1,0, 3, 1);
        eval_lda_strain_kernel_case(1,1, 3, 3);
        eval_lda_strain_kernel_case(2,0, 6, 1);
        eval_lda_strain_kernel_case(2,1, 6, 3);
        eval_lda_strain_kernel_case(2,2, 6, 6);
        eval_lda_strain_kernel_case(3,0,10, 1);
        eval_lda_strain_kernel_case(3,1,10, 3);
        eval_lda_strain_kernel_case(3,2,10, 6);
        eval_lda_strain_kernel_case(3,3,10, 5);
        eval_lda_strain_kernel_case(4,0,15, 1);
        eval_lda_strain_kernel_case(4,1,15, 3);
        eval_lda_strain_kernel_case(4,2, 8, 6);
        eval_lda_strain_kernel_case(4,3, 8, 5);
        eval_lda_strain_kernel_case(4,4, 8, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
