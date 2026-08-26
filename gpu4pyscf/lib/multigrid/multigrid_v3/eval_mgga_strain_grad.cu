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
void eval_mgga_grad_kernel(double *grad, double *strain, double *dm,
                           double *vrho_weights, double *vtau_weights, PBCIntEnvVars envs,
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
        size_t nao = envs.ao_loc[nbas];
        int i0 = envs.ao_loc[ish_cell0];
        int j0 = envs.ao_loc[jsh_cell0];
        int i = n * c_div_nf[LJ];
        int j = n - nfj * i;
        dm_cache[n] = dm[bvk_cell_id*nao*nao + (i0+i)*nao + j0+j] * factor;
    }
    __syncthreads();

    constexpr int XX = 0;
    constexpr int XY = 1;
    constexpr int XZ = 2;
    constexpr int YX = 1;
    constexpr int YY = 3;
    constexpr int YZ = 4;
    constexpr int ZX = 2;
    constexpr int ZY = 4;
    constexpr int ZZ = 5;

    double grad_i[3] = {};
    double grad_j[3] = {};
    double sigma[9] = {};

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
            if (e > 50.) continue; // ~1e-22
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

                double rho_fac = vrho_weights[abc_idx] * gaussian_xyz;
                double x_xi = x - xi;
                double y_yi = y - yi;
                double z_zi = z - zi;
                double i_deriv0[nfi];
                double i_deriv1[3*nfi];
                gto_cartesian<LI>(i_deriv0, x_xi, y_yi, z_zi);
                gto_deriv1<LI>(i_deriv1, i_deriv0, x_xi, y_yi, z_zi, ai);

                double x_xj = x - xj;
                double y_yj = y - yj;
                double z_zj = z - zj;
                double j_deriv0[nfj];
                double j_deriv1[3*nfj];
                gto_cartesian<LJ>(j_deriv0, x_xj, y_yj, z_zj);
                gto_deriv1<LJ>(j_deriv1, j_deriv0, x_xj, y_yj, z_zj, aj);
                double rho_i[3] = {};
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double s0 = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        s0 += dm_cache[i*nfj+j] * j_deriv0[j];
                    }
                    rho_i[0] += s0 * i_deriv1[i      ];
                    rho_i[1] += s0 * i_deriv1[i+nfi  ];
                    rho_i[2] += s0 * i_deriv1[i+nfi*2];
                }
                for (int n = 0; n < 3; ++n) {
                    rho_i[n] *= rho_fac;
                    grad_i[n] -= rho_i[n];
                }

                double rho_j[3] = {};
#pragma unroll
                for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                    double s0 = 0;
#pragma unroll
                    for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                        s0 += dm_cache[i*nfj+j] * i_deriv0[i];
                    }
                    rho_j[0] += s0 * j_deriv1[j      ];
                    rho_j[1] += s0 * j_deriv1[j+nfj  ];
                    rho_j[2] += s0 * j_deriv1[j+nfj*2];
                }
                for (int n = 0; n < 3; ++n) {
                    rho_j[n] *= rho_fac;
                    grad_j[n] -= rho_j[n];
                    rho_i[n] += rho_j[n];
                }

                double tau_fac = vtau_weights[abc_idx] * gaussian_xyz / 2;
                double i_deriv2[6*nfi];
                gto_deriv2<LI>(i_deriv2, x_xi, y_yi, z_zi, ai);
                double tau[3] = {};
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double sx = 0;
                    double sy = 0;
                    double sz = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        double dm_fac = dm_cache[i*nfj+j];
                        sx += dm_fac * j_deriv1[j      ];
                        sy += dm_fac * j_deriv1[j+nfj  ];
                        sz += dm_fac * j_deriv1[j+nfj*2];
                    }
                    tau[0] += sx * i_deriv2[i+nfi*XX];
                    tau[1] += sx * i_deriv2[i+nfi*XY];
                    tau[2] += sx * i_deriv2[i+nfi*XZ];
                    tau[0] += sy * i_deriv2[i+nfi*YX];
                    tau[1] += sy * i_deriv2[i+nfi*YY];
                    tau[2] += sy * i_deriv2[i+nfi*YZ];
                    tau[0] += sz * i_deriv2[i+nfi*ZX];
                    tau[1] += sz * i_deriv2[i+nfi*ZY];
                    tau[2] += sz * i_deriv2[i+nfi*ZZ];
                }
                for (int n = 0; n < 3; ++n) {
                    tau[n] *= tau_fac;
                    grad_i[n] -= tau[n];
                    rho_i[n] += tau[n];
                }

                double j_deriv2[6*nfj];
                gto_deriv2<LJ>(j_deriv2, x_xj, y_yj, z_zj, aj);
                tau[0] = 0;
                tau[1] = 0;
                tau[2] = 0;
#pragma unroll
                for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                    double sx = 0;
                    double sy = 0;
                    double sz = 0;
#pragma unroll
                    for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                        double dm_fac = dm_cache[i*nfj+j];
                        sx += dm_fac * i_deriv1[i      ];
                        sy += dm_fac * i_deriv1[i+nfi  ];
                        sz += dm_fac * i_deriv1[i+nfi*2];
                    }
                    tau[0] += sx * j_deriv2[j+nfj*XX];
                    tau[1] += sx * j_deriv2[j+nfj*XY];
                    tau[2] += sx * j_deriv2[j+nfj*XZ];
                    tau[0] += sy * j_deriv2[j+nfj*YX];
                    tau[1] += sy * j_deriv2[j+nfj*YY];
                    tau[2] += sy * j_deriv2[j+nfj*YZ];
                    tau[0] += sz * j_deriv2[j+nfj*ZX];
                    tau[1] += sz * j_deriv2[j+nfj*ZY];
                    tau[2] += sz * j_deriv2[j+nfj*ZZ];
                }
                for (int n = 0; n < 3; ++n) {
                    tau[n] *= tau_fac;
                    grad_j[n] -= tau[n];
                    rho_i[n] += tau[n];
                    sigma[n*3+0] += rho_i[n] * x;
                    sigma[n*3+1] += rho_i[n] * y;
                    sigma[n*3+2] += rho_i[n] * z;
                }

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
                double rho_fac = vrho_weights[abc_idx] * gaussian_xyz;
                double x_xi = x - xi;
                double y_yi = y - yi;
                double z_zi = z - zi;
                double i_deriv0[nfi];
                double i_deriv1[3*nfi];
                gto_cartesian<LI>(i_deriv0, x_xi, y_yi, z_zi);
                gto_deriv1<LI>(i_deriv1, i_deriv0, x_xi, y_yi, z_zi, ai);

                double x_xj = x - xj;
                double y_yj = y - yj;
                double z_zj = z - zj;
                double j_deriv0[nfj];
                double j_deriv1[3*nfj];
                gto_cartesian<LJ>(j_deriv0, x_xj, y_yj, z_zj);
                gto_deriv1<LJ>(j_deriv1, j_deriv0, x_xj, y_yj, z_zj, aj);
                double rho_i[3] = {};
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double s0 = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        s0 += dm_cache[i*nfj+j] * j_deriv0[j];
                    }
                    rho_i[0] += s0 * i_deriv1[i      ];
                    rho_i[1] += s0 * i_deriv1[i+nfi  ];
                    rho_i[2] += s0 * i_deriv1[i+nfi*2];
                }
                for (int n = 0; n < 3; ++n) {
                    rho_i[n] *= rho_fac;
                    grad_i[n] -= rho_i[n];
                }

                double rho_j[3] = {};
#pragma unroll
                for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                    double s0 = 0;
#pragma unroll
                    for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                        s0 += dm_cache[i*nfj+j] * i_deriv0[i];
                    }
                    rho_j[0] += s0 * j_deriv1[j      ];
                    rho_j[1] += s0 * j_deriv1[j+nfj  ];
                    rho_j[2] += s0 * j_deriv1[j+nfj*2];
                }
                for (int n = 0; n < 3; ++n) {
                    rho_j[n] *= rho_fac;
                    grad_j[n] -= rho_j[n];
                    rho_i[n] += rho_j[n];
                }

                double tau_fac = vtau_weights[abc_idx] * gaussian_xyz / 2;
                double i_deriv2[6*nfi];
                gto_deriv2<LI>(i_deriv2, x_xi, y_yi, z_zi, ai);
                double tau[3] = {};
#pragma unroll
                for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                    double sx = 0;
                    double sy = 0;
                    double sz = 0;
#pragma unroll
                    for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                        double dm_fac = dm_cache[i*nfj+j];
                        sx += dm_fac * j_deriv1[j      ];
                        sy += dm_fac * j_deriv1[j+nfj  ];
                        sz += dm_fac * j_deriv1[j+nfj*2];
                    }
                    tau[0] += sx * i_deriv2[i+nfi*XX];
                    tau[1] += sx * i_deriv2[i+nfi*XY];
                    tau[2] += sx * i_deriv2[i+nfi*XZ];
                    tau[0] += sy * i_deriv2[i+nfi*YX];
                    tau[1] += sy * i_deriv2[i+nfi*YY];
                    tau[2] += sy * i_deriv2[i+nfi*YZ];
                    tau[0] += sz * i_deriv2[i+nfi*ZX];
                    tau[1] += sz * i_deriv2[i+nfi*ZY];
                    tau[2] += sz * i_deriv2[i+nfi*ZZ];
                }
                for (int n = 0; n < 3; ++n) {
                    tau[n] *= tau_fac;
                    grad_i[n] -= tau[n];
                    rho_i[n] += tau[n];
                }

                double j_deriv2[6*nfj];
                gto_deriv2<LJ>(j_deriv2, x_xj, y_yj, z_zj, aj);
                tau[0] = 0;
                tau[1] = 0;
                tau[2] = 0;
#pragma unroll
                for (int j = dm_j0; j < min(dm_j0+SLICE_SIZE_J, nfj); ++j) {
                    double sx = 0;
                    double sy = 0;
                    double sz = 0;
#pragma unroll
                    for (int i = dm_i0; i < min(dm_i0+SLICE_SIZE_I, nfi); ++i) {
                        double dm_fac = dm_cache[i*nfj+j];
                        sx += dm_fac * i_deriv1[i      ];
                        sy += dm_fac * i_deriv1[i+nfi  ];
                        sz += dm_fac * i_deriv1[i+nfi*2];
                    }
                    tau[0] += sx * j_deriv2[j+nfj*XX];
                    tau[1] += sx * j_deriv2[j+nfj*XY];
                    tau[2] += sx * j_deriv2[j+nfj*XZ];
                    tau[0] += sy * j_deriv2[j+nfj*YX];
                    tau[1] += sy * j_deriv2[j+nfj*YY];
                    tau[2] += sy * j_deriv2[j+nfj*YZ];
                    tau[0] += sz * j_deriv2[j+nfj*ZX];
                    tau[1] += sz * j_deriv2[j+nfj*ZY];
                    tau[2] += sz * j_deriv2[j+nfj*ZZ];
                }
                for (int n = 0; n < 3; ++n) {
                    tau[n] *= tau_fac;
                    grad_j[n] -= tau[n];
                    rho_i[n] += tau[n];
                    sigma[n*3+0] += rho_i[n] * x;
                    sigma[n*3+1] += rho_i[n] * y;
                    sigma[n*3+2] += rho_i[n] * z;
                }
            }
        } }
    } }

    for (int n = 0; n < 3; ++n) {
        sigma[n*3+0] += grad_i[n] * xi + grad_j[n] * xj;
        sigma[n*3+1] += grad_i[n] * yi + grad_j[n] * yj;
        sigma[n*3+2] += grad_i[n] * zi + grad_j[n] * zj;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int n = 0; n < 3; ++n) {
            grad_i[n] += __shfl_down_sync(0xffffffff, grad_i[n], offset);
            grad_j[n] += __shfl_down_sync(0xffffffff, grad_j[n], offset);
        }
        for (int n = 0; n < 9; ++n) {
            sigma[n] += __shfl_down_sync(0xffffffff, sigma[n], offset);
        }
    }
    int lane = thread_id % WARP_SIZE;
    int ish_cell0 = ish;
    int bvk_cell_id = jsh / nbas;
    int jsh_cell0 = jsh - nbas * bvk_cell_id;
    int ia = bas[ish_cell0*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh_cell0*BAS_SLOTS+ATOM_OF];
    if (lane == 0) {
        for (int n = 0; n < 3; ++n) {
            atomicAdd(grad+ia*3+n, grad_i[n]);
            atomicAdd(grad+ja*3+n, grad_j[n]);
        }
        for (int n = 0; n < 9; ++n) {
            atomicAdd(strain+n, sigma[n]);
        }
    }
}

extern "C" {
#define eval_mgga_grad_kernel_case(li, lj, slice_i, slice_j) \
    case (li * LMAX1 + lj): \
        eval_mgga_grad_kernel<li,lj,slice_i,slice_j><<<npairs, threads>>>( \
            grad, strain, dm, vxc, tau, *envs, bas_ij_idx, grid_frac_ranges, \
            da_squared, db_squared, dc_squared, mesh_a, mesh_b, mesh_c, npairs, \
            factor, negligible); \
    break

int evaluate_mgga_grad(double *grad, double *strain, double *dm,
                       double *vxc, double *tau, PBCIntEnvVars *envs,
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
        eval_mgga_grad_kernel_case(0,0, 1, 1);
        eval_mgga_grad_kernel_case(1,0, 3, 1);
        eval_mgga_grad_kernel_case(1,1, 3, 3);
        eval_mgga_grad_kernel_case(2,0, 6, 1);
        eval_mgga_grad_kernel_case(2,1, 6, 3);
        eval_mgga_grad_kernel_case(2,2, 6, 3);
        eval_mgga_grad_kernel_case(3,0,10, 1);
        eval_mgga_grad_kernel_case(3,1, 5, 3);
        eval_mgga_grad_kernel_case(3,2, 5, 3);
        eval_mgga_grad_kernel_case(3,3, 5, 3);
        eval_mgga_grad_kernel_case(4,0, 8, 1);
        eval_mgga_grad_kernel_case(4,1, 5, 3);
        eval_mgga_grad_kernel_case(4,2, 5, 3);
        eval_mgga_grad_kernel_case(4,3, 3, 5);
        eval_mgga_grad_kernel_case(4,4, 3, 5);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in eval_mgga_grad_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
