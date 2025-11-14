/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NATOM_PER_BLOCK        128
#define TILE    16

__global__
void GDFTgrid_weight_kernel(double *weight, double *coords, double *atm_coords, double *a,
                            int *atm_idx, int ngrids, int natm)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * TILE + tx;
    int grid_id = blockIdx.x * TILE*TILE + thread_id;
    double xg = 0.0;
    double yg = 0.0;
    double zg = 0.0;
    int atom_id = natm;
    if (grid_id < ngrids) {
        xg = coords[0*ngrids+grid_id];
        yg = coords[1*ngrids+grid_id];
        zg = coords[2*ngrids+grid_id];
        atom_id = atm_idx[grid_id];
    }
    double *atm_x = atm_coords;
    double *atm_y = atm_x + natm;
    double *atm_z = atm_y + natm;
    __shared__ double atom_xi[TILE];
    __shared__ double atom_yi[TILE];
    __shared__ double atom_zi[TILE];
    __shared__ double atom_xj[TILE];
    __shared__ double atom_yj[TILE];
    __shared__ double atom_zj[TILE];
    __shared__ double a_smem[TILE*TILE];
    __shared__ double dij_smem[TILE*TILE];

    double becke_self = 0.;
    double becke_sum = 0.;
    for (int atom_i0 = 0; atom_i0 < natm; atom_i0 += TILE) {
        int i1 = min(natm-atom_i0, TILE);
        __syncthreads();
        if (ty == 0 && atom_i0 + tx < natm) {
            int atom_i = atom_i0 + tx;
            atom_xi[tx] = atm_x[atom_i];
            atom_yi[tx] = atm_y[atom_i];
            atom_zi[tx] = atm_z[atom_i];
        }
        double becke[TILE];
#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            becke[i] = 2.;
        }
        for (int atom_j0 = 0; atom_j0 < natm; atom_j0 += TILE) {
            __syncthreads();
            int atom_i = atom_i0 + ty;
            int atom_j = atom_j0 + tx;
            if (atom_i >= natm) atom_i = 0;
            if (atom_j >= natm) atom_j = 0;
            double xi = atom_xi[ty];
            double yi = atom_yi[ty];
            double zi = atom_zi[ty];
            double xj = atm_x[atom_j];
            double yj = atm_y[atom_j];
            double zj = atm_z[atom_j];
            // distance between atom i and atom j
            double dij_inv = rnorm3d(xi-xj, yi-yj, zi-zj);
            a_smem[thread_id] = a[atom_i * natm + atom_j];
            dij_smem[thread_id] = dij_inv;
            if (ty == 0) {
                atom_xj[tx] = xj;
                atom_yj[tx] = yj;
                atom_zj[tx] = zj;
            }
            __syncthreads();

            int j1 = min(natm-atom_j0, TILE);
            double djg[TILE];
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= j1) {
                    break;
                }
                double dx = xg - atom_xj[j];
                double dy = yg - atom_yj[j];
                double dz = zg - atom_zj[j];
                djg[j] = norm3d(dx, dy, dz);
            }

#pragma unroll
            for (int i = 0; i < TILE; ++i) {
                if (i >= i1) {
                    break;
                }
                double becke_i = becke[i];
                double dx = xg - atom_xi[i];
                double dy = yg - atom_yi[i];
                double dz = zg - atom_zi[i];
                double dig = norm3d(dx, dy, dz);
#pragma unroll
                for (int j = 0; j < TILE; ++j) {
                    if (j >= j1) {
                        break;
                    }
                    double dij = dij_smem[i*TILE+j];
                    double aij = a_smem[i*TILE+j];
                    double g = 0.;
                    if (atom_i0+i != atom_j0+j) {
                        g = (dig - djg[j]) * dij;
                    }

                    // atomic radii adjust function
                    double g1 = g*g - 1.0;
                    g += g1 * aij;

                    // becke scheme
                    g = (3.0 - g*g) * g * .5;
                    g = (3.0 - g*g) * g * .5;
                    g = (3.0 - g*g) * g * .5;

                    becke_i *= 0.5 * (1.0 - g);
                }
                becke[i] = becke_i;
            }
        }
        if (grid_id < ngrids) {
#pragma unroll
            for (int i = 0; i < TILE; ++i) {
                if (i >= i1) {
                    break;
                }
                becke_sum += becke[i];
                if (atom_i0+i == atom_id) {
                    becke_self = becke[i];
                }
            }
        }
    }
    if (grid_id < ngrids) {
        weight[grid_id] *= becke_self / becke_sum;
    }
}

__device__ double3 operator+(const double3& v1, const double3& v2) { return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
__device__ double3 operator-(const double3& v1, const double3& v2) { return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
__device__ double3 operator-(const double3& v) { return { -v.x, -v.y, -v.z }; }
__device__ double3& operator+=(double3& v1, const double3& v2) { v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; return v1; }
__device__ double3& operator-=(double3& v1, const double3& v2) { v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; return v1; }
__device__ double3 operator*(const double k, const double3& v) { return { k * v.x, k * v.y, k * v.z }; }
__device__ double norm(const double3& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
__device__ double inv(const double x)
{
    if (x > 1e-14) return 1.0 / x;
    else return 0.0;
}

__device__ double switch_function(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1.0 - mu * mu);
    double s = nu;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = 0.5 * (1.0 - s);
    return s;
}

__device__ double switch_function_dsdmu_over_s(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double dnu_dmu = 1.0 - 2.0 * a_factor * mu;
    const double f1 = (3.0 - nu * nu) * nu * 0.5;
    const double f2 = (3.0 - f1 * f1) * f1 * 0.5;
    const double f3 = (3.0 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1.0 - f3);
    const double dsdmu = -0.5 * 1.5 * (1 - f2 * f2) * 1.5 * (1 - f1 * f1) * 1.5 * (1 - nu * nu) * dnu_dmu;
    return dsdmu * inv(s);
}

__device__ double switch_function_dsdmu_over_s(const double mu, const double a_factor, double* inv_s)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double dnu_dmu = 1.0 - 2.0 * a_factor * mu;
    const double f1 = (3.0 - nu * nu) * nu * 0.5;
    const double f2 = (3.0 - f1 * f1) * f1 * 0.5;
    const double f3 = (3.0 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1.0 - f3);
    (*inv_s) = inv(s);
    const double dsdmu = -0.5 * 1.5 * (1 - f2 * f2) * 1.5 * (1 - f1 * f1) * 1.5 * (1 - nu * nu) * dnu_dmu;
    return dsdmu * (*inv_s);
}

__global__
void GDFTgrid_weight_derivative_kernel(double* __restrict__ dwdG, const double* __restrict__ grid_coords, const double* __restrict__ grid_quadrature_weights,
                                       const double* __restrict__ atm_coords, const double* __restrict__ a_factor,
                                       const int* __restrict__ atm_idx, const int ngrids, const int natm)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_atom = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_grid >= ngrids || i_derivative_atom >= natm)
        return;
    const int i_associated_atom = atm_idx[i_grid];
    if (i_associated_atom < 0) // Pad grid
        return;
    if (i_associated_atom == i_derivative_atom) // Dealt with later by translation invariance.
        return;

    const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };
    const double3 atom_A = { atm_coords[i_associated_atom * 3 + 0], atm_coords[i_associated_atom * 3 + 1], atm_coords[i_associated_atom * 3 + 2] };
    const double3 atom_G = { atm_coords[i_derivative_atom * 3 + 0], atm_coords[i_derivative_atom * 3 + 1], atm_coords[i_derivative_atom * 3 + 2] };
    const double3 Ar = atom_A - grid_r;
    const double3 Gr = atom_G - grid_r;
    const double norm_Ar = norm(Ar);
    const double norm_Gr = norm(Gr);
    const double norm_Gr_1 = inv(norm_Gr);

    double P_A = 1.0;
    double sum_P_B = 0.0;
    double3 sum_dPB_dG = { 0.0, 0.0, 0.0 };
    double P_G = 1.0;
    double3 dPG_dG = { 0.0, 0.0, 0.0 };

    for (int j_atom = 0; j_atom < natm; j_atom++) {
        const double3 atom_B = { atm_coords[j_atom * 3 + 0], atm_coords[j_atom * 3 + 1], atm_coords[j_atom * 3 + 2] };
        const double3 Br = atom_B - grid_r;
        const double norm_Br = norm(Br);

        const double3 AB = atom_A - atom_B;
        const double norm_AB_1 = inv(norm(AB));

        const double mu_AB = (norm_Ar - norm_Br) * norm_AB_1;
        const double a_factor_AB = a_factor[i_associated_atom * natm + j_atom];
        const double s_AB = switch_function(mu_AB, a_factor_AB);

        P_A *= s_AB;

        double P_B = 1.0;

        for (int k_atom = 0; k_atom < natm; k_atom++) {
            const double3 atom_C = { atm_coords[k_atom * 3 + 0], atm_coords[k_atom * 3 + 1], atm_coords[k_atom * 3 + 2] };
            const double3 Cr = atom_C - grid_r;
            const double3 BC = atom_B - atom_C;
            const double norm_Cr = norm(Cr);
            const double norm_BC_1 = inv(norm(BC));

            const double mu_BC = (norm_Br - norm_Cr) * norm_BC_1;
            const double a_factor_BC = a_factor[j_atom * natm + k_atom];
            const double s_BC = switch_function(mu_BC, a_factor_BC);

            P_B *= s_BC;
        }

        sum_P_B += P_B;

        const double3 BG = atom_B - atom_G;
        const double norm_BG_1 = inv(norm(BG));
        const double mu_BG = (norm_Br - norm_Gr) * norm_BG_1;
        const double3 dmuBG_dG = norm_BG_1 * (-norm_Gr_1 * Gr + mu_BG * norm_BG_1 * BG);
        const double a_factor_BG = a_factor[j_atom * natm + i_derivative_atom];
        const double3 dPB_dG = switch_function_dsdmu_over_s(mu_BG, a_factor_BG) * P_B * dmuBG_dG;

        sum_dPB_dG += dPB_dG;

        const double a_factor_GB = a_factor[i_derivative_atom * natm + j_atom];
        const double s_GB = switch_function(-mu_BG, a_factor_GB);
        P_G *= s_GB;

        const double3 dmuGB_dG = -dmuBG_dG;
        dPG_dG += switch_function_dsdmu_over_s(-mu_BG, a_factor_GB) * dmuGB_dG;
    }

    sum_dPB_dG += P_G * dPG_dG;

    const double3 AG = atom_A - atom_G;
    const double norm_AG_1 = inv(norm(AG));
    const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
    const double3 dmuAG_dG = norm_AG_1 * (-norm_Gr_1 * Gr + mu_AG * norm_AG_1 * AG);
    const double a_factor_AG = a_factor[i_associated_atom * natm + i_derivative_atom];
    const double3 dPA_dG = switch_function_dsdmu_over_s(mu_AG, a_factor_AG) * P_A * dmuAG_dG;

    const double quadrature_weight = grid_quadrature_weights[i_grid];
    const double3 dwi_dG = quadrature_weight * (inv(sum_P_B) * dPA_dG - inv(sum_P_B * sum_P_B) * P_A * sum_dPB_dG);

    dwdG[i_derivative_atom * ngrids * 3 + 0 * ngrids + i_grid] = dwi_dG.x;
    dwdG[i_derivative_atom * ngrids * 3 + 1 * ngrids + i_grid] = dwi_dG.y;
    dwdG[i_derivative_atom * ngrids * 3 + 2 * ngrids + i_grid] = dwi_dG.z;
}

typedef struct {
    double3 x;
    double3 y;
    double3 z;
} double9;
__device__ constexpr double9 identity_3 = { 1,0,0, 0,1,0, 0,0,1 };
__device__ double9 operator+(const double9& v1, const double9& v2) { return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
__device__ double9 operator-(const double9& v1, const double9& v2) { return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
__device__ double9 operator-(const double9& v) { return { -v.x, -v.y, -v.z }; }
__device__ double9& operator+=(double9& v1, const double9 v2) { v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; return v1; }
__device__ double9& operator-=(double9& v1, const double9 v2) { v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; return v1; }
__device__ double9 operator*(const double k, const double9& v) { return { k * v.x, k * v.y, k * v.z }; }
__device__ double9 outer(const double3& v1, const double3& v2)
{
    double9 m;
    m.x.x = v1.x * v2.x; m.x.y = v1.x * v2.y; m.x.z = v1.x * v2.z;
    m.y.x = v1.y * v2.x; m.y.y = v1.y * v2.y; m.y.z = v1.y * v2.z;
    m.z.x = v1.z * v2.x; m.z.y = v1.z * v2.y; m.z.z = v1.z * v2.z;
    return m;
}

__device__ double switch_function_d2sdmu2_over_s(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double f1 = (3 - nu * nu) * nu * 0.5;
    const double f2 = (3 - f1 * f1) * f1 * 0.5;
    const double f3 = (3 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1 - f3);
    const double dnu_dmu = 1 - 2 * a_factor * mu;
    const double df1_dnu = 1.5 * (1 - nu * nu);
    const double df2_df1 = 1.5 * (1 - f1 * f1);
    const double df3_df2 = 1.5 * (1 - f2 * f2);
    const double ds_df3 = -0.5;
    const double d2sdmu2 =
        + (0) * (df3_df2 * df2_df1 * df1_dnu * dnu_dmu) * (df3_df2 * df2_df1 * df1_dnu * dnu_dmu)
        + ds_df3 * (-3 * f2) * (df2_df1 * df1_dnu * dnu_dmu) * (df2_df1 * df1_dnu * dnu_dmu)
        + ds_df3 * df3_df2 * (-3 * f1) * (df1_dnu * dnu_dmu) * (df1_dnu * dnu_dmu)
        + ds_df3 * df3_df2 * df2_df1 * (-3 * nu) * (dnu_dmu) * (dnu_dmu)
        + ds_df3 * df3_df2 * df2_df1 * df1_dnu * (-2 * a_factor);
    return d2sdmu2 * inv(s);
}

__global__
void GDFTgrid_weight_second_derivative_offdiagonal_kernel(double* __restrict__ d2w_dG1dG2, const double* __restrict__ grid_coords, const double* __restrict__ grid_quadrature_weights,
                                                          const double* __restrict__ atm_coords, const double* __restrict__ a_factor, const int* __restrict__ atm_idx,
                                                          const double* __restrict__ PB, const double* __restrict__ invsumPB, const int ngrids, const int natm)
{
    const int i_grid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_atom_G = blockIdx.y * blockDim.y + threadIdx.y;
    const int i_atom_H = blockIdx.z * blockDim.z + threadIdx.z;
    if (i_grid >= ngrids || i_atom_G >= natm || i_atom_H >= natm)
        return;
    const int i_atom_A = atm_idx[i_grid];
    if (i_atom_A < 0) // Pad grid
        return;
    if (i_atom_A == i_atom_G || i_atom_A == i_atom_H) // Dealt with later by translation invariance.
        return;
    if (i_atom_G == i_atom_H) // Dealt with later in diagonal kernel.
        return;

    const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };

    const double3 atom_G = { atm_coords[i_atom_G * 3 + 0], atm_coords[i_atom_G * 3 + 1], atm_coords[i_atom_G * 3 + 2] };
    const double3 Gr = atom_G - grid_r;
    const double norm_Gr = norm(Gr);
    const double norm_Gr_1 = inv(norm_Gr);
    double3 sum_dPB_dG = { 0.0, 0.0, 0.0 };
    const double P_G = PB[i_atom_G * ngrids + i_grid];
    double3 dPG_dG = { 0.0, 0.0, 0.0 };

    const double3 atom_H = { atm_coords[i_atom_H * 3 + 0], atm_coords[i_atom_H * 3 + 1], atm_coords[i_atom_H * 3 + 2] };
    const double3 Hr = atom_H - grid_r;
    const double norm_Hr = norm(Hr);
    const double norm_Hr_1 = inv(norm_Hr);
    double3 sum_dPB_dH = { 0.0, 0.0, 0.0 };
    const double P_H = PB[i_atom_H * ngrids + i_grid];
    double3 dPH_dH = { 0.0, 0.0, 0.0 };

    double9 sum_d2PB_dGdH = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int i_atom_B = 0; i_atom_B < natm; i_atom_B++) {
        const double3 atom_B = { atm_coords[i_atom_B * 3 + 0], atm_coords[i_atom_B * 3 + 1], atm_coords[i_atom_B * 3 + 2] };
        const double3 Br = atom_B - grid_r;
        const double norm_Br = norm(Br);
        const double P_B = PB[i_atom_B * ngrids + i_grid];

        // dPB_dG part
        const double3 BG = atom_B - atom_G;
        const double norm_BG_1 = inv(norm(BG));
        const double mu_BG = (norm_Br - norm_Gr) * norm_BG_1;
        const double3 dmuBG_dG = norm_BG_1 * (-norm_Gr_1 * Gr + mu_BG * norm_BG_1 * BG);
        const double a_factor_BG = a_factor[i_atom_B * natm + i_atom_G];
        double inv_sBG = NAN;
        const double dsBG_dmuBG_over_sBG = switch_function_dsdmu_over_s(mu_BG, a_factor_BG, &inv_sBG);
        const double3 dsBG_dG = dsBG_dmuBG_over_sBG * dmuBG_dG;
        const double3 dPB_dG = P_B * dsBG_dG;
        sum_dPB_dG += dPB_dG;

        const double3 dmuGB_dG = -dmuBG_dG;
        // const double a_factor_GB = a_factor[i_atom_G * natm + i_atom_B];
        // const double dsGB_dmuGB_over_sGB = switch_function_dsdmu_over_s(-mu_BG, a_factor_GB);
        // // Note: this requires a_factor_GB = - a_factor_BG
        const double dsGB_dmuGB_over_sGB = dsBG_dmuBG_over_sBG * inv(inv_sBG - 1);
        const double3 dsGB_dG = dsGB_dmuGB_over_sGB * dmuGB_dG;
        dPG_dG += dsGB_dG;

        // dPB_dH part
        const double3 BH = atom_B - atom_H;
        const double norm_BH_1 = inv(norm(BH));
        const double mu_BH = (norm_Br - norm_Hr) * norm_BH_1;
        const double a_factor_BH = a_factor[i_atom_B * natm + i_atom_H];
        double inv_sBH = NAN;
        const double dsBH_dmuBH_over_sBH = switch_function_dsdmu_over_s(mu_BH, a_factor_BH, &inv_sBH);
        const double3 dmuBH_dH = norm_BH_1 * (-norm_Hr_1 * Hr + mu_BH * norm_BH_1 * BH);
        const double3 dsBH_dH = dsBH_dmuBH_over_sBH * dmuBH_dH;
        const double3 dPB_dH = P_B * dsBH_dH;
        sum_dPB_dH += dPB_dH;

        const double3 dmuHB_dH = -dmuBH_dH;
        // const double a_factor_HB = a_factor[i_atom_H * natm + i_atom_B];
        // const double dsHB_dmuHB_over_sHB = switch_function_dsdmu_over_s(-mu_BH, a_factor_HB);
        // // Note: this requires a_factor_HB = - a_factor_BH
        const double dsHB_dmuHB_over_sHB = dsBH_dmuBH_over_sBH * inv(inv_sBH - 1);
        dPH_dH += dsHB_dmuHB_over_sHB * dmuHB_dH;

        // sum_d2PB_dGdH part
        sum_d2PB_dGdH += P_B * outer(dsBG_dG, dsBH_dH);
    }

    sum_dPB_dG += P_G * dPG_dG;
    sum_dPB_dH += P_H * dPH_dH;

    const double3 GH = atom_G - atom_H;
    const double norm_GH_1 = inv(norm(GH));

    const double mu_GH = (norm_Gr - norm_Hr) * norm_GH_1;
    const double3 dmuGH_dG = norm_GH_1 * ( norm_Gr_1 * Gr - mu_GH * norm_GH_1 * GH);
    const double3 dmuGH_dH = norm_GH_1 * (-norm_Hr_1 * Hr + mu_GH * norm_GH_1 * GH);
    const double a_factor_GH = a_factor[i_atom_G * natm + i_atom_H];
    const double3 dsGH_dH = switch_function_dsdmu_over_s(mu_GH, a_factor_GH) * dmuGH_dH;
    const double3 dsGH_dG = switch_function_dsdmu_over_s(mu_GH, a_factor_GH) * dmuGH_dG;

    const double9 d2muGH_dGdH = (norm_GH_1 * norm_GH_1 * norm_GH_1 * norm_Gr_1) * outer(Gr, GH)
                              + (norm_GH_1 * norm_GH_1 * norm_GH_1 * norm_Hr_1) * outer(GH, Hr)
                              + (-3 * mu_GH * norm_GH_1 * norm_GH_1 * norm_GH_1 * norm_GH_1) * outer(GH, GH)
                              + (mu_GH * norm_GH_1 * norm_GH_1) * identity_3;
    const double9 dsdmu_dmu2GHdGdH = switch_function_dsdmu_over_s(mu_GH, a_factor_GH) * d2muGH_dGdH;
    const double9 d2sdmu2_dmuGHdGdH = switch_function_d2sdmu2_over_s(mu_GH, a_factor_GH) * outer(dmuGH_dG, dmuGH_dH);
    const double9 d2sGH_dGdH = dsdmu_dmu2GHdGdH + d2sdmu2_dmuGHdGdH;

    const double9 d2PG_dGdH = P_G * (outer(dPG_dG - dsGH_dG, dsGH_dH) + d2sGH_dGdH);
    sum_d2PB_dGdH += d2PG_dGdH;

    const double3 dmuHG_dG = -dmuGH_dG;
    const double3 dmuHG_dH = -dmuGH_dH;
    const double a_factor_HG = a_factor[i_atom_H * natm + i_atom_G];
    const double3 dsHG_dG = switch_function_dsdmu_over_s(-mu_GH, a_factor_HG) * dmuHG_dG;
    const double3 dsHG_dH = switch_function_dsdmu_over_s(-mu_GH, a_factor_HG) * dmuHG_dH;

    const double9 d2muHG_dGdH = -d2muGH_dGdH;
    const double9 dsdmu_dmu2HGdGdH = switch_function_dsdmu_over_s(-mu_GH, a_factor_HG) * d2muHG_dGdH;
    const double9 d2sdmu2_dmuHGdGdH = switch_function_d2sdmu2_over_s(-mu_GH, a_factor_HG) * outer(dmuHG_dG, dmuHG_dH);
    const double9 d2sHG_dGdH = dsdmu_dmu2HGdGdH + d2sdmu2_dmuHGdGdH;

    const double9 d2PH_dGdH = P_H * (outer(dsHG_dG, dPH_dH - dsHG_dH) + d2sHG_dGdH);
    sum_d2PB_dGdH += d2PH_dGdH;

    const double3 atom_A = { atm_coords[i_atom_A * 3 + 0], atm_coords[i_atom_A * 3 + 1], atm_coords[i_atom_A * 3 + 2] };
    const double3 Ar = atom_A - grid_r;
    const double norm_Ar = norm(Ar);
    const double P_A = PB[i_atom_A * ngrids + i_grid];

    // dPA_dG part
    const double3 AG = atom_A - atom_G;
    const double norm_AG_1 = inv(norm(AG));
    const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
    const double3 dmuAG_dG = norm_AG_1 * (-norm_Gr_1 * Gr + mu_AG * norm_AG_1 * AG);
    const double a_factor_AG = a_factor[i_atom_A * natm + i_atom_G];
    const double3 dsAG_dG = switch_function_dsdmu_over_s(mu_AG, a_factor_AG) * dmuAG_dG;
    const double3 dPA_dG = P_A * dsAG_dG;

    // dPA_dH part
    const double3 AH = atom_A - atom_H;
    const double norm_AH_1 = inv(norm(AH));
    const double mu_AH = (norm_Ar - norm_Hr) * norm_AH_1;
    const double3 dmuAH_dH = norm_AH_1 * (-norm_Hr_1 * Hr + mu_AH * norm_AH_1 * AH);
    const double a_factor_AH = a_factor[i_atom_A * natm + i_atom_H];
    const double3 dsAH_dH = switch_function_dsdmu_over_s(mu_AH, a_factor_AH) * dmuAH_dH;
    const double3 dPA_dH = P_A * dsAH_dH;

    // d2PA_dGdH part
    const double9 d2PA_dGdH = P_A * outer(dsAG_dG, dsAH_dH);

    const double sum_P_B_1 = invsumPB[i_grid];
    double9 d2wi_dGdH = { 0,0,0, 0,0,0, 0,0,0 };
    d2wi_dGdH += sum_P_B_1 * d2PA_dGdH;
    d2wi_dGdH -= (sum_P_B_1 * sum_P_B_1) * outer(sum_dPB_dG, dPA_dH);
    d2wi_dGdH -= (sum_P_B_1 * sum_P_B_1) * outer(dPA_dG, sum_dPB_dH);
    d2wi_dGdH -= (P_A * sum_P_B_1 * sum_P_B_1) * sum_d2PB_dGdH;
    d2wi_dGdH += (2 * P_A * sum_P_B_1 * sum_P_B_1 * sum_P_B_1) * outer(sum_dPB_dG, sum_dPB_dH);

    const double quadrature_weight = grid_quadrature_weights[i_grid];
    d2wi_dGdH = quadrature_weight * d2wi_dGdH;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 0 * ngrids + i_grid] = d2wi_dGdH.x.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 1 * ngrids + i_grid] = d2wi_dGdH.x.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 2 * ngrids + i_grid] = d2wi_dGdH.x.z;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 3 * ngrids + i_grid] = d2wi_dGdH.y.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 4 * ngrids + i_grid] = d2wi_dGdH.y.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 5 * ngrids + i_grid] = d2wi_dGdH.y.z;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 6 * ngrids + i_grid] = d2wi_dGdH.z.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 7 * ngrids + i_grid] = d2wi_dGdH.z.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_H * 9 * ngrids + 8 * ngrids + i_grid] = d2wi_dGdH.z.z;
}

__global__
void GDFTgrid_weight_second_derivative_diagonal_kernel(double* __restrict__ d2w_dG1dG2, const double* __restrict__ grid_coords, const double* __restrict__ grid_quadrature_weights,
                                                       const double* __restrict__ atm_coords, const double* __restrict__ a_factor, const int* __restrict__ atm_idx,
                                                       const double* __restrict__ PB, const double* __restrict__ invsumPB, const int ngrids, const int natm)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_atom_G = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_grid >= ngrids || i_atom_G >= natm)
        return;
    const int i_atom_A = atm_idx[i_grid];
    if (i_atom_A < 0) // Pad grid
        return;
    if (i_atom_A == i_atom_G) // Dealt with later by translation invariance.
        return;

    const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };

    const double3 atom_G = { atm_coords[i_atom_G * 3 + 0], atm_coords[i_atom_G * 3 + 1], atm_coords[i_atom_G * 3 + 2] };
    const double3 Gr = atom_G - grid_r;
    const double norm_Gr = norm(Gr);
    const double norm_Gr_1 = inv(norm_Gr);
    double3 sum_dPB_dG = { 0.0, 0.0, 0.0 };
    const double P_G = PB[i_atom_G * ngrids + i_grid];
    double3 dPG_dG = { 0.0, 0.0, 0.0 };

    double9 sum_d2PB_dG2 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double9 d2PG_dG2 = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int i_atom_B = 0; i_atom_B < natm; i_atom_B++) {
        const double3 atom_B = { atm_coords[i_atom_B * 3 + 0], atm_coords[i_atom_B * 3 + 1], atm_coords[i_atom_B * 3 + 2] };
        const double3 Br = atom_B - grid_r;
        const double norm_Br = norm(Br);
        const double P_B = PB[i_atom_B * ngrids + i_grid];

        // dPB_dG part
        const double3 BG = atom_B - atom_G;
        const double norm_BG_1 = inv(norm(BG));
        const double mu_BG = (norm_Br - norm_Gr) * norm_BG_1;
        const double3 dmuBG_dG = norm_BG_1 * (-norm_Gr_1 * Gr + mu_BG * norm_BG_1 * BG);
        const double a_factor_BG = a_factor[i_atom_B * natm + i_atom_G];
        const double3 dsBG_dG = switch_function_dsdmu_over_s(mu_BG, a_factor_BG) * dmuBG_dG;
        const double3 dPB_dG = P_B * dsBG_dG;
        sum_dPB_dG += dPB_dG;

        const double a_factor_GB = a_factor[i_atom_G * natm + i_atom_B];
        const double3 dmuGB_dG = -dmuBG_dG;
        const double3 dsGB_dG = switch_function_dsdmu_over_s(-mu_BG, a_factor_GB) * dmuGB_dG;
        dPG_dG += dsGB_dG;

        // sum_d2PB_dG2 part
        const double9 d2mu_BGdG2 = (-norm_BG_1 * norm_BG_1 * norm_BG_1 * norm_Gr_1) * (outer(BG, Gr) + outer(Gr, BG))
                                 + (norm_BG_1 * norm_Gr_1 * norm_Gr_1 * norm_Gr_1) * outer(Gr, Gr)
                                 + (3 * mu_BG * norm_BG_1 * norm_BG_1 * norm_BG_1 * norm_BG_1) * outer(BG, BG)
                                 + (-norm_BG_1 * norm_Gr_1 - mu_BG * norm_BG_1 * norm_BG_1) * identity_3;
        const double9 dsdmu_dmuBG2dG2 = switch_function_dsdmu_over_s(mu_BG, a_factor_BG) * d2mu_BGdG2;
        const double9 d2sdmu2_dmuBGdG_2 = switch_function_d2sdmu2_over_s(mu_BG, a_factor_BG) * outer(dmuBG_dG, dmuBG_dG);
        sum_d2PB_dG2 += P_B * (dsdmu_dmuBG2dG2 + d2sdmu2_dmuBGdG_2);

        // d2PG_dG2 part
        const double9 d2mu_GBdG2 = -d2mu_BGdG2;
        const double9 dsdmu_dmuGB2dG2 = switch_function_dsdmu_over_s(-mu_BG, a_factor_GB) * d2mu_GBdG2;
        const double9 d2sdmu2_dmuGBdG_2 = switch_function_d2sdmu2_over_s(-mu_BG, a_factor_GB) * outer(dmuGB_dG, dmuGB_dG);
        d2PG_dG2 += (dsdmu_dmuGB2dG2 + d2sdmu2_dmuGBdG_2);
        d2PG_dG2 -= outer(dsGB_dG, dsGB_dG);
    }

    sum_dPB_dG += P_G * dPG_dG;

    d2PG_dG2 += outer(dPG_dG, dPG_dG);
    sum_d2PB_dG2 += P_G * d2PG_dG2;

    const double3 atom_A = { atm_coords[i_atom_A * 3 + 0], atm_coords[i_atom_A * 3 + 1], atm_coords[i_atom_A * 3 + 2] };
    const double3 Ar = atom_A - grid_r;
    const double norm_Ar = norm(Ar);
    const double P_A = PB[i_atom_A * ngrids + i_grid];

    // dPA_dG part
    const double3 AG = atom_A - atom_G;
    const double norm_AG_1 = inv(norm(AG));
    const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
    const double3 dmuAG_dG = norm_AG_1 * (-norm_Gr_1 * Gr + mu_AG * norm_AG_1 * AG);
    const double a_factor_AG = a_factor[i_atom_A * natm + i_atom_G];
    const double3 dsAG_dG = switch_function_dsdmu_over_s(mu_AG, a_factor_AG) * dmuAG_dG;
    const double3 dPA_dG = P_A * dsAG_dG;

    // d2PA_dG2 part
    const double9 d2muAGdG2 = (-norm_AG_1 * norm_AG_1 * norm_AG_1 * norm_Gr_1) * (outer(AG, Gr) + outer(Gr, AG))
                            + (norm_AG_1 * norm_Gr_1 * norm_Gr_1 * norm_Gr_1) * outer(Gr, Gr)
                            + (3 * mu_AG * norm_AG_1 * norm_AG_1 * norm_AG_1 * norm_AG_1) * outer(AG, AG)
                            + (-norm_AG_1 * norm_Gr_1 - mu_AG * norm_AG_1 * norm_AG_1) * identity_3;
    const double9 dsdmu_dmu2dG2 = switch_function_dsdmu_over_s(mu_AG, a_factor_AG) * d2muAGdG2;
    const double9 d2sdmu2_dmuAGdG_2 = switch_function_d2sdmu2_over_s(mu_AG, a_factor_AG) * outer(dmuAG_dG, dmuAG_dG);
    const double9 d2PA_dG2 = P_A * (dsdmu_dmu2dG2 + d2sdmu2_dmuAGdG_2);

    const double sum_P_B_1 = invsumPB[i_grid];
    double9 d2wi_dG2 = { 0,0,0, 0,0,0, 0,0,0 };
    d2wi_dG2 += sum_P_B_1 * d2PA_dG2;
    d2wi_dG2 -= (sum_P_B_1 * sum_P_B_1) * outer(sum_dPB_dG, dPA_dG);
    d2wi_dG2 -= (sum_P_B_1 * sum_P_B_1) * outer(dPA_dG, sum_dPB_dG);
    d2wi_dG2 -= (P_A * sum_P_B_1 * sum_P_B_1) * sum_d2PB_dG2;
    d2wi_dG2 += (2 * P_A * sum_P_B_1 * sum_P_B_1 * sum_P_B_1) * outer(sum_dPB_dG, sum_dPB_dG);

    const double quadrature_weight = grid_quadrature_weights[i_grid];
    d2wi_dG2 = quadrature_weight * d2wi_dG2;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 0 * ngrids + i_grid] = d2wi_dG2.x.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 1 * ngrids + i_grid] = d2wi_dG2.x.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 2 * ngrids + i_grid] = d2wi_dG2.x.z;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 3 * ngrids + i_grid] = d2wi_dG2.y.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 4 * ngrids + i_grid] = d2wi_dG2.y.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 5 * ngrids + i_grid] = d2wi_dG2.y.z;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 6 * ngrids + i_grid] = d2wi_dG2.z.x;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 7 * ngrids + i_grid] = d2wi_dG2.z.y;
    d2w_dG1dG2[i_atom_G * natm * 9 * ngrids + i_atom_G * 9 * ngrids + 8 * ngrids + i_grid] = d2wi_dG2.z.z;
}

__global__
void GDFTgrid_becke_eval_PB_kernel(double* __restrict__ PB, const double* __restrict__ grid_coords,
                                   const double* __restrict__ atm_coords, const double* __restrict__ a_factor,
                                   const int ngrids, const int natm)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_atom_B = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_grid >= ngrids || i_atom_B >= natm)
        return;

    const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };
    const double3 atom_B = { atm_coords[i_atom_B * 3 + 0], atm_coords[i_atom_B * 3 + 1], atm_coords[i_atom_B * 3 + 2] };
    const double3 Br = atom_B - grid_r;
    const double norm_Br = norm(Br);

    // P_B part
    double P_B = 1.0;
    for (int i_atom_C = 0; i_atom_C < natm; i_atom_C++) {
        const double3 atom_C = { atm_coords[i_atom_C * 3 + 0], atm_coords[i_atom_C * 3 + 1], atm_coords[i_atom_C * 3 + 2] };
        const double3 Cr = atom_C - grid_r;
        const double3 BC = atom_B - atom_C;
        const double norm_Cr = norm(Cr);
        const double norm_BC_1 = inv(norm(BC));

        const double mu_BC = (norm_Br - norm_Cr) * norm_BC_1;
        const double a_factor_BC = a_factor[i_atom_B * natm + i_atom_C];
        const double s_BC = switch_function(mu_BC, a_factor_BC);

        P_B *= s_BC;
    }

    PB[i_atom_B * ngrids + i_grid] = P_B;
}

__global__
void GDFTgroup_grids_kernel(int* group_ids, const double* atom_coords, const double* coords, int natm, int ngrids){
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;

    double xg = coords[grid_id];
    double yg = coords[grid_id + ngrids];
    double zg = coords[grid_id + 2*ngrids];

    double r2min = 1e30;
    int idx = 0;
    const int tx = threadIdx.x;
    double __shared__ x_atom[NATOM_PER_BLOCK];
    double __shared__ y_atom[NATOM_PER_BLOCK];
    double __shared__ z_atom[NATOM_PER_BLOCK];
    for (int j = 0; j < natm; j+=blockDim.x){
        int atom_idx = j + tx;
        if (atom_idx < natm){
            // distance between atom i and atom j
            x_atom[tx] = atom_coords[atom_idx];
            y_atom[tx] = atom_coords[atom_idx + natm];
            z_atom[tx] = atom_coords[atom_idx + 2*natm];
        }
        __syncthreads();

        for (int l = 0, M = min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
            int atom_j = j + l;
            double xa = x_atom[l] - xg;
            double ya = y_atom[l] - yg;
            double za = z_atom[l] - zg;
            double r2 = xa*xa + ya*ya + za*za;
            if (r2 < r2min){
                r2min = r2;
                idx = atom_j;
            }
        }
    }
    group_ids[grid_id] = idx;
}

extern "C"{
__host__
int GDFTbecke_partition_weights(double *weights, double *coords, double *atm_coords,
                                double *a, int *atm_idx, int ngrids, int natm)
{
    dim3 threads(TILE, TILE);
    int blocks = (ngrids+TILE*TILE-1)/(TILE*TILE);
    GDFTgrid_weight_kernel<<<blocks, threads>>>(weights, coords, atm_coords, a,
                                                atm_idx, ngrids, natm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error in GDFTgrid_weight: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int GDFTbecke_partition_weight_derivative(double *dwdG, const double *grid_coords, const double *grid_quadrature_weights,
                                          const double *atm_coords, const double *a_factor,
                                          const int *atm_idx, const int ngrids, const int natm)
{
    const dim3 threads(TILE, TILE);
    const dim3 blocks((ngrids + TILE - 1) / TILE,
                      (natm + TILE - 1) / TILE);
    GDFTgrid_weight_derivative_kernel<<<blocks, threads>>>(dwdG, grid_coords, grid_quadrature_weights,
                                                           atm_coords, a_factor, atm_idx, ngrids, natm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error in GDFTgrid_weight_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int GDFTbecke_partition_weight_second_derivative(double *d2w_dG1dG2, const double *grid_coords, const double *grid_quadrature_weights,
                                                 const double *atm_coords, const double *a_factor, const int *atm_idx,
                                                 const double *PB, const double *invsumPB, const int ngrids, const int natm)
{
    {
        constexpr int n_grid_per_block = 16;
        constexpr int n_atom_per_block = 4;
        const dim3 threads(n_grid_per_block, n_atom_per_block, n_atom_per_block);
        const dim3 blocks((ngrids + n_grid_per_block - 1) / n_grid_per_block,
                          (natm   + n_atom_per_block - 1) / n_atom_per_block,
                          (natm   + n_atom_per_block - 1) / n_atom_per_block);
        GDFTgrid_weight_second_derivative_offdiagonal_kernel<<<blocks, threads>>>(d2w_dG1dG2, grid_coords, grid_quadrature_weights,
                                                                                  atm_coords, a_factor, atm_idx, PB, invsumPB, ngrids, natm);
    }
    {
        constexpr int n_grid_per_block = 64;
        constexpr int n_atom_per_block = 4;
        const dim3 threads(n_grid_per_block, n_atom_per_block);
        const dim3 blocks((ngrids + n_grid_per_block - 1) / n_grid_per_block,
                          (natm   + n_atom_per_block - 1) / n_atom_per_block);
        GDFTgrid_weight_second_derivative_diagonal_kernel<<<blocks, threads>>>(d2w_dG1dG2, grid_coords, grid_quadrature_weights,
                                                                               atm_coords, a_factor, atm_idx, PB, invsumPB, ngrids, natm);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error in GDFTgrid_weight_second_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int GDFTbecke_eval_PB(double *PB, const double *grid_coords,
                      const double *atm_coords, const double *a_factor,
                      const int ngrids, const int natm)
{
    {
        constexpr int n_grid_per_block = 64;
        constexpr int n_atom_per_block = 4;
        const dim3 threads(n_grid_per_block, n_atom_per_block);
        const dim3 blocks((ngrids + n_grid_per_block - 1) / n_grid_per_block,
                          (natm   + n_atom_per_block - 1) / n_atom_per_block);
        GDFTgrid_becke_eval_PB_kernel<<<blocks, threads>>>(PB, grid_coords, atm_coords, a_factor, ngrids, natm);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error in GDFTbecke_eval_PB: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int GDFTgroup_grids(cudaStream_t stream, int* group_ids, const double* atom_coords, const double* coords,
    int natm, int ngrids){
    if (ngrids % NATOM_PER_BLOCK != 0){
        fprintf(stderr, "CUDA Error of gen grids: grids alignment must be %d.", NATOM_PER_BLOCK);
        return 1;
    }
    dim3 threads(NATOM_PER_BLOCK);
    dim3 blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    GDFTgroup_grids_kernel<<<blocks, threads, 0, stream>>>(group_ids, atom_coords, coords, natm, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error of group grids: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}
