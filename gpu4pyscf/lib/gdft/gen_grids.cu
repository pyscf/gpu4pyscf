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
__device__ double3 operator*(const double3& v, const double k) { return { k * v.x, k * v.y, k * v.z }; }
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

__device__ double switch_function_dmuds_over_s(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double dnu_dmu = 1.0 - 2.0 * a_factor * mu;
    const double f1 = (3.0 - nu * nu) * nu * 0.5;
    const double f2 = (3.0 - f1 * f1) * f1 * 0.5;
    const double f3 = (3.0 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1.0 - f3);
    const double dmuds = -0.5 * 1.5 * (1 - f2 * f2) * 1.5 * (1 - f1 * f1) * 1.5 * (1 - nu * nu) * dnu_dmu;
    return dmuds * inv(s);
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
        const double3 dPB_dG = switch_function_dmuds_over_s(mu_BG, a_factor_BG) * P_B * dmuBG_dG;

        sum_dPB_dG += dPB_dG;

        const double a_factor_GB = a_factor[i_derivative_atom * natm + j_atom];
        const double s_GB = switch_function(-mu_BG, a_factor_GB);
        P_G *= s_GB;

        const double norm_Br_1 = inv(norm_Br);
        const double3 dmuGB_dG = -dmuBG_dG;
        dPG_dG += switch_function_dmuds_over_s(-mu_BG, a_factor_GB) * dmuGB_dG;
    }

    sum_dPB_dG += P_G * dPG_dG;

    const double3 AG = atom_A - atom_G;
    const double norm_AG_1 = inv(norm(AG));
    const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
    const double3 dmuAG_dG = norm_AG_1 * (-norm_Gr_1 * Gr + mu_AG * norm_AG_1 * AG);
    const double a_factor_AG = a_factor[i_associated_atom * natm + i_derivative_atom];
    const double3 dPA_dG = switch_function_dmuds_over_s(mu_AG, a_factor_AG) * P_A * dmuAG_dG;

    const double quadrature_weight = grid_quadrature_weights[i_grid];
    const double3 dwi_dG = quadrature_weight * (inv(sum_P_B) * dPA_dG - inv(sum_P_B * sum_P_B) * P_A * sum_dPB_dG);

    dwdG[i_derivative_atom * ngrids * 3 + 0 * ngrids + i_grid] = dwi_dG.x;
    dwdG[i_derivative_atom * ngrids * 3 + 1 * ngrids + i_grid] = dwi_dG.y;
    dwdG[i_derivative_atom * ngrids * 3 + 2 * ngrids + i_grid] = dwi_dG.z;
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
