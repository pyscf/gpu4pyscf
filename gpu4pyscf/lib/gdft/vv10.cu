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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gint/gint.h"
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define NG_PER_BLOCK      128
#define NG_PER_THREADS    1

__global__
static void vv10_kernel(double *Fvec, double *Uvec, double *Wvec,
    const double *vvcoords, const double *coords,
    const double *W0p, const double *W0, const double *K,
    const double *Kp, const double *RpW,
    int vvngrids, int ngrids)
{
    // grid id
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;
    double xi, yi, zi;
    double W0i, Ki;
    if (active){
        xi = coords[grid_id];
        yi = coords[ngrids + grid_id];
        zi = coords[2*ngrids + grid_id];
        W0i = W0[grid_id];
        Ki = K[grid_id];
    }

    double F = 0.0;
    double U = 0.0;
    double W = 0.0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    //__shared__ double xj_smem[NG_PER_BLOCK];
    //__shared__ double yj_smem[NG_PER_BLOCK];
    //__shared__ double zj_smem[NG_PER_BLOCK];
    //__shared__ double Kp_smem[NG_PER_BLOCK];
    //__shared__ double W0p_smem[NG_PER_BLOCK];
    //__shared__ double RpW_smem[NG_PER_BLOCK];

    __shared__ double3 xj_t[NG_PER_BLOCK];
    __shared__ double3 kp_t[NG_PER_BLOCK];

    const int tx = threadIdx.x;

    for (int j = 0; j < vvngrids; j+=blockDim.x) {
        int idx = j + tx;
        if (idx < vvngrids){
            //xj_smem[tx] = xj[idx];
            //yj_smem[tx] = yj[idx];
            //zj_smem[tx] = zj[idx];
            //Kp_smem[tx] = Kp[idx];
            //W0p_smem[tx] = W0p[idx];
            //RpW_smem[tx] = RpW[idx];

            xj_t[tx] = {xj[idx], yj[idx], zj[idx]};
            kp_t[tx] = {Kp[idx], W0p[idx], RpW[idx]};
        }
        __syncthreads();

        for (int l = 0, M = min(NG_PER_BLOCK, vvngrids - j); l < M; ++l){
            // about 24 operations for each pair
            //double DX = xj_smem[l] - xi;//xj_tmp.x - xi;
            //double DY = yj_smem[l] - yi;//xj_tmp.y - yi;
            //double DZ = zj_smem[l] - zi;//xj_tmp.z - zi;

            double3 xj_tmp = xj_t[l];
            double DX = xj_tmp.x - xi;
            double DY = xj_tmp.y - yi;
            double DZ = xj_tmp.z - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            double3 kp_tmp = kp_t[l]; // (Kpj, W0pj, RpWj)
            double gp = R2*kp_tmp.y + kp_tmp.x;
            //double gp = R2 * W0p_smem[l] + Kp_smem[l];//R2*kp_tmp.y + kp_tmp.x;
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggt = g*gt;
            double g_gt = g + gt;
            //double T = RpW_smem[l] / (gp*ggt*ggt);//kp_tmp.z / (gp*ggt*ggt);
            double T = kp_tmp.z / (gp*ggt*ggt);

            F += T * ggt;
            U += T * g_gt;
            W += T * R2 * g_gt;
            /*
            double ggt = g * gt;
            double ggt2 = ggt * ggt;
            double T = kp_tmp.z/(gp*ggt2);

            F += T * ggt;
            T *= (g + gt);
            U += T;
            W += T * R2;
            */
        }
        __syncthreads();
    }
    if(active){
        Fvec[grid_id] = F * -1.5;
        Uvec[grid_id] = U;
        Wvec[grid_id] = W;
    }

}

__global__
static void vv10_grad_kernel(double *Fvec, const double *vvcoords, const double *coords,
    const double *W0p, const double *W0,
    const double *K, const double *Kp, const double *RpW,
    int vvngrids, int ngrids)
{
    const int outer_grid_id = blockIdx.x * NG_PER_BLOCK + threadIdx.x;
    const bool active = outer_grid_id < ngrids;

    double xi, yi, zi, W0i, Ki;
    if (active) {
        xi  = coords[outer_grid_id * 3    ];
        yi  = coords[outer_grid_id * 3 + 1];
        zi  = coords[outer_grid_id * 3 + 2];
        W0i = W0[outer_grid_id];
        Ki  =  K[outer_grid_id];
    }
    double FX = 0;
    double FY = 0;
    double FZ = 0;

    __shared__ double3 xj_t[NG_PER_BLOCK];
    __shared__ double3 kp_t[NG_PER_BLOCK];

    for (int j = 0; j < vvngrids; j += NG_PER_BLOCK) {
        const int idx = j + threadIdx.x;
        if (idx < vvngrids) {
            const double *xyzj = vvcoords + idx * 3;
            xj_t[threadIdx.x] = { xyzj[0], xyzj[1], xyzj[2] };
            kp_t[threadIdx.x] = { Kp[idx], W0p[idx], RpW[idx] };
        }
        __syncthreads();

        const int M = min(NG_PER_BLOCK, vvngrids - j);
        for (int l = 0; l < M; ++l) {
            const double3 xj_tmp = xj_t[l];
            const double DX = xj_tmp.x - xi;
            const double DY = xj_tmp.y - yi;
            const double DZ = xj_tmp.z - zi;
            const double R2 = DX*DX + DY*DY + DZ*DZ;

            const double3 kp_tmp = kp_t[l];
            const double Kpj  = kp_tmp.x;
            const double W0pj = kp_tmp.y;
            const double RpWj = kp_tmp.z;
            const double gp = R2*W0pj + Kpj;
            const double g  = R2*W0i + Ki;
            const double gt = g + gp;
            const double T = RpWj / (g*gp*gt);
            const double Q = T * (W0i/g + W0pj/gp + (W0i+W0pj)/gt);

            FX += Q * DX;
            FY += Q * DY;
            FZ += Q * DZ;
        }
         __syncthreads();
    }

    if (active) {
        Fvec[outer_grid_id * 3    ] = FX * -3;
        Fvec[outer_grid_id * 3 + 1] = FY * -3;
        Fvec[outer_grid_id * 3 + 2] = FZ * -3;
    }
}

__global__
static void vv10_hess_eval_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                         const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                         const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                         const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double U_i = 0;
    double W_i = 0;
    double A_i = 0;
    double B_i = 0;
    double C_i = 0;
    double E_i = 0;

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double weight_j = grid_weight[j];
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

        const double E_ij = weight_j * rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum_1 + g_ij_1);
        const double W_ij = U_ij * r_ij2;
        const double A_ij = E_ij * (g_sum_1 * g_sum_1 + g_sum_1 * g_ij_1 + g_ij_1 * g_ij_1);
        const double B_ij = A_ij * r_ij2;
        const double C_ij = B_ij * r_ij2;

        U_i += U_ij;
        W_i += W_ij;
        A_i += A_ij;
        B_i += B_ij;
        C_i += C_ij;
        E_i += E_ij;
    }

    U[i] = -U_i;
    W[i] = -W_i;
    A[i] = 2 * A_i;
    B[i] = 2 * B_i;
    C[i] = 2 * C_i;
    E[i] = E_i;
}

__global__
static void vv10_hess_eval_omega_derivative_kernel(double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   double* __restrict__ d2omega_drho2, double* __restrict__ d2omega_dgamma2, double* __restrict__ d2omega_drho_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * NG_PER_BLOCK + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_3 = rho_1 * rho_2;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega = sqrt(omega2);
    const double omega_1 = 1 / omega;

    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;

    const double omega_3 = omega_1 / omega2;
    d2omega_drho2[i] = (-0.25 * four_pi_over_three * four_pi_over_three
                        + 12 * four_pi_over_three * C_factor * gamma2 * rho_5
                        + 6 * C_factor * C_factor * gamma2 * gamma2 * rho_5 * rho_5) * omega_3;
    d2omega_dgamma2[i] = four_pi_over_three * C_factor * rho_3 * omega_3;
    d2omega_drho_dgamma[i] = -C_factor * gamma_i * (4.5 * four_pi_over_three * rho_4
                                                    + 2 * C_factor * gamma2 * rho_4 * rho_5) * omega_3;
}

template <int n_trial_per_thread>
__global__
static void vv10_hess_eval_f_t_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                      const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                      const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                      const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                                      const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                      const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                                      const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                      const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_trial_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_trial_per_thread;
    if (i >= ngrids || i_trial_start >= ntrial)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    double f_rho_t_i[n_trial_per_thread] {0};
    double f_gamma_t_i[n_trial_per_thread] {0};
    // #pragma unroll
    // for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
    //     f_rho_t_i  [i_trial] = 0;
    //     f_gamma_t_i[i_trial] = 0;
    // }

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double domega_drho_j = domega_drho[j];
        const double domega_dgamma_j = domega_dgamma[j];
        const double dkappa_drho_j = dkappa_drho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

        const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
        const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
        const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_1 * g_ji_1);

        const double f_rho_rho_ij = Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                              - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                              - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
        const double f_gamma_rho_ij = rho_i * domega_dgamma_i * r_ij2 * Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
        const double f_rho_gamma_ij = rho_j * domega_dgamma_j * r_ij2 * Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
        const double f_gamma_gamma_ij = rho_i * rho_j * domega_dgamma_i * domega_dgamma_j * r_ij2 * r_ij2 * Phi_ij * d2Phi_dgij_dgji_over_Phi;

        const double weight_j = grid_weight[j];

        #pragma unroll
        for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
            if (i_trial + i_trial_start >= ntrial) continue;
            const double   rho_t_j =   rho_t[(i_trial + i_trial_start) * ngrids + j];
            const double gamma_t_j = gamma_t[(i_trial + i_trial_start) * ngrids + j];
            f_rho_t_i  [i_trial] += weight_j * (  f_rho_rho_ij * rho_t_j +   f_rho_gamma_ij * gamma_t_j);
            f_gamma_t_i[i_trial] += weight_j * (f_gamma_rho_ij * rho_t_j + f_gamma_gamma_ij * gamma_t_j);
        }
    }

    const double U_i = U[i];
    const double W_i = W[i];
    const double A_i = A[i];
    const double B_i = B[i];
    const double C_i = C[i];
    const double d2omega_drho2_i = d2omega_drho2[i];
    const double d2omega_dgamma2_i = d2omega_dgamma2[i];
    const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
    const double d2kappa_drho2_i = d2kappa_drho2[i];

    const double f_rho_rho_ii = 2 * domega_drho_i * W_i + 2 * dkappa_drho_i * U_i
                                + rho_i * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i * dkappa_drho_i * A_i
                                           + domega_drho_i * domega_drho_i * C_i + 2 * domega_drho_i * dkappa_drho_i * B_i);
    const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i * (d2omega_drho_dgamma_i * W_i
                                                                   + domega_dgamma_i * (dkappa_drho_i * B_i + domega_drho_i * C_i));
    const double f_rho_gamma_ii = f_gamma_rho_ii;
    const double f_gamma_gamma_ii = rho_i * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

    #pragma unroll
    for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
        if (i_trial + i_trial_start >= ntrial) continue;
        const double rho_t_i   =   rho_t[(i_trial + i_trial_start) * ngrids + i];
        const double gamma_t_i = gamma_t[(i_trial + i_trial_start) * ngrids + i];
        f_rho_t_i  [i_trial] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
        f_gamma_t_i[i_trial] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);

        f_rho_t  [(i_trial + i_trial_start) * ngrids + i] = f_rho_t_i  [i_trial];
        f_gamma_t[(i_trial + i_trial_start) * ngrids + i] = f_gamma_t_i[i_trial];
    }
}

__global__
static void vv10_hess_eval_EUW_grid_response_kernel(double* __restrict__ Egr, double* __restrict__ Ugr, double* __restrict__ Wgr,
                                                    const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                    const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                    const int* __restrict__ grid_associated_atom,
                                                    const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        return;
    }
    const bool i_in_B = (i_associated_atom == B_atom);

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };
    double3 Ugr_i = { 0, 0, 0 };
    double3 Wgr_i = { 0, 0, 0 };

    for (int j = 0; j < ngrids; j++) {
        const int j_associated_atom = grid_associated_atom[j];
        if (j_associated_atom < 0)
            continue;
        const int j_in_B = (j_associated_atom == B_atom);
        if (!i_in_B && !j_in_B)
            continue;
        if (i_in_B && j_in_B)
            continue;

        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double weight_j = grid_weight[j];
        const double rho_j = rho[j];

        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

        const double E_ij = weight_j * rho_j * Phi_ij;
        const double dPhi_drj_over_Phi = omega_i * g_ij_1 + omega_j * g_ji_1 + (omega_i + omega_j) * g_sum_1;
        const double d2Phi_dgij_drj_over_Phi = omega_i * g_ij_1 * g_ij_1 + (omega_i + omega_j) * g_sum_1 * g_sum_1;
        const double dPhi_dgij_over_Phi = g_sum_1 + g_ij_1;

        const double Egr_ij = E_ij * dPhi_drj_over_Phi;
        const double Ugr_ij = E_ij * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi);
        const double Wgr_ij = E_ij * (r_ij2 * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi) - dPhi_dgij_over_Phi);

        Egr_i.x += Egr_ij * r_ji.x;
        Egr_i.y += Egr_ij * r_ji.y;
        Egr_i.z += Egr_ij * r_ji.z;
        Ugr_i.x += Ugr_ij * r_ji.x;
        Ugr_i.y += Ugr_ij * r_ji.y;
        Ugr_i.z += Ugr_ij * r_ji.z;
        Wgr_i.x += Wgr_ij * r_ji.x;
        Wgr_i.y += Wgr_ij * r_ji.y;
        Wgr_i.z += Wgr_ij * r_ji.z;
    }

    if (i_in_B) {
        Egr_i.x *= -1;
        Egr_i.y *= -1;
        Egr_i.z *= -1;
        Ugr_i.x *= -1;
        Ugr_i.y *= -1;
        Ugr_i.z *= -1;
        Wgr_i.x *= -1;
        Wgr_i.y *= -1;
        Wgr_i.z *= -1;
    }

    Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = -2 * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = -2 * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = -2 * Egr_i.z;
    Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] =  2 * Ugr_i.x;
    Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] =  2 * Ugr_i.y;
    Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] =  2 * Ugr_i.z;
    Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] =  2 * Wgr_i.x;
    Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] =  2 * Wgr_i.y;
    Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] =  2 * Wgr_i.z;
}

template <int n_derivative_per_thread>
__global__
static void vv10_hess_eval_EUW_with_weight1_kernel(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                   const int ngrids, const int nderivative)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_derivative_per_thread;
    if (i >= ngrids || i_derivative_start >= nderivative)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double Ew_i[n_derivative_per_thread] {0};
    double Uw_i[n_derivative_per_thread] {0};
    double Ww_i[n_derivative_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

        const double E_ij = rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum_1 + g_ij_1);
        const double W_ij = U_ij * r_ij2;

        #pragma unroll
        for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
            if (i_derivative + i_derivative_start >= nderivative) continue;
            const double weight_j = grid_weight1[(i_derivative + i_derivative_start) * ngrids + j];
            Ew_i[i_derivative] += weight_j * E_ij;
            Uw_i[i_derivative] += weight_j * U_ij;
            Ww_i[i_derivative] += weight_j * W_ij;
        }
    }

    #pragma unroll
    for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
        if (i_derivative + i_derivative_start >= nderivative) continue;
        Ew[(i_derivative + i_derivative_start) * ngrids + i] =  Ew_i[i_derivative];
        Uw[(i_derivative + i_derivative_start) * ngrids + i] = -Uw_i[i_derivative];
        Ww[(i_derivative + i_derivative_start) * ngrids + i] = -Ww_i[i_derivative];
    }
}

extern "C" {
__host__
int VXC_vv10nlc(cudaStream_t stream, double *Fvec, double *Uvec, double *Wvec,
                 const double *vvcoords, const double *coords,
                 const double *W0p, const double *W0, const double *K,
                 const double *Kp, const double *RpW,
                 int vvngrids, int ngrids)
{
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids/NG_PER_THREADS+1+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_kernel<<<blocks, threads, 0, stream>>>(Fvec, Uvec, Wvec,
                 vvcoords, coords,
                 W0p, W0, K, Kp, RpW, vvngrids, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_grad(cudaStream_t stream, double *Fvec,
                    const double *vvcoords, const double *coords,
                    const double *W0p, const double *W0, const double *K,
                    const double *Kp, const double *RpW,
                    int vvngrids, int ngrids)
{
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_grad_kernel<<<blocks, threads, 0, stream>>>(Fvec, vvcoords, coords,
                      W0p, W0, K, Kp, RpW, vvngrids, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_UWABCE(const cudaStream_t stream,
                                 double* U, double* W, double* A, double* B, double* C, double* E,
                                 const double* grid_coord, const double* grid_weight,
                                 const double* rho, const double* omega, const double* kappa,
                                 const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_UWABCE_kernel<<<blocks, threads, 0, stream>>>(U, W, A, B, C, E,
                                                                 grid_coord, grid_weight, rho, omega, kappa, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_UWABC: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_omega_derivative(const cudaStream_t stream,
                                           double* domega_drho, double* domega_dgamma,
                                           double* d2omega_drho2, double* d2omega_dgamma2, double* d2omega_drho_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(domega_drho, domega_dgamma,
                                                                           d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_f_t(const cudaStream_t stream,
                              double* f_rho_t, double* f_gamma_t,
                              const double* grid_coord, const double* grid_weight,
                              const double* rho, const double* omega, const double* kappa,
                              const double* U, const double* W, const double* A, const double* B, const double* C,
                              const double* domega_drho, const double* domega_dgamma, const double* dkappa_drho,
                              const double* d2omega_drho2, const double* d2omega_dgamma2, const double* d2omega_drho_dgamma, const double* d2kappa_drho2,
                              const double* rho_t, const double* gamma_t,
                              const int ngrids, const int ntrial)
{
    constexpr int n_trial_per_thread = 6; // Notice: ntrial is likely a multiple of 3
    const dim3 threads(NG_PER_BLOCK, 1);
    const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                      (ntrial + n_trial_per_thread - 1) / n_trial_per_thread);
    vv10_hess_eval_f_t_kernel<n_trial_per_thread> <<<blocks, threads, 0, stream>>> (
        f_rho_t, f_gamma_t,
        grid_coord, grid_weight, rho, omega, kappa,
        U, W, A, B, C,
        domega_drho, domega_dgamma, dkappa_drho,
        d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2,
        rho_t, gamma_t, ngrids, ntrial
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_f_t: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_grid_response(const cudaStream_t stream,
                                            double* Egr, double* Ugr, double* Wgr,
                                            const double* grid_coord, const double* grid_weight,
                                            const double* rho, const double* omega, const double* kappa,
                                            const int* grid_associated_atom,
                                            const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 32;
    constexpr int n_atoms_per_block = 4;
    const dim3 threads(n_grids_per_block, n_atoms_per_block);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block,
                      (  natm + n_atoms_per_block - 1) / n_atoms_per_block);
    vv10_hess_eval_EUW_grid_response_kernel<<<blocks, threads, 0, stream>>>(Egr, Ugr, Wgr,
                                                                            grid_coord, grid_weight, rho, omega, kappa,
                                                                            grid_associated_atom, ngrids, natm);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_with_weight1(const cudaStream_t stream,
                                           double* Ew, double* Uw, double* Ww,
                                           const double* grid_coord, const double* grid_weight1,
                                           const double* rho, const double* omega, const double* kappa,
                                           const int ngrids, const int nderivative)
{
    constexpr int n_derivative_per_thread = 6; // Notice: ntrial is always a multiple of 3
    const dim3 threads(NG_PER_BLOCK, 1);
    const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                      (nderivative + n_derivative_per_thread - 1) / n_derivative_per_thread);
    vv10_hess_eval_EUW_with_weight1_kernel<n_derivative_per_thread> <<<blocks, threads, 0, stream>>> (
        Ew, Uw, Ww,
        grid_coord, grid_weight1, rho, omega, kappa,
        ngrids, nderivative
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_with_weight1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
