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

__global__
static void vv10_fock_eval_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                      const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                      const double* __restrict__ omega, const double* __restrict__ kappa,
                                      const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ngrids;

    double omega_i = NAN;
    double kappa_i = NAN;
    double3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

__global__
static void vv10_fock_eval_omega_derivative_kernel(double* __restrict__ omega, double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega1 = sqrt(omega2);
    const double omega_1 = 1 / omega1;

    omega[i] = omega1;
    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;
}

__global__
static void vv10_grad_eval_E_grid_response_offdiagonal_kernel(double* __restrict__ Egr,
                                                              const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                              const double* __restrict__ omega, const double* __restrict__ kappa,
                                                              const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                              const int natoms, const int i_grid_begin, const int ngrids)
{
    const int i_unoffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i_unoffset >= ngrids || B_atom >= natoms)
        return;
    const int i = i_unoffset + i_grid_begin;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }
    if (i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ij = { r_i.x - r_j.x, r_i.y - r_j.y, r_i.z - r_j.z };
        const double r_ij2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
        const double Egr_ij = E_ij * dPhi_drj_over_Phi;

        Egr_i.x += Egr_ij * r_ij.x;
        Egr_i.y += Egr_ij * r_ij.y;
        Egr_i.z += Egr_ij * r_ij.z;
    }

    Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 1.5 * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 1.5 * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 1.5 * Egr_i.z;
}

__global__
static void vv10_hess_eval_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                         const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                         const double* __restrict__ omega, const double* __restrict__ kappa,
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
        const double rho_weight_j = rho_weight[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
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
                                                    const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                    const double* __restrict__ omega, const double* __restrict__ kappa,
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
        const double rho_weight_j = rho_weight[j];

        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
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
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

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

__global__
static void vv10_hess_eval_D_B_in_double_grid_response_kernel(double* __restrict__ D_B,
                                                              const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                              const double* __restrict__ omega, const double* __restrict__ kappa,
                                                              const int* __restrict__ grid_associated_atom,
                                                              const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = 0;
        return;
    }
    const bool i_in_B = (i_associated_atom == B_atom);

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double D_B_i[9] { 0,0,0, 0,0,0, 0,0,0, };

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
        const double rho_weight_j = rho_weight[j];

        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_ij_1 = 1 / g_ij;
        const double g_ji_1 = 1 / g_ji;
        const double g_sum_1 = 1 / (g_ij + g_ji);
        const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;
        const double omega_i_over_g_ij = omega_i * g_ij_1;
        const double omega_j_over_g_ji = omega_j * g_ji_1;
        const double omega_sum_over_g_sum = (omega_i + omega_j) * g_sum_1;
        const double omega_over_g_three_term_sum = omega_i_over_g_ij + omega_j_over_g_ji + omega_sum_over_g_sum;

        const double E_ij = rho_weight_j * Phi_ij;
        const double outer_product_prefactor = 2 * E_ij * (
            omega_over_g_three_term_sum * omega_over_g_three_term_sum
            + omega_i_over_g_ij * omega_i_over_g_ij
            + omega_j_over_g_ji * omega_j_over_g_ji
            + omega_sum_over_g_sum * omega_sum_over_g_sum
        );
        const double identity_prefactor = -E_ij * omega_over_g_three_term_sum;

        D_B_i[0] += outer_product_prefactor * r_ji.x * r_ji.x + identity_prefactor;
        D_B_i[1] += outer_product_prefactor * r_ji.x * r_ji.y;
        D_B_i[2] += outer_product_prefactor * r_ji.x * r_ji.z;
        D_B_i[3] += outer_product_prefactor * r_ji.y * r_ji.x;
        D_B_i[4] += outer_product_prefactor * r_ji.y * r_ji.y + identity_prefactor;
        D_B_i[5] += outer_product_prefactor * r_ji.y * r_ji.z;
        D_B_i[6] += outer_product_prefactor * r_ji.z * r_ji.x;
        D_B_i[7] += outer_product_prefactor * r_ji.z * r_ji.y;
        D_B_i[8] += outer_product_prefactor * r_ji.z * r_ji.z + identity_prefactor;
    }

    if (i_in_B) {
        D_B_i[0] *= -1;
        D_B_i[1] *= -1;
        D_B_i[2] *= -1;
        D_B_i[3] *= -1;
        D_B_i[4] *= -1;
        D_B_i[5] *= -1;
        D_B_i[6] *= -1;
        D_B_i[7] *= -1;
        D_B_i[8] *= -1;
    }

    D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = -2 * D_B_i[0];
    D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = -2 * D_B_i[1];
    D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = -2 * D_B_i[2];
    D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = -2 * D_B_i[3];
    D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = -2 * D_B_i[4];
    D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = -2 * D_B_i[5];
    D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = -2 * D_B_i[6];
    D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = -2 * D_B_i[7];
    D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = -2 * D_B_i[8];
}

extern "C" {
__host__
int VXC_vv10nlc_fock_eval_UWE(const cudaStream_t stream,
                              double* U, double* W, double* E,
                              const double* grid_coord, const double* rho_weight,
                              const double* omega, const double* kappa,
                              const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_fock_eval_UWE_kernel<<<blocks, threads, 0, stream>>>(U, W, E,
                                                              grid_coord, rho_weight, omega, kappa, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_UWE: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_fock_eval_omega_derivative(const cudaStream_t stream,
                                           double* omega, double* domega_drho, double* domega_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_fock_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(omega, domega_drho, domega_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(const cudaStream_t stream,
                                                      double* Egr,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                      const int natm, const int i_grid_begin, const int ngrids)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_grad_eval_E_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        Egr, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, natm, i_grid_begin, ngrids
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad eval_E_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_UWABCE(const cudaStream_t stream,
                                 double* U, double* W, double* A, double* B, double* C, double* E,
                                 const double* grid_coord, const double* rho_weight,
                                 const double* omega, const double* kappa,
                                 const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_UWABCE_kernel<<<blocks, threads, 0, stream>>>(U, W, A, B, C, E,
                                                                 grid_coord, rho_weight, omega, kappa, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_UWABCE: %s\n", cudaGetErrorString(err));
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
                                            const double* grid_coord, const double* rho_weight,
                                            const double* omega, const double* kappa,
                                            const int* grid_associated_atom,
                                            const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 32;
    constexpr int n_atoms_per_block = 4;
    const dim3 threads(n_grids_per_block, n_atoms_per_block);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block,
                      (  natm + n_atoms_per_block - 1) / n_atoms_per_block);
    vv10_hess_eval_EUW_grid_response_kernel<<<blocks, threads, 0, stream>>>(Egr, Ugr, Wgr,
                                                                            grid_coord, rho_weight, omega, kappa,
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

__host__
int VXC_vv10nlc_hess_eval_D_B_in_double_grid_response(const cudaStream_t stream,
                                                      double* D_B,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom,
                                                      const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 32;
    constexpr int n_atoms_per_block = 4;
    const dim3 threads(n_grids_per_block, n_atoms_per_block);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block,
                      (  natm + n_atoms_per_block - 1) / n_atoms_per_block);
    vv10_hess_eval_D_B_in_double_grid_response_kernel<<<blocks, threads, 0, stream>>>(
        D_B, grid_coord, rho_weight, omega, kappa, grid_associated_atom, ngrids, natm
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_E_grgr_AB: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
