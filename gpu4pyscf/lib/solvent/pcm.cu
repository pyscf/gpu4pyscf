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

#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS        32
#define SQRT2_PI       0.7978845608028654
#define SQRT_PI        1.7724538509055159

// D and S matrix in J. Chem. Phys. 133, 244111 (2010)
__global__
static void _pcm_d_s(double *matrix_d, double *matrix_s,
                    const double *coords, const double *norm_vec, const double *r_vdw,
                    const double *charge_exp, const double *switch_fun,
                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n){
        return;
    }

    // calculate xi
    double ei = charge_exp[i];
    double ej = charge_exp[j];
    double xi_ij = ei * ej / sqrt(ei*ei + ej*ej);

    // calculate r
    double xi = coords[3*i];
    double yi = coords[3*i+1];
    double zi = coords[3*i+2];
    double xj = coords[3*j];
    double yj = coords[3*j+1];
    double zj = coords[3*j+2];
    double dx = xi - xj;
    double dy = yi - yj;
    double dz = zi - zj;
    double rij = norm3d(dx, dy, dz);

    double xi_r_ij = xi_ij * rij;
    if (i == j) rij = 1.0;
    double s = erf(xi_r_ij) / rij;
    if (i == j) s = charge_exp[i] * SQRT2_PI / switch_fun[i];
    matrix_s[i*n+j] = s;

    if (matrix_d != NULL){
        double nxj = norm_vec[3*j];
        double nyj = norm_vec[3*j+1];
        double nzj = norm_vec[3*j+2];

        double nrij = 0.0;
        nrij += (xi - xj) * nxj;
        nrij += (yi - yj) * nyj;
        nrij += (zi - zj) * nzj;

        double rij2 = rij*rij;
        double rij3 = rij2*rij;
        double xi_r2_ij = xi_r_ij * xi_r_ij;
        double d = s * nrij / rij2 - 2.0*xi_r_ij/SQRT_PI*exp(-xi_r2_ij)*nrij/rij3;
        if (i == j) d = -charge_exp[i] * SQRT2_PI / (2.0*r_vdw[i]);
        matrix_d[i*n+j] = d;
    }
}

__global__
static void _pcm_dD_dS(double *matrix_dd, double *matrix_ds,
                       const double *coords, const double *norm_vec,
                       const double *charge_exp,
                       int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n){
        return;
    }

    // calculate xi
    double ei = charge_exp[i];
    double ej = charge_exp[j];
    double xi_ij = ei * ej / sqrt(ei*ei + ej*ej);

    // calculate r
    double dx = coords[3*i]   - coords[3*j];
    double dy = coords[3*i+1] - coords[3*j+1];
    double dz = coords[3*i+2] - coords[3*j+2];
    double rij = norm3d(dx, dy, dz);

    double xi_r_ij = xi_ij * rij;
    double xi_r2_ij = xi_r_ij * xi_r_ij;
    if (i == j) rij = 1.0;
    double rij2 = rij*rij;

    double dS_dr = -(erf(xi_r_ij) -  2.0*xi_r_ij/ SQRT_PI * exp(-xi_r2_ij)) / rij2;
    if (i == j) dS_dr = 0.0;
    double dx_rij = dx / rij;
    double dy_rij = dy / rij;
    double dz_rij = dz / rij;

    matrix_ds[i*n+j       ] = dS_dr * dx_rij;
    matrix_ds[i*n+j +  n*n] = dS_dr * dy_rij;
    matrix_ds[i*n+j +2*n*n] = dS_dr * dz_rij;

    if (matrix_dd != NULL){
        double nxj = norm_vec[3*j];
        double nyj = norm_vec[3*j+1];
        double nzj = norm_vec[3*j+2];
        double nj_rij = dx*nxj + dy*nyj + dz*nzj;
        double rij3 = rij2*rij;
        double dD_dri = 4.0*xi_r2_ij*xi_ij / SQRT_PI*exp(-xi_r2_ij)*nj_rij/rij3;
        if (i == j) dD_dri = 0.0;

        nj_rij = 3.0*nj_rij/rij2;
        matrix_dd[i*n+j        ] = dD_dri*dx_rij + dS_dr*(-nxj/rij + nj_rij*dx_rij);
        matrix_dd[i*n+j +   n*n] = dD_dri*dy_rij + dS_dr*(-nyj/rij + nj_rij*dy_rij);
        matrix_dd[i*n+j + 2*n*n] = dD_dri*dz_rij + dS_dr*(-nzj/rij + nj_rij*dz_rij);
    }
}

__global__
static void _pcm_left_multiply_dS(double *output, const double* right_vector,
                                  const double *coords, const double *charge_exp,
                                  int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double rix = coords[3*i  ];
    const double riy = coords[3*i+1];
    const double riz = coords[3*i+2];
    const double ei = charge_exp[i];

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int j = threadIdx.y; j < n; j += blockDim.y) {
        // calculate xi
        const double ej = charge_exp[j];
        const double xi_ij = ei * ej / sqrt(ei*ei + ej*ej);

        // calculate r
        const double dx = rix - coords[3*j  ];
        const double dy = riy - coords[3*j+1];
        const double dz = riz - coords[3*j+2];
        double rij = norm3d(dx, dy, dz);

        const double xi_r_ij = xi_ij * rij;
        const double xi_r2_ij = xi_r_ij * xi_r_ij;
        if (i == j) rij = 1.0;
        const double rij2 = rij*rij;

        double dS_dr = -(erf(xi_r_ij) -  2.0*xi_r_ij/ SQRT_PI * exp(-xi_r2_ij)) / rij2;
        if (i == j) dS_dr = 0.0;
        const double dx_rij = dx / rij;
        const double dy_rij = dy / rij;
        const double dz_rij = dz / rij;

        const double dSx = dS_dr * dx_rij;
        const double dSy = dS_dr * dy_rij;
        const double dSz = dS_dr * dz_rij;

        const double right_vector_j = right_vector[j];
        sum_x += dSx * right_vector_j;
        sum_y += dSy * right_vector_j;
        sum_z += dSz * right_vector_j;
    }

    __shared__ double sum_shared[THREADS * THREADS];

    sum_shared[threadIdx.y * THREADS + threadIdx.x] = sum_x;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            sum_shared[threadIdx.y * THREADS + threadIdx.x] += sum_shared[(threadIdx.y + stride) * THREADS + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        output[        i] = sum_shared[threadIdx.x];
    }

    sum_shared[threadIdx.y * THREADS + threadIdx.x] = sum_y;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            sum_shared[threadIdx.y * THREADS + threadIdx.x] += sum_shared[(threadIdx.y + stride) * THREADS + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        output[n     + i] = sum_shared[threadIdx.x];
    }

    sum_shared[threadIdx.y * THREADS + threadIdx.x] = sum_z;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            sum_shared[threadIdx.y * THREADS + threadIdx.x] += sum_shared[(threadIdx.y + stride) * THREADS + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        output[n * 2 + i] = sum_shared[threadIdx.x];
    }
}

__global__
static void _pcm_d2D_d2S(double *matrix_d2D, double *matrix_d2S,
                         const double *coords, const double *norm_vec,
                         const double *charge_exp,
                         int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) {
        return;
    }

    // calculate xi
    const double ei = charge_exp[i];
    const double ej = charge_exp[j];
    const double eij = ei * ej / sqrt(ei*ei + ej*ej);

    // calculate r
    const double dx = coords[3*i]   - coords[3*j];
    const double dy = coords[3*i+1] - coords[3*j+1];
    const double dz = coords[3*i+2] - coords[3*j+2];
    const double rij = norm3d(dx, dy, dz);
    const double rij_1 = (i != j) ? (1.0 / rij) : 0.0; // This guarantees that if i == j, all matrix elements = 0
    const double rij_2 = rij_1 * rij_1;
    const double rij_3 = rij_2 * rij_1;
    const double rij_4 = rij_2 * rij_2;
    const double rij_5 = rij_2 * rij_3;
    const double eij2 = eij * eij;

    const double eij_rij = eij * rij;
    const double erf_eij_rij = erf(eij_rij);
    const double exp_minus_eij2_rij2 = exp(-eij_rij * eij_rij);
    const double two_eij_over_sqrt_pi = 2.0 * eij / SQRT_PI;
    const double two_eij_over_sqrt_pi_exp_minus_eij2_rij2 = exp_minus_eij2_rij2 * two_eij_over_sqrt_pi;

    const double S_direct_product_prefactor = -two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * (3 * rij_4 + 2 * eij2 * rij_2)
                                              + 3 * rij_5 * erf_eij_rij;
    const double S_xyz_diagonal_prefactor = two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * rij_2 - rij_3 * erf_eij_rij;

    const int n2 = n * n;
    matrix_d2S[i*n + j         ] = dx * dx * S_direct_product_prefactor + S_xyz_diagonal_prefactor;
    matrix_d2S[i*n + j + n2    ] = dx * dy * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 2] = dx * dz * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 3] = dy * dx * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 4] = dy * dy * S_direct_product_prefactor + S_xyz_diagonal_prefactor;
    matrix_d2S[i*n + j + n2 * 5] = dy * dz * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 6] = dz * dx * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 7] = dz * dy * S_direct_product_prefactor;
    matrix_d2S[i*n + j + n2 * 8] = dz * dz * S_direct_product_prefactor + S_xyz_diagonal_prefactor;

    if (matrix_d2D != NULL) {
        const double nxj = norm_vec[3*j];
        const double nyj = norm_vec[3*j+1];
        const double nzj = norm_vec[3*j+2];
        const double nj_rij = dx * nxj + dy * nyj + dz * nzj;

        const double eij4 = eij2 * eij2;
        const double rij_6 = rij_4 * rij_2;
        const double rij_7 = rij_4 * rij_3;

        const double D_direct_product_prefactor = (-two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * (15 * rij_6 + 10 * eij2 * rij_4 + 4 * eij4 * rij_2)
                                                   + 15 * rij_7 * erf_eij_rij) * nj_rij;
        matrix_d2D[i*n + j         ] = D_direct_product_prefactor * dx * dx - S_direct_product_prefactor * (dx * nxj + dx * nxj + nj_rij);
        matrix_d2D[i*n + j + n2    ] = D_direct_product_prefactor * dx * dy - S_direct_product_prefactor * (dy * nxj + dx * nyj);
        matrix_d2D[i*n + j + n2 * 2] = D_direct_product_prefactor * dx * dz - S_direct_product_prefactor * (dz * nxj + dx * nzj);
        matrix_d2D[i*n + j + n2 * 3] = D_direct_product_prefactor * dy * dx - S_direct_product_prefactor * (dx * nyj + dy * nxj);
        matrix_d2D[i*n + j + n2 * 4] = D_direct_product_prefactor * dy * dy - S_direct_product_prefactor * (dy * nyj + dy * nyj + nj_rij);
        matrix_d2D[i*n + j + n2 * 5] = D_direct_product_prefactor * dy * dz - S_direct_product_prefactor * (dz * nyj + dy * nzj);
        matrix_d2D[i*n + j + n2 * 6] = D_direct_product_prefactor * dz * dx - S_direct_product_prefactor * (dx * nzj + dz * nxj);
        matrix_d2D[i*n + j + n2 * 7] = D_direct_product_prefactor * dz * dy - S_direct_product_prefactor * (dy * nzj + dz * nyj);
        matrix_d2D[i*n + j + n2 * 8] = D_direct_product_prefactor * dz * dz - S_direct_product_prefactor * (dz * nzj + dz * nzj + nj_rij);
    }
}

__global__
static void _pcm_d2F_to_d2Sii(const double* F, const double* dF, const double* d2F, const double* charge_exp,
                              double* d2Sii, const int n_atom, const int n_grid)
{
    const int i_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ij_atom = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_grid >= n_grid || ij_atom >= n_atom * n_atom) {
        return;
    }

    const int i_atom = ij_atom / n_atom;
    const int j_atom = ij_atom % n_atom;

    const double zeta = charge_exp[i_grid];
    const double F_value = F[i_grid];
    const double F_1 = 1.0 / F_value;
    const double F_2 = F_1 * F_1;
    const double combined_factor = SQRT2_PI * zeta * F_2;

    const double dFix = dF[(i_atom * 3    ) * n_grid + i_grid];
    const double dFiy = dF[(i_atom * 3 + 1) * n_grid + i_grid];
    const double dFiz = dF[(i_atom * 3 + 2) * n_grid + i_grid];
    const double dFjx = dF[(j_atom * 3    ) * n_grid + i_grid];
    const double dFjy = dF[(j_atom * 3 + 1) * n_grid + i_grid];
    const double dFjz = dF[(j_atom * 3 + 2) * n_grid + i_grid];

    const double d2Fixjx = d2F[((i_atom * n_atom + j_atom) * 9 + 0 * 3    ) * n_grid + i_grid];
    const double d2Fixjy = d2F[((i_atom * n_atom + j_atom) * 9 + 0 * 3 + 1) * n_grid + i_grid];
    const double d2Fixjz = d2F[((i_atom * n_atom + j_atom) * 9 + 0 * 3 + 2) * n_grid + i_grid];
    const double d2Fiyjx = d2F[((i_atom * n_atom + j_atom) * 9 + 1 * 3    ) * n_grid + i_grid];
    const double d2Fiyjy = d2F[((i_atom * n_atom + j_atom) * 9 + 1 * 3 + 1) * n_grid + i_grid];
    const double d2Fiyjz = d2F[((i_atom * n_atom + j_atom) * 9 + 1 * 3 + 2) * n_grid + i_grid];
    const double d2Fizjx = d2F[((i_atom * n_atom + j_atom) * 9 + 2 * 3    ) * n_grid + i_grid];
    const double d2Fizjy = d2F[((i_atom * n_atom + j_atom) * 9 + 2 * 3 + 1) * n_grid + i_grid];
    const double d2Fizjz = d2F[((i_atom * n_atom + j_atom) * 9 + 2 * 3 + 2) * n_grid + i_grid];

    d2Sii[((i_atom * n_atom + j_atom) * 9 + 0 * 3    ) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFix * dFjx - d2Fixjx);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 0 * 3 + 1) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFix * dFjy - d2Fixjy);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 0 * 3 + 2) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFix * dFjz - d2Fixjz);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 1 * 3    ) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiy * dFjx - d2Fiyjx);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 1 * 3 + 1) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiy * dFjy - d2Fiyjy);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 1 * 3 + 2) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiy * dFjz - d2Fiyjz);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 2 * 3    ) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiz * dFjx - d2Fizjx);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 2 * 3 + 1) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiz * dFjy - d2Fizjy);
    d2Sii[((i_atom * n_atom + j_atom) * 9 + 2 * 3 + 2) * n_grid + i_grid] = combined_factor * (2 * F_1 * dFiz * dFjz - d2Fizjz);
}

extern "C" {
int pcm_d_s(cudaStream_t stream, double *matrix_d, double *matrix_s,
                    const double *coords, const double *norm_vec, const double *r_vdw,
                    const double *charge_exp, const double *switch_fun,
                    int n)
{
    int ntilex = (n + THREADS - 1) / THREADS;
    int ntiley = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntilex, ntiley);
    _pcm_d_s<<<blocks, threads, 0, stream>>>(matrix_d, matrix_s, coords, norm_vec, r_vdw, charge_exp, switch_fun, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int pcm_dd_ds(cudaStream_t stream, double *matrix_dD, double *matrix_dS,
              const double *coords, const double *norm_vec,
              const double *charge_exp,
              int n)
{
    int ntilex = (n + THREADS - 1) / THREADS;
    int ntiley = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntilex, ntiley);
    _pcm_dD_dS<<<blocks, threads, 0, stream>>>(matrix_dD, matrix_dS, coords, norm_vec, charge_exp, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int pcm_left_multiply_ds(const cudaStream_t stream, double *output, const double *right_vector,
                         const double *coords, const double *charge_exp,
                         int n)
{
    int ntilex = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntilex, 1);
    _pcm_left_multiply_dS<<<blocks, threads, 0, stream>>>(output, right_vector, coords, charge_exp, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int pcm_d2d_d2s(cudaStream_t stream, double *matrix_d2D, double *matrix_d2S,
                const double *coords, const double *norm_vec,
                const double *charge_exp,
                int n)
{
    const int ntilex = (n + THREADS - 1) / THREADS;
    const int ntiley = (n + THREADS - 1) / THREADS;
    const dim3 threads(THREADS, THREADS);
    const dim3 blocks(ntilex, ntiley);
    _pcm_d2D_d2S<<<blocks, threads, 0, stream>>>(matrix_d2D, matrix_d2S, coords, norm_vec, charge_exp, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int pcm_d2f_to_d2sii(cudaStream_t stream, const double* F, const double* dF, const double* d2F, const double* charge_exp,
                     double* d2Sii, const int n_atom, const int n_grid)
{
    const int ntilex = (n_grid + THREADS - 1) / THREADS;
    const int ntiley = (n_atom * n_atom + THREADS - 1) / THREADS;
    const dim3 threads(THREADS, THREADS);
    const dim3 blocks(ntilex, ntiley);
    _pcm_d2F_to_d2Sii<<<blocks, threads, 0, stream>>>(F, dF, d2F, charge_exp, d2Sii, n_atom, n_grid);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
