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
                    const double *coords, const double *norm_vec, const double *r_vdw,
                    const double *charge_exp, const double *switch_fun,
                    int n)
{
    int ntilex = (n + THREADS - 1) / THREADS;
    int ntiley = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntilex, ntiley);
    _pcm_dD_dS<<<blocks, threads, 0, stream>>>(matrix_dD, matrix_dS, coords, norm_vec, r_vdw, charge_exp, switch_fun, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
