/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
#include "int3c2e.cuh"

#define M_PI_F 3.14159265f
#define REMOTE_THRESHOLD 50

__global__ static
void overlap_img_counts_kernel(int *img_idx, int *img_counts, PBCInt3c2eEnvVars envs,
                               float *exps, float *log_coeff)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int nbas2 = nbas * nbas;
    if (bas_ij >= nbas2) {
        return;
    }
    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    if (ish < jsh) {
        return;
    }
    int cell0_ish = ish % envs.bvk_ncells;
    int cell0_jsh = jsh % envs.bvk_ncells;
    int li = bas[ANG_OF + cell0_ish*BAS_SLOTS];
    int lj = bas[ANG_OF + cell0_jsh*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float log_cicj = log_ci + log_cj;
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij);
    float log_cutoff = envs.log_cutoff - log_fac;

    img_idx += bas_ij;
    int counts = 0;
    for (int img = 0; img < nimgs; ++img) {
        float xjL = xj + img_coords[img*3+0];
        float yjL = yj + img_coords[img*3+1];
        float zjL = zj + img_coords[img*3+2];
        float xjxi = xjL - xi;
        float yjyi = yjL - yi;
        float zjzi = zjL - zi;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float theta_ij_rr = theta_ij * rr_ij;
        if (theta_ij_rr > REMOTE_THRESHOLD) {
            continue;
        }

        float dr = sqrtf(rr_ij);
        float dri = fj * dr;
        float drj = fi * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float estimator = dri_fac + drj_fac + theta_ij_rr;
        if (estimator > log_cutoff) {
            img_idx[counts*nbas2] = img;
            counts++;
        }
    }
    img_counts[bas_ij] = counts;
}

__global__ static
void sr_int3c2e_img_sparse_kernel(int *img_idx, int *img_counts, int *bas_mapping, int npairs,
                                  int *ovlp_img_idx, int *ovlp_img_counts,
                                  PBCInt3c2eEnvVars envs,
                                  float *exps, float *log_coeff, float *atom_aux_exps,
                                  int ish0, int jsh0, int nish, int njsh)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) {
        return;
    }
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int bas_ij = bas_mapping[pair_id];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int nimgs = envs.nimgs;
    int cell0_ish = ish % envs.bvk_ncells;
    int cell0_jsh = jsh % envs.bvk_ncells;

    int cell0_natm = envs.cell0_natm;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    extern __shared__ float xyz_cache[];
    int li = bas[ANG_OF + cell0_ish*BAS_SLOTS];
    int lj = bas[ANG_OF + cell0_jsh*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float aij = ai + aj;
    float u = .5f / aij;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float log_cicj = log_ci + log_cj;
    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
    //           ~ between [0, 2]
    float fac_guess = .5f - logf(omega2)/4;
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij) + fac_guess;
    float log_cutoff = envs.log_cutoff - log_fac;

    float theta = (omega2 * aij) / (omega2 + aij);
    for (int k = thread_id; k < cell0_natm; k += threads) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        xyz_cache[k*4+0] = rk[0];
        xyz_cache[k*4+1] = rk[1];
        xyz_cache[k*4+2] = rk[2];
        float ak = atom_aux_exps[k];
        xyz_cache[k*4+3] = theta * ak / (theta + ak);
    }
    __syncthreads();

    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];

    int ovlp_img_count = ovlp_img_counts[bas_ij];
    img_idx += pair_id;
    ovlp_img_idx += bas_ij;
    int nbas2 = nbas * nbas;
    int counts = 0;
    for (int jL = 0; jL < ovlp_img_count; ++jL) {
        int ptr = ovlp_img_idx[jL*nbas2];
        float xjL = xj + img_coords[ptr*3+0];
        float yjL = yj + img_coords[ptr*3+1];
        float zjL = zj + img_coords[ptr*3+2];
        float xjxi = xjL - xi;
        float yjyi = yjL - yi;
        float zjzi = zjL - zi;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float theta_ij_rr = theta_ij * rr_ij;

        float xij_0 = xjxi * fj + xi;
        float yij_0 = yjyi * fj + yi;
        float zij_0 = zjzi * fj + zi;
        for (int iL = 0; iL < nimgs; ++iL) {
            float xij = xij_0 + img_coords[iL*3+0];
            float yij = yij_0 + img_coords[iL*3+1];
            float zij = zij_0 + img_coords[iL*3+2];
            float rr_min = 1e3f;
            float theta_rr_min = 1e6f;
            for (int k = 0; k < cell0_natm; ++k) {
                float dx = xij - xyz_cache[k*4+0];
                float dy = yij - xyz_cache[k*4+1];
                float dz = zij - xyz_cache[k*4+2];
                float rr = dx * dx + dy * dy + dz * dz;
                float theta_k = xyz_cache[k*4+3];
                float theta_rr = theta_k * rr;
                if (theta_rr < theta_rr_min) {
                    theta_rr_min = theta_rr;
                    rr_min = rr;
                }
            }
            theta_rr_min += theta_ij_rr;
            if (theta_rr_min > REMOTE_THRESHOLD) {
                continue;
            }

            float rt_aij = omega2 * sqrtf(rr_min) / aij + 1e-9f;
            float dr = sqrtf(rr_ij);
            float dri = fj * dr + rt_aij;
            float drj = fi * dr + rt_aij;
            float dri_fac = .5f*li * logf(dri*dri + li*u);
            float drj_fac = .5f*lj * logf(drj*drj + lj*u);
            float estimator = dri_fac + drj_fac - theta_rr_min;
            if (estimator > log_cutoff) {
                img_idx[counts*npairs] = iL*nimgs+jL;
                counts++;
            }
        }
    }
    img_counts[pair_id] = counts;
}

// Concatenate dis-continuous 
__global__ static
void conc_img_idx_kernel(int *output, int *offsets, int *idx_sparse,
                         int *where, int rows)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= rows) {
        return;
    }
    int p = where[row_id];
    int count = offsets[row_id+1] - offsets[row_id];
    output += offsets[row_id];
    idx_sparse += p;
    for (int k = 0; k < count; ++k) {
        output[k] = idx_sparse[k*rows];
    }
}

extern "C" {
int overlap_img_counts(int *img_idx, int *img_counts, PBCInt3c2eEnvVars *envs,
                       float *exps, float *log_coeff)
{
    constexpr int threads = 512;
    int nbas = envs->cell0_nbas * envs->bvk_ncells;
    int blocks = (nbas*nbas + threads-1)/threads;
    overlap_img_counts_kernel<<<blocks, threads>>>(
        img_idx, img_counts, *envs, exps, log_coeff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in overlap_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int sr_int3c2e_img_idx_sparse(int *img_idx, int *img_counts, int *bas_mapping, int npairs,
                           int *ovlp_img_idx, int *ovlp_img_counts,
                           PBCInt3c2eEnvVars *envs, int *shls_slice,
                           float *exps, float *log_coeff, float *atom_aux_exps)
{
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    int jsh0 = shls_slice[2];
    int jsh1 = shls_slice[3];
    int nish = ish1 - ish0;
    int njsh = jsh1 - jsh0;
    constexpr int threads = 256;
    int blocks = (npairs + threads-1) / threads;
    int cell0_natm = envs->cell0_natm;
    int buflen = cell0_natm * 4 * sizeof(float);
    sr_int3c2e_img_sparse_kernel<<<blocks, threads, buflen>>>(
        img_idx, img_counts, bas_mapping, npairs, ovlp_img_idx, ovlp_img_counts,
        *envs, exps, log_coeff, atom_aux_exps, ish0, jsh0, nish, njsh);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in sr_int3c2e_img_idx_sparse: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int conc_img_idx(int *output, int *offsets, int *idx_sparse,
                 int *where, int rows)
{
    constexpr int threads = 1024;
    int blocks = (rows + threads-1) / threads;
    conc_img_idx_kernel<<<blocks, threads>>>(
        output, offsets, idx_sparse, where, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in conc_img_idx: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
