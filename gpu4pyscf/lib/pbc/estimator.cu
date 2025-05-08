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

#define REMOTE_THRESHOLD 50

__global__ static
void overlap_img_counts_kernel(int *img_counts, int *p2c_mapping,
                               int ish0, int jsh0, int nish, int njsh,
                               PBCInt3c2eEnvVars envs, float *exps,
                               float *log_coeff, float log_cutoff)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int bvk_nish = envs.bvk_ncells * nish;
    int bvk_njsh = envs.bvk_ncells * njsh;
    if (bas_ij >= bvk_nish*bvk_njsh) {
        return;
    }
    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int ish = bas_ij / bvk_njsh;
    int jsh = bas_ij % bvk_njsh;
    int cell0_ish = ish % nish + ish0;;
    int cell0_jsh = jsh % njsh + jsh0;;
    if (cell0_ish < cell0_jsh &&
        // filtering based on the contracted orbital-pairs than the primitive shells
        p2c_mapping[cell0_ish] != p2c_mapping[cell0_jsh]) {
        return;
    }
    ish = ish / nish * envs.cell0_nbas + cell0_ish;
    jsh = jsh / njsh * envs.cell0_nbas + cell0_jsh;
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
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff = log_cutoff - log_fac;

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
        float estimator = dri_fac + drj_fac - theta_ij_rr;
        if (estimator > log_cutoff) {
            counts++;
        }
    }
    img_counts[bas_ij] = counts;
}

__global__ static
void overlap_img_idx_kernel(int *img_idx, int *img_offsets, int *bas_ij_mapping,
                            int npairs, int ish0, int jsh0, int nish, int njsh,
                            PBCInt3c2eEnvVars envs, float *exps, float *log_coeff,
                            float log_cutoff)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) {
        return;
    }
    int bas_ij = bas_ij_mapping[pair_id];
    int bvk_njsh = envs.bvk_ncells * njsh;
    int ish = bas_ij / bvk_njsh;
    int jsh = bas_ij % bvk_njsh;
    int cell0_ish = ish % nish + ish0;;
    int cell0_jsh = jsh % njsh + jsh0;;
    ish = ish / nish * envs.cell0_nbas + cell0_ish;
    jsh = jsh / njsh * envs.cell0_nbas + cell0_jsh;

    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
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
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff = log_cutoff - log_fac;

    int counts = 0;
    img_idx += img_offsets[pair_id];
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
        float estimator = dri_fac + drj_fac - theta_ij_rr;
        if (estimator > log_cutoff) {
            img_idx[counts] = img;
            counts++;
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(64) static
#else
__global__ static
#endif
void sr_int3c2e_img_kernel(int *img_idx, int *counts_or_offsets, int *bas_ij_mapping,
                           int *pair_sorting, int *ovlp_img_idx, int *ovlp_img_offsets,
                           int npairs, int ish0, int jsh0, int nish, int njsh,
                           PBCInt3c2eEnvVars envs, float *exps, float *log_coeff,
                           float *atom_aux_exps, float log_cutoff)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int cell0_natm = envs.cell0_natm;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ float xyz_cache[];
    for (int k = thread_id; k < cell0_natm; k += threads) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        xyz_cache[k*3+0] = rk[0];
        xyz_cache[k*3+1] = rk[1];
        xyz_cache[k*3+2] = rk[2];
    }
    __syncthreads();
    if (pair_id >= npairs) {
        return;
    }
    int bas_ij = bas_ij_mapping[pair_id];
    int bvk_njsh = envs.bvk_ncells * njsh;
    int ish = bas_ij / bvk_njsh;
    int jsh = bas_ij % bvk_njsh;
    int cell0_ish = ish % nish + ish0;;
    int cell0_jsh = jsh % njsh + jsh0;;
    ish = ish / nish * envs.cell0_nbas + cell0_ish;
    jsh = jsh / njsh * envs.cell0_nbas + cell0_jsh;

    int nimgs = envs.nimgs;
    double *img_coords = envs.img_coords;
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
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij) + fac_guess;
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff = log_cutoff - log_fac;
    float theta = (omega2 * aij) / (omega2 + aij);
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];

    if (img_idx != NULL) {
        int *img_offsets = counts_or_offsets;
        img_idx += img_offsets[pair_id];
    }
    int ovlp_pair_id = pair_sorting[pair_id];
    int jL0 = ovlp_img_offsets[ovlp_pair_id];
    int jL1 = ovlp_img_offsets[ovlp_pair_id+1];
    int counts = 0;
    for (int jLp = jL0; jLp < jL1; ++jLp) {
        int jL = ovlp_img_idx[jLp];
        float xi = ri[0];
        float yi = ri[1];
        float zi = ri[2];
        float xj = rj[0];
        float yj = rj[1];
        float zj = rj[2];
        float xjL = xj + img_coords[jL*3+0];
        float yjL = yj + img_coords[jL*3+1];
        float zjL = zj + img_coords[jL*3+2];
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
                float dx = xij - xyz_cache[k*3+0];
                float dy = yij - xyz_cache[k*3+1];
                float dz = zij - xyz_cache[k*3+2];
                float rr = dx * dx + dy * dy + dz * dz;
                float ak = atom_aux_exps[k];
                float theta_k = theta * ak / (theta + ak);
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

            float rt_aij = omega2 * sqrtf(rr_min) / aij;
            float dr = sqrtf(rr_ij);
            float dri = fj * dr + rt_aij;
            float drj = fi * dr + rt_aij;
            float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
            float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
            float estimator = dri_fac + drj_fac - theta_rr_min;
            if (estimator > log_cutoff) {
                if (img_idx != NULL) {
                    img_idx[counts] = iL*nimgs+jL;
                }
                counts++;
            }
        }
    }
    if (img_idx == NULL) {
        int *img_counts = counts_or_offsets;
        img_counts[pair_id] = counts;
    }
}


// Concatenate dis-continuous 
__global__ static
void conc_img_idx_kernel(int *output, int *offsets, int *idx_sparse,
                         int *where, int rows, int strides)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= rows) {
        return;
    }
    int count = offsets[row_id+1] - offsets[row_id];
    output += offsets[row_id];
    idx_sparse += where[row_id];
    for (int k = 0; k < count; ++k) {
        output[k] = idx_sparse[k*strides];
    }
}

extern "C" {
int bvk_overlap_img_counts(int *img_counts, int *p2c_mapping, int *shls_slice,
                           PBCInt3c2eEnvVars *envs, float *exps, float *log_coeff,
                           float log_cutoff)
{
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    int jsh0 = shls_slice[2];
    int jsh1 = shls_slice[3];
    int nish = ish1 - ish0;
    int njsh = jsh1 - jsh0;
    constexpr int threads = 512;
    int ncells = envs->bvk_ncells;
    int blocks = (ncells*nish*ncells*njsh + threads-1)/threads;
    overlap_img_counts_kernel<<<blocks, threads>>>(
        img_counts, p2c_mapping, ish0, jsh0, nish, njsh,
        *envs, exps, log_coeff, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_overlap_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bvk_overlap_img_idx(int *img_idx, int *img_offsets, int *bas_ij_mapping,
                        int npairs, int *shls_slice, PBCInt3c2eEnvVars *envs,
                        float *exps, float *log_coeff, float log_cutoff)
{
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    int jsh0 = shls_slice[2];
    int jsh1 = shls_slice[3];
    int nish = ish1 - ish0;
    int njsh = jsh1 - jsh0;
    constexpr int threads = 512;
    int blocks = (npairs + threads-1)/threads;
    overlap_img_idx_kernel<<<blocks, threads>>>(
        img_idx, img_offsets, bas_ij_mapping, npairs, ish0, jsh0, nish, njsh,
        *envs, exps, log_coeff, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_overlap_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
int sr_int3c2e_img_idx(int *img_idx, int *counts_or_offsets, int *bas_ij_mapping,
                       int *pair_sorting, int *ovlp_img_idx, int *ovlp_img_offsets,
                       int npairs, int *shls_slice, PBCInt3c2eEnvVars *envs,
                       float *exps, float *log_coeff, float *atom_aux_exps,
                       float log_cutoff)
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
    int buflen = cell0_natm * 3 * sizeof(float);
    sr_int3c2e_img_kernel<<<blocks, threads, buflen>>>(
        img_idx, counts_or_offsets, bas_ij_mapping, pair_sorting, ovlp_img_idx, ovlp_img_offsets,
        npairs, ish0, jsh0, nish, njsh, *envs, exps, log_coeff, atom_aux_exps,
        log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in sr_int3c2e_img_idx: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int conc_img_idx(int *output, int *offsets, int *idx_sparse,
                 int *where, int rows, int strides)
{
    if (rows == 0) {
        return 0;
    }
    constexpr int threads = 512;
    int blocks = (rows + threads-1) / threads;
    conc_img_idx_kernel<<<blocks, threads>>>(
        output, offsets, idx_sparse, where, rows, strides);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in conc_img_idx: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
