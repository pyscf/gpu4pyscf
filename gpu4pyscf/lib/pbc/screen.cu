/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

#define THREADS 256

__global__ static
void estimate_aft_Ecut_kernel(float *Ecut, int64_t *bas_ij_idx, PBCIntEnvVars envs,
                              float *exps, float *coef, int npairs, float log_cutoff)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) {
        return;
    }
    int *bas = envs.bas;
    int bvk_nbas = envs.bvk_ncells * envs.nbas;
    int nimgs = envs.nimgs;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / bvk_nbas;
    int jsh = bas_ij / bvk_nbas;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    float ai = exps[ish];
    float aj = exps[jsh];
    float ci = coef[ish];
    float cj = coef[jsh];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    float log_cicj = logf(max(fabsf(ci * cj), 1e-37f));
    float log_fac = log_cicj + 1.717f - 1.5f * logf(aij) - log_cutoff;
    log_fac = max(log_fac, 1e-9f);
    float rr_raw = log_fac / theta;
    float Ecut_raw = log_fac * (2*aij);
    float Ecut_2a = Ecut_raw / (2*aij*aij);
    float dri_fac = .5f * logf(.5f*li/aij + fi*fi*rr_raw + Ecut_2a);
    float drj_fac = .5f * logf(.5f*lj/aij + fj*fj*rr_raw + Ecut_2a);
    // An approximate penalty for the polynomial part of the gaussian product
    log_fac += li * dri_fac + lj * drj_fac;

    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float Ecut_required = 0.f;
    for (int img = 0; img < nimgs; ++img) {
        float xjLxi = xjxi + img_coords[img*3+0];
        float yjLyi = yjyi + img_coords[img*3+1];
        float zjLzi = zjzi + img_coords[img*3+2];
        float rr = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        float Ecut_estimate = (log_fac - theta*rr) * (2*aij);
        Ecut_required = max(Ecut_estimate, Ecut_required);
    }
    Ecut[pair_id] = Ecut_required;
}

extern "C" {
int estimate_aft_Ecut1(float *Ecut, int64_t *bas_ij_idx, PBCIntEnvVars *envs,
                      float *exps, float *coef, int npairs, float log_cutoff)
{
    int blocks = (npairs + THREADS-1)/THREADS;
    estimate_aft_Ecut_kernel<<<blocks, THREADS>>>(
        Ecut, bas_ij_idx, *envs, exps, coef, npairs, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in estimate_aft_Ecut_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
