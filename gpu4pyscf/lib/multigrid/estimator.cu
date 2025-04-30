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
#include "multigrid.cuh"

#define REMOTE_THRESHOLD 50

// An estimation of the upper bound of the overlap |<cell0|supcmol>| for
// shell pairs between the primitve cell and the super-mol
__global__ static
void overlap_estimation_kernel(float *log_ovlp, float *exps, float *log_coeff,
                               float *bas_coords, int *ao_loc_in_cell0,
                               int *ls, int cell0_nbas, int nbas, int hermi)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int npairs = cell0_nbas * nbas;
    if (bas_ij >= npairs) {
        return;
    }
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    // assume the hermitian symmetry in Coulomb matrix.
    // Note: hermitian symmetry might not be available in methods like TDDFT
    if (hermi && ao_loc_in_cell0[ish] < ao_loc_in_cell0[jsh]) {
        log_ovlp[bas_ij] = -1000;
        return;
    }

    int li = ls[ish];
    int lj = ls[jsh];
    float ai = exps[ish];
    float aj = exps[jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    float *bas_x = bas_coords;
    float *bas_y = bas_coords + nbas;
    float *bas_z = bas_coords + nbas * 2;
    float xi = bas_x[ish];
    float yi = bas_y[ish];
    float zi = bas_z[ish];
    float xj = bas_x[jsh];
    float yj = bas_y[jsh];
    float zj = bas_z[jsh];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
    float dr = sqrtf(rr_ij);
    float dri = fj * dr;
    float drj = fi * dr;
    float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
    float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
    float fac_norm = log_coeff[ish] + log_coeff[jsh] + 1.717f - 1.5f * logf(aij);
    float s = fac_norm - theta*rr_ij + dri_fac + drj_fac;
    log_ovlp[bas_ij] = min(s, 0.f);
}

extern "C" {
int overlap_estimation(float *log_ovlp, float *exps, float *log_coeff,
                       float *bas_coords, int *ao_loc_in_cell0,
                       int *ls, int cell0_nbas, int nbas, int hermi)
{
    constexpr int threads = 512;
    int blocks = (cell0_nbas*nbas + threads-1)/threads;
    overlap_estimation_kernel<<<blocks, threads>>>(
        log_ovlp, exps, log_coeff, bas_coords, ao_loc_in_cell0,
        ls, cell0_nbas, nbas, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in overlap_estimation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
