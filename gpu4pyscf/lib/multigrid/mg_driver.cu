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

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define MALLOC(type, var, size) \

#define MEMSET(addr, val, size) \
    checkCudaErrors(cudaMemset(addr, val, size))


__constant__ Fold2Index c_i_in_fold2idx[165];
__constant__ Fold3Index c_i_in_fold3idx[495];
int eval_rho_orth(double *rho, double *dm, MGridEnvVars *envs, MGridBounds *bounds,
                  int l, double *pool, uint32_t *batch_head, int workers);
int eval_mat_lda_orth(double *out, double *rho, MGridEnvVars *envs, MGridBounds *bounds,
                      int l, double *pool, uint32_t *batch_head, int workers);

extern "C" {
int MG_eval_rho_orth(double *rho, double *dm, MGridEnvVars envs,
                     int l, int n_radius, int *mesh, int nshl_pair,
                     int *bas_ij_idx, double *pool, int workers)
{
    MGridBounds bounds = {
        nshl_pair, bas_ij_idx, n_radius, {mesh[0], mesh[1], mesh[2]},
    };
    uint32_t *batch_head;
    cudaMalloc(reinterpret_cast<void **>(&batch_head), sizeof(uint32_t) * 1);
    cudaMemset(batch_head, 0, sizeof(uint32_t));
    eval_rho_orth(rho, dm, &envs, &bounds, l, pool, batch_head, workers);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_rho_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}

int MG_eval_mat_lda_orth(double *out, double *rho, MGridEnvVars envs,
                         int l, int n_radius, int *mesh, int nshl_pair,
                         int *bas_ij_idx, double *pool, int workers)
{
    MGridBounds bounds = {
        nshl_pair, bas_ij_idx, n_radius, {mesh[0], mesh[1], mesh[2]},
    };
    uint32_t *batch_head;
    cudaMalloc(reinterpret_cast<void **>(&batch_head), sizeof(uint32_t) * 1);
    cudaMemset(batch_head, 0, sizeof(uint32_t));
    eval_mat_lda_orth(out, rho, &envs, &bounds, l, pool, batch_head, workers);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_rho_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}

int MG_init_constant(int shm_size)
{
    Fold2Index i_in_fold2idx[165];
    Fold3Index i_in_fold3idx[495];
    int n2 = 0;
    int n3 = 0;
    for (int l = 0; l <= LMAX*2; ++l) {
        for (int i = 0; i <= l; ++i) {
        for (int j = 0; j <= l-i; ++j) {
            for (int k = 0; k <= l-i-j; ++k, ++n3) {
                i_in_fold3idx[n3].x = i;
                i_in_fold3idx[n3].y = j;
                i_in_fold3idx[n3].z = k;
            }
        } }
        for (int i = l; i >= 0; --i) {
        for (int j = l-i; j >= 0; --j, ++n2) {
            i_in_fold2idx[n2].x = i;
            i_in_fold2idx[n2].y = j;
        } }
    }
    cudaMemcpyToSymbol(c_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index));
    cudaMemcpyToSymbol(c_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
