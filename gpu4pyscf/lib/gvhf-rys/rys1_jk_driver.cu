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
#include <cuda.h>
#include <cuda_runtime.h>

#include "vhf.cuh"

extern __global__
void rys1_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, double *hrr_pool, uint32_t *batch_head);

extern "C" {
int RYS1_build_jk(double *vj, double *vk, double *dm, int n_dm, int nao,
                 RysIntEnvVars envs, int *scheme, int *shls_slice,
                 int ntile_ij_pairs, int ntile_kl_pairs,
                 int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                 float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                 ShellQuartet *pool, double *hrr_pool, uint32_t *batch_head, int workers,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 1;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, vk, dm, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (1) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        dim3 threads(quartets_per_block, gout_stride);
        int ij_prims = iprim * jprim;
        size_t buflen = ij_prims * quartets_per_block * sizeof(double);
        buflen = MAX(quartets_per_block*gout_stride*sizeof(int), buflen);
        rys1_jk_kernel<<<workers, threads, buflen>>>(
                envs, jk, bounds, pool, hrr_pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in RYS1_build_jk: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int RYS_init_constant(int *g_pair_idx, int *offsets,
                      double *env, int env_size, int shm_size);

int RYS1_init_constant(int *g_pair_idx, int *offsets,
                      double *env, int env_size, int shm_size)
{
    RYS_init_constant(g_pair_idx, offsets, env, env_size, shm_size);
    cudaFuncSetAttribute(rys1_jk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
