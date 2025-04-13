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
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

__constant__ Fold2Index c_i_in_fold2idx[165];
__constant__ Fold3Index c_i_in_fold3idx[495];

extern __global__ void md_j_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley);
int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int workers);

extern "C" {
int MD_build_j(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int ntile_ij_pairs, int ntile_kl_pairs,
                int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                uint32_t *batch_head, int workers,
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
    uint8_t order = li + lj + lk + ll;
    BoundsInfo bounds = {li, lj, lk, ll,
        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0 , 0,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, NULL, dm, (uint16_t)n_dm};

    if (!md_j_unrolled(&envs, &jk, &bounds, workers)) {
        int lij = li + lj;
        int lkl = lk + ll;
        int threads_ij = scheme[0];
        int threads_kl = scheme[1];
        int gout_stride = scheme[2];
        int tilex = scheme[3];
        int tiley = scheme[4];
        int bsizex = threads_ij * tilex;
        int bsizey = threads_kl * tiley;
        int nsq_per_block = threads_ij * threads_kl;
        int threads = threads_ij * threads_kl * gout_stride;
        int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int buflen = (order+1) * nsq_per_block
            + bsizex * (4+nf3ij) + bsizey * (4+nf3kl)
            + (order+1)*(order+2)*(order+3)/6 * nsq_per_block;
        buflen += MAX(order*(order+1)*(order+2)/6, gout_stride) * nsq_per_block;
        int blocks_ij = (ntile_ij_pairs + bsizex - 1) / bsizex;
        int blocks_kl = (ntile_kl_pairs + bsizey - 1) / bsizey;
        dim3 blocks(blocks_ij, blocks_kl);
        md_j_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
            envs, jk, bounds, threads_ij, threads_kl, tilex, tiley);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MD_build_j: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int init_mdj_constant(int shm_size)
{
    Fold2Index i_in_fold2idx[165];
    Fold3Index i_in_fold3idx[495];
    int n2 = 0;
    int n3 = 0;
    for (int l = 0; l <= LMAX*2; ++l) {
        for (int i = 0, ijk = 0; i <= l; ++i) {
        for (int j = 0; j <= l-i; ++j, ++n2) {
            i_in_fold2idx[n2].x = i;
            i_in_fold2idx[n2].y = j;
            i_in_fold2idx[n2].fold3offset = ijk;
            for (int k = 0; k <= l-i-j; ++k, ++n3, ++ijk) {
                i_in_fold3idx[n3].x = i;
                i_in_fold3idx[n3].y = j;
                i_in_fold3idx[n3].z = k;
                i_in_fold3idx[n3].fold2yz = (l+1)*(l+2)/2 - (l-j+1)*(l-j+2)/2 + k;
            }
        } }
    }
    cudaMemcpyToSymbol(c_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index));
    cudaMemcpyToSymbol(c_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index));
    cudaFuncSetAttribute(md_j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
