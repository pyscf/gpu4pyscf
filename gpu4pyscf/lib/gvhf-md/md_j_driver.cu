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
#include "gvhf-md/md_j.cuh"

extern __global__ void md_j_1dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley);
extern __global__ void md_j_4dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley, int dm_size);
int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, double omega);
int md_j_4dm_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, double omega, int dm_size);

static int block_id_for_threads(int threads)
{
    switch (threads) {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 8: return 3;
    case 16: return 4;
    case 32: return 5;
    }
    return 0;
}

extern "C" {
int MD_build_j(double *vj, double *dm, int n_dm, int dm_size,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int npairs_ij, int npairs_kl,
                int *pair_ij_mapping, int *pair_kl_mapping,
                int *pair_ij_loc, int *pair_kl_loc,
                float **qd_ij_max, float **qd_kl_max,
                float *q_cond, float cutoff,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int lij = li + lj;
    int lkl = lk + ll;
    int order = lij + lkl;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
    int nf3ijkl = (order+1)*(order+2)*(order+3)/6;
    float *tile16_qd_ij_max = qd_ij_max[block_id_for_threads(16)];
    float *tile16_qd_kl_max = qd_kl_max[block_id_for_threads(16)];
    MDBoundsInfo bounds = {li, lj, lk, ll, nf3ij, nf3kl, nf3ijkl,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        pair_ij_loc, pair_kl_loc, tile16_qd_ij_max, tile16_qd_kl_max,
        q_cond, cutoff};

    double omega = env[PTR_RANGE_OMEGA];
    JKMatrix jk = {vj, NULL, dm, n_dm, 0, omega};

    int threads_ij = scheme[0];
    int threads_kl = scheme[1];
    int gout_stride = scheme[2];
    int tilex = scheme[3];
    int tiley = scheme[4];
    int buflen = scheme[5];
    int bsizex = threads_ij * tilex;
    int bsizey = threads_kl * tiley;
    int nsq_per_block = threads_ij * threads_kl;
    dim3 threads(nsq_per_block, gout_stride);
    int blocks_ij = (npairs_ij + bsizex - 1) / bsizex;
    int blocks_kl = (npairs_kl + bsizey - 1) / bsizey;
    dim3 blocks(blocks_ij, blocks_kl);
    if (n_dm == 1) {
        if (!md_j_unrolled(&envs, &jk, &bounds, omega)) {
            bounds.qd_ij_max = qd_ij_max[block_id_for_threads(threads_ij)];
            bounds.qd_kl_max = qd_kl_max[block_id_for_threads(threads_kl)];
            md_j_1dm_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley);
        }
    } else {
        if (!md_j_4dm_unrolled(&envs, &jk, &bounds, omega, dm_size)) {
            bounds.qd_ij_max = qd_ij_max[block_id_for_threads(threads_ij)];
            bounds.qd_kl_max = qd_kl_max[block_id_for_threads(threads_kl)];
            md_j_4dm_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley, dm_size);
        }
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
    cudaFuncSetAttribute(md_j_1dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(md_j_4dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
