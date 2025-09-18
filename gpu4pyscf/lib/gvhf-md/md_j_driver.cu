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
                     int threadsx, int threadsy, int tilex, int tiley,
                     uint16_t *pRt2_ij_kl, uint16_t *pRt2_kl_ij, int8_t *efg_phase);
extern __global__ void md_j_4dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                     int threadsx, int threadsy, int tilex, int tiley, int dm_size,
                     uint16_t *pRt2_ij_kl, uint16_t *pRt2_kl_ij, int8_t *efg_phase);
int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, double omega);
int md_j_4dm_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, double omega, int dm_size);

extern __constant__ int8_t c_Rt2_efg_phase[];
extern __device__ uint16_t Rt2_kl_ij[];
extern __device__ uint16_t Rt2_ij_kl[];

#define RT2_MAX 9
static int _Rt2_idx_offsets[] = {
0,1,5,15,35,70,126,210,330,
495,499,515,555,635,775,999,1335,1815,
2475,2485,2525,2625,2825,3175,3735,4575,5775,
7425,7445,7525,7725,8125,8825,9945,11625,14025,
17325,17360,17500,17850,18550,19775,21735,24675,28875,
34650,34706,34930,35490,36610,38570,41706,46410,53130,
62370,62454,62790,63630,65310,68250,72954,80010,90090,
103950,104070,104550,105750,108150,112350,119070,129150,143550,
163350,163515,164175,165825,169125,174900,184140,198000,217800,
245025,
};

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
    // 16x16 threads are applied to all unrolled code
    float *tile16_qd_ij_max = qd_ij_max[block_id_for_threads(16)];
    float *tile16_qd_kl_max = qd_kl_max[block_id_for_threads(16)];
    MDBoundsInfo bounds = {li, lj, lk, ll, lij, lkl, order, nf3ij, nf3kl, nf3ijkl,
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
    uint16_t *pRt2_ij_kl;// = Rt2_ij_kl + _Rt2_idx_offsets[lij*RT2_MAX+lkl];
    uint16_t *pRt2_kl_ij;// = Rt2_kl_ij + _Rt2_idx_offsets[lij*RT2_MAX+lkl];
    int8_t *efg_phase;
    cudaGetSymbolAddress((void**)&pRt2_ij_kl, Rt2_ij_kl);
    cudaGetSymbolAddress((void**)&pRt2_kl_ij, Rt2_kl_ij);
    cudaGetSymbolAddress((void**)&efg_phase, c_Rt2_efg_phase);
    pRt2_kl_ij += _Rt2_idx_offsets[lij*RT2_MAX+lkl];
    pRt2_ij_kl += _Rt2_idx_offsets[lij*RT2_MAX+lkl];
    efg_phase += _Rt2_idx_offsets[lkl];
    if (n_dm == 1) {
        if (!md_j_unrolled(&envs, &jk, &bounds, omega)) {
            bounds.qd_ij_max = qd_ij_max[block_id_for_threads(threads_ij)];
            bounds.qd_kl_max = qd_kl_max[block_id_for_threads(threads_kl)];
            md_j_1dm_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley,
                pRt2_ij_kl, pRt2_kl_ij, efg_phase);
        }
    } else {
        if (!md_j_4dm_unrolled(&envs, &jk, &bounds, omega, dm_size)) {
            bounds.qd_ij_max = qd_ij_max[block_id_for_threads(threads_ij)];
            bounds.qd_kl_max = qd_kl_max[block_id_for_threads(threads_kl)];
            for (int dm_offset = 0; dm_offset < n_dm; dm_offset+=4) {
                jk.vj = vj + dm_offset * dm_size;
                jk.dm = dm + dm_offset * dm_size;
                jk.n_dm = n_dm - dm_offset;
                bounds.qd_ij_max = qd_ij_max[block_id_for_threads(threads_ij)];
                bounds.qd_kl_max = qd_kl_max[block_id_for_threads(threads_kl)];
                md_j_4dm_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                    envs, jk, bounds, threads_ij, threads_kl, tilex, tiley, dm_size,
                    pRt2_ij_kl, pRt2_kl_ij, efg_phase);
            }
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
