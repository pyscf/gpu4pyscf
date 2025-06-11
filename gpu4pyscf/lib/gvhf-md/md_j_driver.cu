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
#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else
#include <cuda_runtime.h>
#endif
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/md_j.cuh"

#ifdef USE_SYCL
SYCL_EXTERNAL sycl_device_global<Fold2Index[165]> s_i_in_fold2idx;
SYCL_EXTERNAL sycl_device_global<Fold3Index[495]> s_i_in_fold3idx;
#else
__constant__ Fold2Index c_i_in_fold2idx[165];
__constant__ Fold3Index c_i_in_fold3idx[495];
#endif

#ifdef USE_SYCL
SYCL_EXTERNAL __global__ void md_j_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley, sycl::nd_item<2> &item, double *shm_mem);
SYCL_EXTERNAL __global__ void md_j_s4_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley, sycl::nd_item<2> &item, double *shm_mem);
#else
extern __global__ void md_j_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley);
extern __global__ void md_j_s4_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                                   int threadsx, int threadsy, int tilex, int tiley);
#endif

int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds);

extern "C" {
int MD_build_j(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int npairs_ij, int npairs_kl,
                int *pair_ij_mapping, int *pair_kl_mapping,
                int *pair_ij_loc, int *pair_kl_loc,
                float **qd_ij_max, float **qd_kl_max,
                float *q_cond, float cutoff,
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
    float *tile16_qd_ij_max = qd_ij_max[4];
    float *tile16_qd_kl_max = qd_kl_max[4];
    MDBoundsInfo bounds = {li, lj, lk, ll,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        pair_ij_loc, pair_kl_loc, tile16_qd_ij_max, tile16_qd_kl_max,
        q_cond, cutoff};

    JKMatrix jk = {vj, NULL, dm, (uint16_t)n_dm};

    if (!md_j_unrolled(&envs, &jk, &bounds)) {
        int lij = li + lj;
        int lkl = lk + ll;
        int threads_ij = scheme[0];
        int threads_kl = scheme[1];
        int gout_stride = scheme[2];
        int tilex = scheme[3];
        int tiley = scheme[4];
        switch (threads_ij) {
        case 1: bounds.qd_ij_max = qd_ij_max[0]; break;
        case 2: bounds.qd_ij_max = qd_ij_max[1]; break;
        case 4: bounds.qd_ij_max = qd_ij_max[2]; break;
        case 8: bounds.qd_ij_max = qd_ij_max[3]; break;
        case 16: bounds.qd_ij_max = qd_ij_max[4]; break;
        case 32: bounds.qd_ij_max = qd_ij_max[5]; break;
        }
        switch (threads_kl) {
        case 1: bounds.qd_kl_max = qd_kl_max[0]; break;
        case 2: bounds.qd_kl_max = qd_kl_max[1]; break;
        case 4: bounds.qd_kl_max = qd_kl_max[2]; break;
        case 8: bounds.qd_kl_max = qd_kl_max[3]; break;
        case 16: bounds.qd_kl_max = qd_kl_max[4]; break;
        case 32: bounds.qd_kl_max = qd_kl_max[5]; break;
        }
        int bsizex = threads_ij * tilex;
        int bsizey = threads_kl * tiley;
        int nsq_per_block = threads_ij * threads_kl;
        int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int blocks_ij = (npairs_ij + bsizex - 1) / bsizex;
        int blocks_kl = (npairs_kl + bsizey - 1) / bsizey;
//        if (li == lk && lj == ll) {
//            int buflen = (order+1) * nsq_per_block
//                + threads_ij * 4 + bsizey * 4
//                + nf3ij * threads_ij + nf3kl * threads_kl
//                + (order+1)*(order+2)*(order+3)/6 * nsq_per_block;
//            buflen += max(order*(order+1)*(order+2)/6, gout_stride) * nsq_per_block;
//            md_j_s4_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
//                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley);
//        } else {
            int buflen = (order+1) * nsq_per_block
                + threads_ij * 4 + bsizey * 4
                + nf3ij * threads_ij * 2 + nf3kl * threads_kl * 2
                + (order+1)*(order+2)*(order+3)/6 * nsq_per_block;
            buflen += max(order*(order+1)*(order+2)/6, gout_stride) * nsq_per_block;

            #ifdef USE_SYCL
            sycl::range<2> threads(gout_stride, threads_ij*threads_kl);
            sycl::range<2> blocks(blocks_kl, blocks_ij);
            sycl_get_queue()->submit([&](sycl::handler &cgh) {
              sycl::local_accessor<double, 1> local_acc(buflen, cgh);
              cgh.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
                md_j_kernel(envs, jk, bounds, threads_ij, threads_kl, tilex, tiley,
                            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
              });
            });
            #else
            dim3 threads(threads_ij*threads_kl, gout_stride);
            dim3 blocks(blocks_ij, blocks_kl);
            md_j_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley);
            #endif
        }
//    }

#ifndef USE_SYCL
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MD_build_j: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
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
    #ifdef USE_SYCL
    sycl_get_queue()->memcpy(s_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index)).wait();
    sycl_get_queue()->memcpy(s_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index)).wait();
    #else
    cudaMemcpyToSymbol(c_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index));
    cudaMemcpyToSymbol(c_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index));
    cudaFuncSetAttribute(md_j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(md_j_s4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
