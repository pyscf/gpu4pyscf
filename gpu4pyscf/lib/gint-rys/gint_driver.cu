/*
 * Copyright 2024 The PySCF Developers. All Rights Reserved.
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

#include "gvhf-rys/vhf.cuh"
#include "int3c2e.cuh"

#ifdef USE_SYCL
SYCL_EXTERNAL sycl_device_global<int[3675]> s_g_pair_idx; // corresponding to LMAX=4
SYCL_EXTERNAL sycl_device_global<int[LMAX1*LMAX1]> s_g_pair_offsets;
SYCL_EXTERNAL sycl_device_global<int[252]> s_g_cart_idx; // corresponding to LMAX=6
#else
__constant__ int c_g_pair_idx[3675]; // corresponding to LMAX=4
__constant__ int c_g_pair_offsets[LMAX1*LMAX1];
__constant__ int c_g_cart_idx[252]; // corresponding to LMAX=6
#endif

extern __global__
#ifdef USE_SYCL
SYCL_EXTERNAL
#endif
void int3c2e_kernel(double *out, Int3c2eEnvVars envs, Int3c2eBounds bounds
                    #ifdef USE_SYCL
                    , sycl::nd_item<2> &item, char *rw_buffer
                    #endif
                    );
int int3c2e_unrolled(double *out, Int3c2eEnvVars *envs, Int3c2eBounds *bounds);

extern __global__
#ifdef USE_SYCL
SYCL_EXTERNAL
#endif
void int3c2e_bdiv_kernel(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds
                         #ifdef USE_SYCL
                         , sycl::nd_item<2> &item, char *rw_buffer
                         #endif
                         );

extern "C" {
int fill_int3c2e(double *out, Int3c2eEnvVars *envs, int *scheme, int *shls_slice,
                 int *aux_loc, int naux, int nshl_pair, int *bas_ij_idx,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4] + nbas;
    int ksh1 = shls_slice[5] + nbas;
    int nksh = ksh1 - ksh0;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    int jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfij = nfi * nfj;
    int order = li + lj + lk;
    int nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int stride_i = 1;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    // up to (gg|i)
    int g_size = stride_k * (lk + 1);
    Int3c2eBounds bounds = {li, lj, lk, nroots, nfi, nfij, nfk,
        iprim, jprim, kprim, stride_i, stride_j, stride_k, g_size,
        naux, nksh, ksh0, nshl_pair, bas_ij_idx};

    int k0 = aux_loc[ksh0 - nbas];
    out += k0; // offset when writing output
    if (!int3c2e_unrolled(out, envs, &bounds)) {
        int nst_per_block = scheme[0];
        int gout_stride = scheme[1];
        int tasks_per_block = BATCHES_PER_BLOCK * nst_per_block;
        int st_blocks = (nksh*nshl_pair + tasks_per_block - 1) / tasks_per_block;
        int buflen = (nroots*2+g_size*3+7) * nst_per_block * sizeof(double);
        #ifdef USE_SYCL 
        sycl::range<2> threads(gout_stride, nst_per_block);
        sycl::range<2> blocks(1, st_blocks);
        auto dev_envs = *envs;
        sycl_get_queue()->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<char, 1> local_acc(sycl::range<1>(buflen), cgh);
          cgh.parallel_for<class int3c2e_kernel_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
            int3c2e_kernel(out, dev_envs, bounds,
                           item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
          }); });       
        #else
        dim3 threads(nst_per_block, gout_stride);        
        int3c2e_kernel<<<st_blocks, threads, buflen>>>(out, *envs, bounds);
        #endif
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int3c2e_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}  


int fill_int3c2e_bdiv(double *out, Int3c2eEnvVars *envs, int shm_size, int naux,
                      int nbatches_shl_pair, int nbatches_ksh, int aux_sh_offset,
                      int *shl_pair_offsets, int *ao_pair_loc, int *ksh_offsets,
                      int *bas_ij_idx, int *nst_lookup,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    BDiv3c2eBounds bounds = {naux, aux_sh_offset, bas_ij_idx, shl_pair_offsets,
        ao_pair_loc, ksh_offsets, nst_lookup};
    int threads = 256;
#ifdef USE_SYCL
    sycl::range<2> thread(1, threads);
    sycl::range<2> blocks(nbatches_ksh, nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) {
        int3c2e_bdiv_kernel(out, dev_envs, bounds,
                            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      }); });
#else // USE_SYCL
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    int3c2e_bdiv_kernel<<<blocks, threads, shm_size>>>(out, *envs, bounds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int3c2e_bdiv_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif // USE_SYCL
    return 0;
}

int init_constant(int *g_pair_idx, int *offsets,
                  double *env, int env_size, int shm_size)
{
    #ifdef USE_SYCL
    sycl_get_queue()->memcpy(s_g_pair_idx, g_pair_idx, 3675*sizeof(int)).wait();
    sycl_get_queue()->memcpy(s_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1).wait();
    #else
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);
    #endif

    int *g_cart_idx = (int *)malloc(252*sizeof(int));
    int *idx, *idy, *idz;
    idx = g_cart_idx;
    for (int l = 0; l <= L_AUX_MAX; ++l) {
        int nf = (l + 1) * (l + 2) / 2;
        idy = idx + nf;
        idz = idy + nf;
        for (int i = 0, ix = l; ix >= 0; --ix) {
        for (int iy = l - ix; iy >= 0; --iy, ++i) {
            int iz = l - ix - iy;
            idx[i] = ix;
            idy[i] = iy;
            idz[i] = iz;
        } }
        idx += nf * 3;
    }
    #ifdef USE_SYCL
    sycl_get_queue()->memcpy(s_g_cart_idx, g_cart_idx, 252*sizeof(int)).wait();
    #else
    cudaMemcpyToSymbol(c_g_cart_idx, g_cart_idx, 252*sizeof(int));

    cudaFuncSetAttribute(int3c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(int3c2e_bdiv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    #endif // USE_SYCL

    free(g_cart_idx);
    return 0;
}
}
