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
#include "ft_ao.cuh"

__constant__ int c_g_pair_idx[3675]; // corresponding to LMAX=4
__constant__ int c_g_pair_offsets[LMAX1*LMAX1];
__constant__ int c_g_cart_idx[252]; // corresponding to LMAX=6

extern __global__
void ft_aopair_kernel(double *out, AFTIntEnvVars envs, AFTBoundsInfo bounds,
                      int compressing);
extern __global__
void ft_aopair_fill_triu(double *out, int *conj_mapping, int bvk_ncells, int nGv);
extern __global__
void pbc_int3c2e_kernel(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds);

int ft_ao_unrolled(double *out, AFTIntEnvVars *envs, AFTBoundsInfo *bounds,
                   int *scheme, int compressing);
int int3c2e_unrolled(double *out, PBCInt3c2eEnvVars *envs, PBCInt3c2eBounds *bounds);

extern "C" {
int build_ft_ao(double *out, int compressing, AFTIntEnvVars *envs,
                int *scheme, int *shls_slice, int npairs_ij, int ngrids,
                int *bas_ij, double *grids, int *img_offsets, int *img_idx,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t stride_i = 1;
    uint8_t stride_j = li + 1;
    // up to g functions
    uint8_t g_size = stride_j * (uint16_t)(lj + 1);
    AFTBoundsInfo bounds = {li, lj, nfij, g_size,
        stride_i, stride_j, iprim, jprim,
        npairs_ij, ngrids, bas_ij, grids, img_offsets, img_idx};

    if (!ft_ao_unrolled(out, envs, &bounds, scheme, compressing)) {
        int nGv_per_block = scheme[0];
        int gout_stride = scheme[1];
        int nsp_per_block = scheme[2];
        dim3 threads(nGv_per_block, gout_stride, nsp_per_block);
        int sp_blocks = (npairs_ij + nsp_per_block - 1) / nsp_per_block;
        int Gv_batches = (ngrids + nGv_per_block - 1) / nGv_per_block;
        dim3 blocks(sp_blocks, Gv_batches);
        int buflen = g_size*6 * nGv_per_block * nsp_per_block;
        ft_aopair_kernel<<<blocks, threads, buflen*sizeof(double)>>>(
                out, *envs, bounds, compressing);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in build_ft_ao: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int ft_aopair_fill_triu(double *out, int *conj_mapping, int nao, int bvk_ncells, int nGv)
{
    int nGv2 = nGv * 2; // *2 for complex number
    int threads = 1024;
    dim3 blocks(nao, nao);
    ft_aopair_fill_triu<<<blocks, threads>>>(out, conj_mapping, bvk_ncells, nGv2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_fill_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int fill_int3c2e(double *out, PBCInt3c2eEnvVars *envs, int *scheme, int *shls_slice,
                 int naux, int n_prim_pairs, int n_ctr_pairs,
                 int *bas_ij_idx, int *pair_mapping, int *img_idx, int *img_offsets,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4] + nbas;
    uint16_t ksh1 = shls_slice[5] + nbas;
    uint16_t nksh = ksh1 - ksh0;
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t order = li + lj + lk;
    uint8_t nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 1;
    uint8_t stride_k = stride_j * (lj + 1);
    // up to (gg|i)
    uint8_t g_size = stride_k * (lk + 1);
    PBCInt3c2eBounds bounds = {
        (uint8_t)li, (uint8_t)lj, lk, nroots, nfij, nfk, kprim,
        stride_j, stride_k, g_size, (uint16_t)naux, (uint16_t)nksh, (uint16_t)ksh0,
        n_prim_pairs, n_ctr_pairs,
        bas_ij_idx, pair_mapping, img_offsets, img_idx
    };

    if (!int3c2e_unrolled(out, envs, &bounds)) {
        int nksh_per_block = scheme[0];
        int gout_stride = scheme[1];
        int nsp_per_block = scheme[2];
        dim3 threads(nksh_per_block, gout_stride, nsp_per_block);
        int tasks_per_block = SPTAKS_PER_BLOCK * nsp_per_block;
        int sp_blocks = (n_prim_pairs + tasks_per_block - 1) / tasks_per_block;
        int ksh_blocks = (nksh + nksh_per_block - 1) / nksh_per_block;
        dim3 blocks(sp_blocks, ksh_blocks);
        int buflen = (nroots*2+g_size*3+7) * (nksh_per_block * nsp_per_block) * sizeof(double);
        pbc_int3c2e_kernel<<<blocks, threads, buflen>>>(out, *envs, bounds);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int init_constant(int *g_pair_idx, int *offsets,
                  double *env, int env_size, int shm_size)
{
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);

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
    cudaMemcpyToSymbol(c_g_cart_idx, g_cart_idx, 252*sizeof(int));
    free(g_cart_idx);

    cudaFuncSetAttribute(ft_aopair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(pbc_int3c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
