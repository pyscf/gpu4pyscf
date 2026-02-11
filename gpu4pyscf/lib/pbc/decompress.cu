/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include "pbc.cuh"

#define RBLKSIZE 16
#define CBLKSIZE 64
#define STRIDE   4
#define OF_COMPLEX 2

__global__ static
void d_t_kernel(double *out, double *cderi, int *pair_idx, int npairs, int nao,
                int naux, int aux0, int aux1, int fill_triu)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int bx = item.get_group(1);
    int by = item.get_group(0);
    int thread_id = item.get_local_id(1);
    double (&buf)[RBLKSIZE][CBLKSIZE+1] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[RBLKSIZE][CBLKSIZE+1]>(item.get_group());
    #else
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_id = threadIdx.x;
    __shared__ double buf[RBLKSIZE][CBLKSIZE+1];
    #endif

    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    if (pair_start+tx < npairs) {
        for (int k = ty; k < min(RBLKSIZE, daux-aux_start); k += STRIDE) {
            buf[k][tx] = cderi[(aux_start+k)*Npairs+pair_start+tx];
        }
    }
    __syncthreads();
    int stride = threads / RBLKSIZE;
    int pair_id = thread_id / RBLKSIZE;
    int aux_id = thread_id % RBLKSIZE;
    if (aux_start+aux_id < daux) {
        for (int k = pair_id; k < min(CBLKSIZE, npairs-pair_start); k += stride) {
            int pair_ij = pair_idx[pair_start+k];
            int i = pair_ij / nao;
            int j = pair_ij - nao * i;
            double s = buf[aux_id][k];
            out[(i*Nao+j)*daux+aux_start+aux_id] = s;
            if (fill_triu) {
                out[(j*Nao+i)*daux+aux_start+aux_id] = s;
            }
        }
    }
}

__global__ static
void z_d_t_kernel(double2 *out, double2 *cderi, int *pair_idx, int npairs, int nao,
                  int naux, int aux0, int aux1)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int bx = item.get_group(1);
    int by = item.get_group(0);
    int thread_id = item.get_local_id(1);
    double2 (&buf)[RBLKSIZE][CBLKSIZE+1] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double2[RBLKSIZE][CBLKSIZE+1]>(item.get_group());
    #else
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_id = threadIdx.x;
    __shared__ double2 buf[RBLKSIZE][CBLKSIZE+1];
    #endif

    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    if (pair_start+tx < npairs) {
        for (int k = ty; k < min(RBLKSIZE, daux-aux_start); k += STRIDE) {
            buf[k][tx] = cderi[(aux_start+k)*Npairs+pair_start+tx];
        }
    }
    __syncthreads();
    int stride = threads / RBLKSIZE;
    int pair_id = thread_id / RBLKSIZE;
    int aux_id = thread_id % RBLKSIZE;
    if (aux_start+aux_id < daux) {
        for (int k = pair_id; k < min(CBLKSIZE, npairs-pair_start); k += stride) {
            int pair_ij = pair_idx[pair_start+k];
            int i = pair_ij / nao;
            int j = pair_ij - nao * i;
            double2 s = buf[aux_id][k];
            out[(i*Nao+j)*daux+aux_start+aux_id] = s;
        }
    }
}

__global__ static
void store_col_segment_kernel(double *out, double *inp, int ncol, int col0, int col1)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int blockIdx_x = item.get_group(0);
    int threadIdx_x = item.get_local_id(0);
    int blockDim_x = item.get_local_range(0);
    #else
    int blockIdx_x = blockIdx.x;
    int threadIdx_x = threadIdx.x;
    int blockDim_x = blockDim.x;
    #endif

    int row = blockIdx_x;
    size_t Ncol = ncol;
    size_t dcol = col1 - col0;
    out += row * Ncol + col0;
    inp += row * dcol;
    for (int k = threadIdx_x; k < dcol; k += blockDim_x) {
        out[k] = inp[k];
    }
}

extern "C" {
int decompress_and_transpose(double *out, double *cderi, int *pair_idx,
                             int npairs, int nao, int naux, int aux0, int aux1,
                             int fill_triu, int on_host)
{
    #ifdef USE_SYCL
    // double *cderi is a pointer allocated via `malloc_shared`, hence no need of
    // equivalent to cudaHostGetDevicePointer
    sycl::range<2> threads(1, CBLKSIZE * STRIDE);
    sycl::range<2> blocks((aux1-aux0+RBLKSIZE-1)/RBLKSIZE, (npairs+CBLKSIZE-1)/CBLKSIZE);
    sycl_get_queue()->parallel_for<class d_t_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      d_t_kernel(out, cderi, pair_idx, npairs, nao, naux, aux0, aux1, fill_triu);
    });
    #else
    double *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, cderi, 0);
        if(err != cudaSuccess){
            fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    dim3 threads(CBLKSIZE * STRIDE);
    dim3 blocks((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    d_t_kernel<<<blocks, threads>>>(
            out, eri_gpu, pair_idx, npairs, nao, naux, aux0, aux1, fill_triu);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int z_decompress_and_transpose(double2 *out, double2 *cderi, int *pair_idx,
                               int npairs, int nao, int naux, int aux0, int aux1,
                               int fill_triu, int on_host)
{
    #ifdef USE_SYCL
    // double2 *cderi is a pointer allocated via `malloc_shared`, hence no need of
    // equivalent to cudaHostGetDevicePointer
    sycl::range<2> threads(1, (CBLKSIZE * STRIDE));
    sycl::range<2> blocks((aux1-aux0+RBLKSIZE-1)/RBLKSIZE, (npairs+CBLKSIZE-1)/CBLKSIZE);
    sycl_get_queue()->parallel_for<class z_d_t_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      z_d_t_kernel(out, cderi, pair_idx, npairs, nao, naux, aux0, aux1);
    });
    #else
    double2 *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, cderi, 0);
        if(err != cudaSuccess){
            fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    dim3 threads(CBLKSIZE * STRIDE);
    dim3 blocks((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    z_d_t_kernel<<<blocks, threads>>>(
            out, eri_gpu, pair_idx, npairs, nao, naux, aux0, aux1);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int store_col_segment(double *out, double *inp, int nrow, int ncol, int col0, int col1)
{
    #ifdef USE_SYCL
    // double *out is a pointer allocated via `malloc_shared`, hence no need of
    // equivalent to cudaHostGetDevicePointer
    sycl_get_queue()->parallel_for<class store_col_segment_sycl>(sycl::nd_range<1>(nrow * 512, 512), [=](auto item) {
      store_col_segment_kernel(out, inp, ncol, col0, col1);
    });
    #else
    double *out_gpu;
    cudaError_t err = cudaHostGetDevicePointer(&out_gpu, out, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "store_col_segment error %s\n", cudaGetErrorString(err));
        return 1;
    }
    dim3 blocks(nrow);
    store_col_segment_kernel<<<blocks, 512>>>(out_gpu, inp, ncol, col0, col1);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "store_col_segment error %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
