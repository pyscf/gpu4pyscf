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


#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#define THREADS        32
#define COUNT_BLOCK     80

__global__
static void _take_last2d(double *a, const double *b, int *indices, int na, int nb)
{
  #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    size_t i = item.get_group(0);
    int j = item.get_global_id(2);
    int k = item.get_global_id(1);
#else
    size_t i = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (j >= na || k >= na) {
        return;
    }

    int j_b = indices[j];
    int k_b = indices[k];
    size_t offa = i * na * na;
    size_t offb = i * nb * nb;
    a[offa + j * na + k] = b[offb + j_b * nb + k_b];
}

__global__
static void _takebak(double *out, double *a, int *indices,
                     int count, int n_o, int n_a)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int i0 = item.get_group(0) * COUNT_BLOCK;
    int j = item.get_global_id(1);
#else
    int i0 = blockIdx.y * COUNT_BLOCK;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (j >= n_a) {
        return;
    }

    // a is on host with zero-copy memory. We need enough iterations for
    // data prefetch to hide latency
    int i1 = i0 + COUNT_BLOCK;
    if (i1 > count) i1 = count;
    int jp = indices[j];
#pragma unroll
    for (size_t i = i0; i < i1; ++i) {
        out[i * n_o + jp] = a[i * n_a + j];
    }
}

extern "C" {
int take_last2d(cudaStream_t stream, double *a, const double *b, int *indices,
                int blk_size, int na, int nb)
{
    // reorder j and k in a[i,j,k] with indicies
    int ntile = (na + THREADS - 1) / THREADS;
    #ifdef USE_SYCL
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(blk_size, ntile, ntile);
    stream.parallel_for<class _take_last2d_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _take_last2d(a, b, indices, na, nb);
    });
    #else
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, blk_size);
    _take_last2d<<<blocks, threads, 0, stream>>>(a, b, indices, na, nb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    #endif
    return 0;
}

int takebak(cudaStream_t stream, double *out, double *a_h, int *indices,
            int count, int n_o, int n_a)
{
    double *a_d;
    int ntile = (n_a + THREADS*THREADS - 1) / (THREADS*THREADS);
    int ncount = (count + COUNT_BLOCK - 1) / COUNT_BLOCK;

    #ifdef USE_SYCL
    *(void **)&a_d = (double *)a_h;
    sycl::range<2> threads(1, THREADS*THREADS);
    sycl::range<2> blocks(ncount, ntile);
    stream.parallel_for<class _takebak_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      _takebak(out, a_d, indices, count, n_o, n_a);
    });
    #else
    cudaError_t err;
    err = cudaHostGetDevicePointer(&a_d, a_h, 0); // zero-copy check
    if (err != cudaSuccess) {
        return 1;
    }

    dim3 threads(THREADS*THREADS);
    dim3 blocks(ntile, ncount);
    _takebak<<<blocks, threads, 0, stream>>>(out, a_d, indices, count, n_o, n_a);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    #endif
    return 0;
}
}
