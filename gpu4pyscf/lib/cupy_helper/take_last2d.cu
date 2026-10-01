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

#include <cuda_runtime.h>
#include <stdio.h>
#include "gsycl/gpu_compat.h"
#define THREADS        32
#define COUNT_BLOCK     80

__global__
static void _take_last2d(double *a, const double *b, int *indices, int na, int nb)
{
    setup_context();
    size_t i = blockIdx_z;
    int j = global_x;
    int k = global_y;
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
    setup_context();
    int i0 = blockIdx_y * COUNT_BLOCK;
    int j = global_x;
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
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(ntile, ntile, blk_size);
    LAUNCH_KERNEL_S(_take_last2d, blocks, threads, 0, stream,
                  a, b, indices, na, nb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int takebak(cudaStream_t stream, double *out, double *a_h, int *indices,
            int count, int n_o, int n_a)
{
    double *a_d;
    int ntile = (n_a + THREADS*THREADS - 1) / (THREADS*THREADS);
    int ncount = (count + COUNT_BLOCK - 1) / COUNT_BLOCK;

    cudaError_t err = cudaHostGetDevicePointer((void **)&a_d, (void *)a_h, 0); // zero-copy check
    if (err != cudaSuccess) {
        return 1;
    }

    auto threads = make_block(THREADS*THREADS);
    auto blocks = make_grid(ntile, ncount);
    LAUNCH_KERNEL_S(_takebak, blocks, threads, 0, stream,
                  out, a_d, indices, count, n_o, n_a);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
