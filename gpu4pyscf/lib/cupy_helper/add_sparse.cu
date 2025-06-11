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
#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else
#include <cuda_runtime.h>
#endif

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _add_sparse(double *a, double *b, int *indices, int n, int m, int count)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
#else
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
#endif
    int row = blockIdx_x * BLOCK_DIM + threadIdx_x;
    int col = blockIdx_y * BLOCK_DIM + threadIdx_y;
    if (row >= m || col >= m){
        return;
    }
    int idx_a = indices[row] * n + indices[col];
    int idx_b = row * m + col;
    for (int i = 0; i < count; i++){
        a[idx_a + i*n*n] += b[idx_b + i*m*m];
    }
}

extern "C" {
__host__
int add_sparse(cudaStream_t stream, double *a, double *b, int *indices, int n, int m, int count){
    int ntile = (m + THREADS - 1) / THREADS;
#ifdef USE_SYCL
    sycl::range<2> threads(THREADS, THREADS);
    sycl::range<2> blocks(ntile, ntile);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      _add_sparse(a, b, indices, n, m, count);          
    });
#else // USE_SYCL
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _add_sparse<<<blocks, threads, 0, stream>>>(a, b, indices, n, m, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
#endif
    return 0;
}
}
