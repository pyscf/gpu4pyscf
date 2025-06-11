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

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _transpose_sum(double *a, int n)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    sycl::group thread_block = item.get_group();
    int blockIdx_x = item.get_group(2);
    int blockIdx_y = item.get_group(1);
    int blockIdx_z = item.get_group(0);
    int threadIdx_x = item.get_local_id(2);
    int threadIdx_y = item.get_local_id(1);
    using tile_t = double[BLOCK_DIM][BLOCK_DIM+1];
    tile_t& block = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
#else
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int blockIdx_z = blockIdx.z;
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    __shared__ double block[BLOCK_DIM][BLOCK_DIM+1];
#endif

    if(blockIdx_x > blockIdx_y){
        return;
    }

    unsigned int blockx_off = blockIdx_x * BLOCK_DIM;
    unsigned int blocky_off = blockIdx_y * BLOCK_DIM;
	unsigned int x0 = blockx_off + threadIdx_x;
	unsigned int y0 = blocky_off + threadIdx_y;
    unsigned int x1 = blocky_off + threadIdx_x;
	unsigned int y1 = blockx_off + threadIdx_y;
    unsigned int z = blockIdx_z;

    size_t off = n * n * z;
    size_t xy0 = y0 * n + x0 + off;
    size_t xy1 = y1 * n + x1 + off;

    if (x0 < n && y0 < n){
        block[threadIdx_y][threadIdx_x] = a[xy0];
    }
    __syncthreads();
    if (x1 < n && y1 < n){
        block[threadIdx_x][threadIdx_y] += a[xy1];
    }
    __syncthreads();

    if(x0 < n && y0 < n){
        a[xy0] = block[threadIdx_y][threadIdx_x];
    }
    if(x1 < n && y1 < n){
        a[xy1] = block[threadIdx_x][threadIdx_y];
    }
}

extern "C" {
__host__
int transpose_sum(cudaStream_t stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
#ifdef USE_SYCL
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(counts, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _transpose_sum(a, n);    
    });
#else //USE_SYCL
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _transpose_sum<<<blocks, threads, 0, stream>>>(a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
#endif //USE_SYCL
    return 0;
}
}
