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

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _transpose_sum(double *a, int n)
{
    if(blockIdx.x > blockIdx.y){
        return;
    }
	__shared__ double block[BLOCK_DIM][BLOCK_DIM+1];

    unsigned int blockx_off = blockIdx.x * BLOCK_DIM;
    unsigned int blocky_off = blockIdx.y * BLOCK_DIM;
	unsigned int x0 = blockx_off + threadIdx.x;
	unsigned int y0 = blocky_off + threadIdx.y;
    unsigned int x1 = blocky_off + threadIdx.x;
	unsigned int y1 = blockx_off + threadIdx.y;
    unsigned int z = blockIdx.z;

    size_t off = n * n * z;
    size_t xy0 = y0 * n + x0 + off;
    size_t xy1 = y1 * n + x1 + off;

    if (x0 < n && y0 < n){
        block[threadIdx.y][threadIdx.x] = a[xy0];
    }
    __syncthreads();
    if (x1 < n && y1 < n){
        block[threadIdx.x][threadIdx.y] += a[xy1];
    }
    __syncthreads();

    if(x0 < n && y0 < n){
        a[xy0] = block[threadIdx.y][threadIdx.x];
    }
    if(x1 < n && y1 < n){
        a[xy1] = block[threadIdx.x][threadIdx.y];
    }
}

extern "C" {
__host__
int transpose_sum(cudaStream_t stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _transpose_sum<<<blocks, threads, 0, stream>>>(a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
