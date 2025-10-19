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

#define THREADS     16
#define BLOCK_DIM   16

static __global__
void _transpose_dsum(double *a, int n, int counts)
{
    if(blockIdx.x > blockIdx.y){
        return;
    }
    __shared__ double block[THREADS][THREADS];

    int blockx_off = blockIdx.x * BLOCK_DIM;
    int blocky_off = blockIdx.y * BLOCK_DIM;
    size_t x0 = blockx_off + threadIdx.x;
    size_t y0 = blocky_off + threadIdx.y;
    size_t x1 = blocky_off + threadIdx.x;
    size_t y1 = blockx_off + threadIdx.y;
    size_t nn = n * n;
    size_t xy0 = y0 * n + x0;
    size_t xy1 = y1 * n + x1;

    for (int k = 0; k < counts; ++k) {
        double *pa = a + nn * k;
        if (x0 < n && y0 < n){
            block[threadIdx.y][threadIdx.x] = pa[xy0];
        }
        __syncthreads();
        if (x1 < n && y1 < n){
            block[threadIdx.x][threadIdx.y] += pa[xy1];
        }
        __syncthreads();

        if(x0 < n && y0 < n){
            pa[xy0] = block[threadIdx.y][threadIdx.x];
        }
        if(x1 < n && y1 < n){
            pa[xy1] = block[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }
}

static __global__
void _transpose_zsum(double *a, int n, int counts)
{
    if(blockIdx.x > blockIdx.y){
        return;
    }
    __shared__ double blockR[THREADS][THREADS];
    __shared__ double blockI[THREADS][THREADS];

    int blockx_off = blockIdx.x * BLOCK_DIM;
    int blocky_off = blockIdx.y * BLOCK_DIM;
    size_t x0 = blockx_off + threadIdx.x;
    size_t y0 = blocky_off + threadIdx.y;
    size_t x1 = blocky_off + threadIdx.x;
    size_t y1 = blockx_off + threadIdx.y;
    size_t nn = n * n * 2;
    size_t xy0 = (y0 * n + x0) * 2;
    size_t xy1 = (y1 * n + x1) * 2;

    for (int k = 0; k < counts; ++k) {
        double *pa = a + nn * k;
        if (x0 < n && y0 < n){
            blockR[threadIdx.y][threadIdx.x] = pa[xy0  ];
            blockI[threadIdx.y][threadIdx.x] = pa[xy0+1];
        }
        __syncthreads();
        if (x1 < n && y1 < n){
            blockR[threadIdx.x][threadIdx.y] += pa[xy1  ];
            blockI[threadIdx.x][threadIdx.y] -= pa[xy1+1];
        }
        __syncthreads();

        if(x0 < n && y0 < n){
            pa[xy0  ] = blockR[threadIdx.y][threadIdx.x];
            pa[xy0+1] = blockI[threadIdx.y][threadIdx.x];
        }
        if(x1 < n && y1 < n){
            pa[xy1  ] =  blockR[threadIdx.x][threadIdx.y];
            pa[xy1+1] = -blockI[threadIdx.x][threadIdx.y];
        }
        __syncthreads();
    }
}

extern "C" {
int transpose_dsum(cudaStream_t stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _transpose_dsum<<<blocks, threads, 0, stream>>>(a, n, counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int transpose_zsum(cudaStream_t stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _transpose_zsum<<<blocks, threads, 0, stream>>>(a, n, counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
