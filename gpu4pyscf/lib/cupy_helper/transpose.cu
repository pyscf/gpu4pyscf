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
#include "gsycl/gpu_compat.h"

#define THREADS     16
#define BLOCK_DIM   16

static __global__
void _transpose_dsum(double *a, int n, int counts, int hermi)
{
    setup_context();
    SHARED_ARRAY(double, block, [THREADS][THREADS]);
    if(blockIdx_x > blockIdx_y){
        return;
    }

    int blockx_off = blockIdx_x * BLOCK_DIM;
    int blocky_off = blockIdx_y * BLOCK_DIM;
    size_t x0 = blockx_off + threadIdx_x;
    size_t y0 = blocky_off + threadIdx_y;
    size_t x1 = blocky_off + threadIdx_x;
    size_t y1 = blockx_off + threadIdx_y;
    size_t nn = n * n;
    size_t xy0 = y0 * n + x0;
    size_t xy1 = y1 * n + x1;

    for (int k = 0; k < counts; ++k) {
        double *pa = a + nn * k;
        if (x0 < n && y0 < n){
            block[threadIdx_y][threadIdx_x] = pa[xy0];
        }
        __syncthreads();
        if (x1 < n && y1 < n){
            if (hermi == 1) {
                block[threadIdx_x][threadIdx_y] += pa[xy1];
            } else {
                block[threadIdx_x][threadIdx_y] -= pa[xy1];
            }
        }
        __syncthreads();

        if(x0 < n && y0 < n){
            pa[xy0] = block[threadIdx_y][threadIdx_x];
        }
        if(x1 < n && y1 < n){
            if (hermi == 1) {
                pa[xy1] = block[threadIdx_x][threadIdx_y];
            } else {
                pa[xy1] = -block[threadIdx_x][threadIdx_y];
            }
        }
        __syncthreads();
    }
}

static __global__
void _transpose_zsum(double *a, int n, int counts, int hermi)
{
    setup_context();
    SHARED_ARRAY(double, blockR, [THREADS][THREADS]);
    SHARED_ARRAY(double, blockI, [THREADS][THREADS]);
    if(blockIdx_x > blockIdx_y){
        return;
    }

    int blockx_off = blockIdx_x * BLOCK_DIM;
    int blocky_off = blockIdx_y * BLOCK_DIM;
    size_t x0 = blockx_off + threadIdx_x;
    size_t y0 = blocky_off + threadIdx_y;
    size_t x1 = blocky_off + threadIdx_x;
    size_t y1 = blockx_off + threadIdx_y;
    size_t nn = n * n * 2;
    size_t xy0 = (y0 * n + x0) * 2;
    size_t xy1 = (y1 * n + x1) * 2;

    for (int k = 0; k < counts; ++k) {
        double *pa = a + nn * k;
        if (x0 < n && y0 < n){
            blockR[threadIdx_y][threadIdx_x] = pa[xy0  ];
            blockI[threadIdx_y][threadIdx_x] = pa[xy0+1];
        }
        __syncthreads();
        if (x1 < n && y1 < n){
            if (hermi == 1) {
                blockR[threadIdx_x][threadIdx_y] += pa[xy1  ];
                blockI[threadIdx_x][threadIdx_y] -= pa[xy1+1];
            } else {
                blockR[threadIdx_x][threadIdx_y] -= pa[xy1  ];
                blockI[threadIdx_x][threadIdx_y] += pa[xy1+1];
            }
        }
        __syncthreads();

        if(x0 < n && y0 < n){
            pa[xy0  ] = blockR[threadIdx_y][threadIdx_x];
            pa[xy0+1] = blockI[threadIdx_y][threadIdx_x];
        }
        if(x1 < n && y1 < n){
            if (hermi == 1) {
                pa[xy1  ] =  blockR[threadIdx_x][threadIdx_y];
                pa[xy1+1] = -blockI[threadIdx_x][threadIdx_y];
            } else {
                pa[xy1  ] = -blockR[threadIdx_x][threadIdx_y];
                pa[xy1+1] =  blockI[threadIdx_x][threadIdx_y];
            }
        }
        __syncthreads();
    }
}

extern "C" {
int transpose_dsum(cudaStream_t stream, double *a, int n, int counts, int hermi)
{
    int ntile = (n + THREADS - 1) / THREADS;
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(ntile, ntile);
    LAUNCH_KERNEL(_transpose_dsum, blocks, threads, 0, ON_STREAM(stream),
                  a, n, counts, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int transpose_zsum(cudaStream_t stream, double *a, int n, int counts, int hermi)
{
    int ntile = (n + THREADS - 1) / THREADS;
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(ntile, ntile);
    LAUNCH_KERNEL(_transpose_zsum, blocks, threads, 0, ON_STREAM(stream),
                  a, n, counts, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
