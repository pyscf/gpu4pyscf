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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gsycl/gpu_compat.h"

#define RBLKSIZE 16
#define CBLKSIZE 64
#define STRIDE   4
#define OF_COMPLEX 2

__global__ static
void write_kernel(double *out, double *inp, size_t ncol, int col0, int col1)
{
    setup_context();
    int thread_id = threadIdx_x;
    int threads = blockDim_x;
    size_t row = blockIdx_x;
    int dcol = col1 - col0;
    out += row * ncol + col0;
    inp += row * dcol;
    for (int k = thread_id; k < dcol; k += threads) {
        out[k] = inp[k];
    }
}

__global__ static
void transpose_write_kernel(double *out, double *inp, size_t nrow, size_t ncol,
                            int col0, int col1)
{
    setup_context();
    int thread_id = threadIdx_x;
    int threads = blockDim_x;
    int row_id = blockIdx_x;
    int dcol = col1 - col0;
    out = out + row_id * ncol + col0;
    for (int k = thread_id; k < dcol; k += threads) {
        out[k] = inp[k * nrow + row_id];
    }
}

extern "C" {
int store_col_segment(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
    MAP_PINNED_PTR(double, out_gpu, out_cpu);
    auto blocks = make_grid(nrow);
    auto threads = make_block(512);
    LAUNCH_KERNEL( write_kernel, blocks, threads, 0,
                    out_gpu, inp, (size_t)ncol, col0, col1);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "store_col_segment error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int transpose_write(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
    MAP_PINNED_PTR(double, out_gpu, out_cpu);
    auto blocks = make_grid(nrow);
    auto threads = make_block(512);
    LAUNCH_KERNEL( transpose_write_kernel, blocks, threads, 0,
                    out_gpu, inp, (size_t)nrow, (size_t)ncol, col0, col1);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "transpose_write error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
