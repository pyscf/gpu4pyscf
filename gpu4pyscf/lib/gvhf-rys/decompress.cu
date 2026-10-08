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

#define RBLKSIZE 16
#define CBLKSIZE 64
#define STRIDE   4
#define OF_COMPLEX 2

__global__ static
void write_kernel(double *out, double *inp, size_t ncol, int col0, int col1)
{
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    size_t row = blockIdx.x;
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
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int row_id = blockIdx.x;
    int dcol = col1 - col0;
    out = out + row_id * ncol + col0;
    for (int k = thread_id; k < dcol; k += threads) {
        out[k] = inp[k * nrow + row_id];
    }
}

// Transpose [pair, aux] input into [aux, pair] mapped host storage and add.
// Columns are unique within a batch. The caller serializes overlapping writes
// from separate GPUs; cart2sph and radial recontraction have already run.
__global__ static
void add_columns_kernel(double *out, const double *inp, const int *columns,
                        size_t nrow, size_t ncol, size_t nin, int stride)
{
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         t < nrow*nin*stride; t += (size_t)gridDim.x*blockDim.x) {
        size_t row = t / (nin*stride);
        size_t col = (t / stride) % nin;
        size_t component = t % stride;
        out[(row*ncol + columns[col])*stride + component] +=
            inp[(col*nrow + row)*stride + component];
    }
}

extern "C" {
int add_col_segment(cudaStream_t stream, double *out_cpu, const double *inp,
                    const int *columns, int nrow, int ncol, int nin, int stride)
{
    if (nrow == 0 || nin == 0) {
        return 0;
    }
    double *out_gpu;
    cudaError_t err = cudaHostGetDevicePointer(&out_gpu, out_cpu, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "add_col_segment address mapping error %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    size_t blocks = ((size_t)nrow*nin*stride + 255) / 256;
    if (blocks > 65535) blocks = 65535;
    add_columns_kernel<<<blocks, 256, 0, stream>>>(
        out_gpu, inp, columns, nrow, ncol, nin, stride);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "add_col_segment error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int store_col_segment(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
    double *out_gpu;
    cudaError_t err = cudaHostGetDevicePointer(&out_gpu, out_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "store_col_segment address mapping error %s\n", cudaGetErrorString(err));
        return 1;
    }
    write_kernel<<<nrow, 512>>>(out_gpu, inp, ncol, col0, col1);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "store_col_segment error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int transpose_write(cudaStream_t stream, double *out_cpu, double *inp,
                    int nrow, int ncol, int col0, int col1)
{
    double *out_gpu;
    cudaError_t err = cudaHostGetDevicePointer(&out_gpu, out_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "transpose_write address mapping error %s\n", cudaGetErrorString(err));
        return 1;
    }
    transpose_write_kernel<<<nrow, 512, 0, stream>>>(out_gpu, inp, nrow, ncol, col0, col1);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "transpose_write error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
