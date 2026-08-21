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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int thread_id = item.get_local_id(0);
    int threads = item.get_local_range(0);
    size_t row = item.get_group(0);
#else
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    size_t row = blockIdx.x;
#endif
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int thread_id = item.get_local_id(0);
    int threads = item.get_local_range(0);
    int row_id = item.get_group(0);
#else
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int row_id = blockIdx.x;
#endif
    int dcol = col1 - col0;
    out = out + row_id * ncol + col0;
    for (int k = thread_id; k < dcol; k += threads) {
        out[k] = inp[k * nrow + row_id];
    }
}

extern "C" {
int store_col_segment(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
#ifdef USE_SYCL
    // Host USM allocations are directly device-accessible; no address mapping.
    double *out_gpu = out_cpu;
    size_t Ncol = ncol;
    sycl::range<1> threads(512);
    sycl::range<1> blocks(nrow);
    sycl_get_queue()->parallel_for<class decompress_write_kernel_sycl>(
        sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
      write_kernel(out_gpu, inp, Ncol, col0, col1);
    });
#else
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
#endif
    return 0;
}

int transpose_write(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
#ifdef USE_SYCL
    // Host USM allocations are directly device-accessible; no address mapping.
    double *out_gpu = out_cpu;
    size_t Nrow = nrow;
    size_t Ncol = ncol;
    sycl::range<1> threads(512);
    sycl::range<1> blocks(nrow);
    sycl_get_queue()->parallel_for<class decompress_transpose_write_sycl>(
        sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
      transpose_write_kernel(out_gpu, inp, Nrow, Ncol, col0, col1);
    });
#else
    double *out_gpu;
    cudaError_t err = cudaHostGetDevicePointer(&out_gpu, out_cpu, 0);
    if(err != cudaSuccess){
        fprintf(stderr, "transpose_write address mapping error %s\n", cudaGetErrorString(err));
        return 1;
    }
    transpose_write_kernel<<<nrow, 512>>>(out_gpu, inp, nrow, ncol, col0, col1);
    err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "transpose_write error %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
    return 0;
}
}
