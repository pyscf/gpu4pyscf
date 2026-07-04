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
void decompress_kernel(double *out, size_t out_stride,
                       double *cderi, int *pair_idx, int npairs, int nao,
                       size_t naux, int aux0, int aux1)
{
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int batch_id = blockIdx.x;
    int dcol = aux1 - aux0;
    int pair0 = batch_id * RBLKSIZE;
    int pair1 = min(pair0 + RBLKSIZE, npairs);
    for (int pair_id = pair0; pair_id < pair1; ++pair_id) {
        int ij = pair_idx[pair_id];
        int i = ij / nao;
        int j = ij - nao * i;
        double *inp = cderi + pair_id * naux + aux0;
        double *out_ij = out + ij * out_stride;
        double *out_ji = out + (j * nao + i) * out_stride;
        for (int k = thread_id; k < dcol; k += threads) {
            double s = inp[k];
            out_ij[k] = s;
            if (i != j) {
                out_ji[k] = s;
            }
        }
    }
}

__global__ static
void d_t_kernel(double *out, size_t out_stride,
                double *cderi, int *pair_idx, int npairs, int nao,
                int naux, int aux0, int aux1, int fill_triu)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_id = threadIdx.x;
    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    __shared__ double buf[RBLKSIZE][CBLKSIZE+1];
    if (pair_start+tx < npairs) {
        for (int k = ty; k < min(RBLKSIZE, daux-aux_start); k += STRIDE) {
            buf[k][tx] = cderi[(aux_start+k)*Npairs+pair_start+tx];
        }
    }
    __syncthreads();
    int stride = threads / RBLKSIZE;
    int pair_id = thread_id / RBLKSIZE;
    int aux_id = thread_id % RBLKSIZE;
    if (aux_start+aux_id < daux) {
        for (int k = pair_id; k < min(CBLKSIZE, npairs-pair_start); k += stride) {
            int pair_ij = pair_idx[pair_start+k];
            int i = pair_ij / nao;
            int j = pair_ij - nao * i;
            double s = buf[aux_id][k];
            out[(i*Nao+j)*out_stride+aux_start+aux_id] = s;
            if (fill_triu && i != j) {
                out[(j*Nao+i)*out_stride+aux_start+aux_id] = s;
            }
        }
    }
}

__global__ static
void z_d_t_kernel(double2 *out, size_t out_stride,
                  double2 *cderi, int *pair_idx, int npairs, int nao,
                  int naux, int aux0, int aux1)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_id = threadIdx.x;
    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    __shared__ double2 buf[RBLKSIZE][CBLKSIZE+1];
    if (pair_start+tx < npairs) {
        for (int k = ty; k < min(RBLKSIZE, daux-aux_start); k += STRIDE) {
            buf[k][tx] = cderi[(aux_start+k)*Npairs+pair_start+tx];
        }
    }
    __syncthreads();
    int stride = threads / RBLKSIZE;
    int pair_id = thread_id / RBLKSIZE;
    int aux_id = thread_id % RBLKSIZE;
    if (aux_start+aux_id < daux) {
        for (int k = pair_id; k < min(CBLKSIZE, npairs-pair_start); k += stride) {
            int pair_ij = pair_idx[pair_start+k];
            int i = pair_ij / nao;
            int j = pair_ij - nao * i;
            double2 s = buf[aux_id][k];
            out[(i*Nao+j)*out_stride+aux_start+aux_id] = s;
        }
    }
}

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

extern "C" {
int decompress_and_fill(double *out, int out_stride,
                        double *cderi, int *pair_idx, int npairs, int nao,
                        int naux, int aux0, int aux1)
{
    dim3 blocks((npairs+RBLKSIZE-1)/RBLKSIZE);
    decompress_kernel<<<blocks, 512>>>(
            out, out_stride, cderi, pair_idx, npairs, nao, naux, aux0, aux1);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "decompress_and_fill error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int decompress_and_transpose(double *out, int out_stride,
                             double *cderi, int *pair_idx, int npairs, int nao,
                             int naux, int aux0, int aux1, int fill_triu, int on_host)
{
    double *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, cderi, 0);
        if(err != cudaSuccess){
            fprintf(stderr, "decompress_and_transpose address mapping error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    dim3 threads(CBLKSIZE * STRIDE);
    dim3 blocks((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    d_t_kernel<<<blocks, threads>>>(
            out, out_stride, eri_gpu, pair_idx, npairs, nao, naux, aux0, aux1, fill_triu);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int z_decompress_and_transpose(double2 *out, int out_stride,
                               double2 *cderi, int *pair_idx, int npairs, int nao,
                               int naux, int aux0, int aux1, int fill_triu, int on_host)
{
    double2 *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer(&eri_gpu, cderi, 0);
        if(err != cudaSuccess){
            fprintf(stderr, "decompress_and_transpose address mapping error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    dim3 threads(CBLKSIZE * STRIDE);
    dim3 blocks((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    z_d_t_kernel<<<blocks, threads>>>(
            out, out_stride, eri_gpu, pair_idx, npairs, nao, naux, aux0, aux1);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
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

int transpose_write(double *out_cpu, double *inp, int nrow, int ncol, int col0, int col1)
{
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
    return 0;
}
}
