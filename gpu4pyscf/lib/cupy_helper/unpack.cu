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
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gsycl/gpu_compat.h"

#define THREADS         16
#define RBLKSIZE        16
#define CBLKSIZE        64
#define STRIDE          4
#define OF_COMPLEX      2

__global__ static
void _pack_tril(double *a_tril, double *a, size_t n, int counts)
{
    setup_context();
    size_t j = global_x;
    size_t i = global_y;

    if (i >= n || j >= n || i < j) {
        return;
    }
    size_t stride = ((n + 1) * n) / 2;
    size_t ptr = i*(i+1)/2 + j;
    size_t nao2 = n * n;
    for (int p = 0; p < counts; ++p) {
        a_tril[ptr + p*stride] = a[p*nao2 + i*n + j];
    }
}

__global__ static
void _unpack_tril(double *eri_tril, double *eri, size_t nao, int counts)
{
    setup_context();
    size_t j = global_x;
    size_t i = global_y;
    if (i >= nao || j >= nao || i < j) {
        return;
    }
    size_t stride = ((nao + 1) * nao) / 2;
    size_t ptr = i*(i+1)/2 + j;
    size_t nao2 = nao * nao;
    for (int p = 0; p < counts; ++p) {
        eri[p*nao2 + i*nao + j] = eri_tril[ptr + p*stride];
    }
}

__global__ static
void _dfill_triu(double *eri, size_t nao, int counts, int hermi)
{
    setup_context();
    int j = global_x;
    int i = global_y;
    if (i >= nao || j >= nao || i >= j) {
        return;
    }
    size_t nao2 = nao * nao;
    for (int p = 0; p < counts; ++p) {
        size_t off = p * nao2;
        if (hermi == 1) {
            eri[off + i*nao + j] = eri[off + j*nao + i];
        } else if (hermi == 2) {
            eri[off + i*nao + j] = -eri[off + j*nao + i];
        }
    }
}

__global__ static
void _zfill_triu(double *eri, size_t nao, int counts, int hermi)
{
    setup_context();
    int j = global_x;
    int i = global_y;
    if (i >= nao || j >= nao || i >= j) {
        return;
    }
    size_t nao2 = nao * nao * OF_COMPLEX;
    size_t ij = (i * nao + j) * OF_COMPLEX;
    size_t ji = (j * nao + i) * OF_COMPLEX;
    for (int p = 0; p < counts; ++p) {
        size_t off = p * nao2;
        if (hermi == 1) {
            eri[off + ij + 0] =  eri[off + ji + 0];
            eri[off + ij + 1] = -eri[off + ji + 1];
        } else if (hermi == 2) {
            eri[off + ij + 0] = -eri[off + ji + 0];
            eri[off + ij + 1] =  eri[off + ji + 1];
        }
    }
}

__global__ static
void decompress_kernel(double *out, size_t out_stride,
                       double *cderi, int *pair_idx, int npairs, int nao,
                       size_t naux, int aux0, int aux1)
{
    setup_context();
    int thread_id = threadIdx_x;
    int threads = blockDim_x;
    int batch_id = blockIdx_x;
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
                int aux0, int aux1, int fill_triu)
{
    setup_context();
    int bx = blockIdx_x;
    int by = blockIdx_y;
    int thread_id = threadIdx_x;
    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    SHARED_ARRAY(double, buf, [RBLKSIZE][CBLKSIZE+1]);
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
                  int aux0, int aux1)
{
    setup_context();
    int bx = blockIdx_x;
    int by = blockIdx_y;
    int thread_id = threadIdx_x;
    int threads = STRIDE * CBLKSIZE;
    int tx = thread_id % CBLKSIZE;
    int ty = thread_id / CBLKSIZE;
    int aux_start = by * RBLKSIZE;
    int pair_start = bx * CBLKSIZE;
    int daux = aux1 - aux0;
    size_t Npairs = npairs;
    size_t Nao = nao;

    SHARED_ARRAY(double2, buf, [RBLKSIZE][CBLKSIZE+1]);
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

extern "C" {
int fill_triu(cudaStream_t stream, double *a, int n, int counts, int hermi,
              int dtype)
{
    int nx = (n + THREADS - 1) / THREADS;
    int ny = (n + THREADS - 1) / THREADS;
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(nx, ny);
    if (dtype == 1) { // float64
        LAUNCH_KERNEL(_dfill_triu, blocks, threads, 0, stream,
                      a, n, counts, hermi);
    } else {
        LAUNCH_KERNEL(_zfill_triu, blocks, threads, 0, stream,
                      a, n, counts, hermi);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fill_tril error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int pack_tril(cudaStream_t stream, double *a_tril, double *a, int n, int counts)
{
    int nx = (n + THREADS - 1) / THREADS;
    int ny = (n + THREADS - 1) / THREADS;
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(nx, ny);
    LAUNCH_KERNEL(_pack_tril, blocks, threads, 0, stream,
                  a_tril, a, n, counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pack_tril error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int unpack_tril(cudaStream_t stream, double *eri_tril, double *eri,
                int nao, int counts, int hermi)
{
    int nx = (nao + THREADS - 1) / THREADS;
    int ny = (nao + THREADS - 1) / THREADS;
    auto threads = make_block(THREADS, THREADS);
    auto blocks = make_grid(nx, ny);
    LAUNCH_KERNEL(_unpack_tril, blocks, threads, 0, stream,
                  eri_tril, eri, nao, counts);
    LAUNCH_KERNEL(_dfill_triu, blocks, threads, 0, stream,
                  eri, nao, counts, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int decompress_and_fill(cudaStream_t stream, double *out, int out_stride,
                        double *cderi, int *pair_idx, int npairs, int nao,
                        int naux, int aux0, int aux1)
{
    auto threads = make_block(512);
    auto blocks = make_grid((npairs+RBLKSIZE-1)/RBLKSIZE);
    LAUNCH_KERNEL(decompress_kernel, blocks, threads, 0, stream,
                  out, out_stride, cderi, pair_idx, npairs, nao, naux, aux0, aux1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "decompress_and_fill error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int decompress_and_transpose(cudaStream_t stream, double *out, int out_stride,
                             double *cderi, int *pair_idx, int npairs, int nao,
                             int aux0, int aux1, int fill_triu, int on_host)
{
    double *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer((void **)&eri_gpu, (void *)cderi, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "decompress_and_transpose address mapping error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    auto threads = make_block(CBLKSIZE * STRIDE);
    auto blocks = make_grid((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    LAUNCH_KERNEL(d_t_kernel, blocks, threads, 0, stream,
                  out, out_stride, eri_gpu, pair_idx, npairs, nao, aux0, aux1, fill_triu);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int z_decompress_and_transpose(cudaStream_t stream, double2 *out, int out_stride,
                               double2 *cderi, int *pair_idx, int npairs, int nao,
                               int aux0, int aux1, int fill_triu, int on_host)
{
    double2 *eri_gpu = cderi;
    if (on_host) {
        cudaError_t err = cudaHostGetDevicePointer((void **)&eri_gpu, (void *)cderi, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "decompress_and_transpose address mapping error %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    auto threads = make_block(CBLKSIZE * STRIDE);
    auto blocks = make_grid((npairs+CBLKSIZE-1)/CBLKSIZE, (aux1-aux0+RBLKSIZE-1)/RBLKSIZE);
    LAUNCH_KERNEL(z_d_t_kernel, blocks, threads, 0, stream,
                  out, out_stride, eri_gpu, pair_idx, npairs, nao, aux0, aux1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "decompress_and_transpose error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
