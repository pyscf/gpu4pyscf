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
#include <stdio.h>
#define THREADS       32
#define BDIM 32

__global__ static
void _pack_tril(double *a_tril, double *a, size_t n)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    size_t j = item.get_global_id(2);
    size_t i = item.get_global_id(1);
    size_t p = item.get_group(0);
#else
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t p = blockIdx.z;
#endif
    size_t stride = ((n + 1) * n) / 2;

    if (i >= n || j >= n || i < j) {
        return;
    }
    size_t ptr = i*(i+1)/2 + j;
    a_tril[ptr + p*stride] = a[p*n*n + i*n + j];
}

__global__ static
void _unpack_tril(double *eri_tril, double *eri, size_t nao)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    size_t j = item.get_global_id(2);
    size_t i = item.get_global_id(1);
    size_t p = item.get_group(0);
#else
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t p = blockIdx.z;
#endif
    size_t stride = ((nao + 1) * nao) / 2;

    if (i >= nao || j >= nao || i < j) {
        return;
    }
    size_t ptr = i*(i+1)/2 + j;
    eri[p*nao*nao + i*nao + j] = eri_tril[ptr + p*stride];
}

__global__ static
void _fill_triu(double *eri, size_t nao, int hermi)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    int j = item.get_global_id(2);
    int i = item.get_global_id(1);
    size_t p = item.get_group(0);
#else
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t p = blockIdx.z;
#endif
    if (i >= nao || j >= nao || i >= j) {
        return;
    }
    size_t off = p * nao * nao;
    if (hermi == 1) {
        eri[off + i*nao + j] = eri[off + j*nao + i];
    } else if (hermi == 2) {
        eri[off + i*nao + j] = -eri[off + j*nao + i];
    }
}

__global__ static
void _unpack_sparse(const double *cderi_sparse, const long *row, const long *col,
                    double *out, size_t nao, int nij, int stride_sparse, int p0, int p1)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int ij = item.get_global_id(1);
    int k = item.get_global_id(0);
#else
    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
#endif

    int idx_aux = k + p0;
    if (idx_aux >= p1 || ij >= nij){
        return;
    }

    int i = row[ij];
    int j = col[ij];
    double e = cderi_sparse[ij*stride_sparse + idx_aux];
    out[k + i*(p1-p0) + j*(p1-p0)*nao] = e;
    out[k + j*(p1-p0) + i*(p1-p0)*nao] = e;
}

extern "C" {
int fill_triu(cudaStream_t stream, double *a, int n, int counts, int hermi)
{
#ifdef USE_SYCL
    sycl::range<3> threads(1, THREADS, THREADS);
    int nx = (n + threads[2] - 1) / threads[2];
    int ny = (n + threads[1] - 1) / threads[1];
    sycl::range<3> blocks(counts, ny, nx);
    stream.parallel_for<class _fill_triu_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _fill_triu(a, n, hermi);
    });
#else
    dim3 threads(THREADS, THREADS);
    int nx = (n + threads.x - 1) / threads.x;
    int ny = (n + threads.y - 1) / threads.y;
    dim3 blocks(nx, ny, counts);
    _fill_triu<<<blocks, threads, 0, stream>>>(a, n, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
#endif
    return 0;
}

int pack_tril(cudaStream_t stream, double *a_tril, double *a, int n, int counts)
{
#ifdef USE_SYCL
    sycl::range<3> threads(1, THREADS, THREADS);
    int nx = (n + threads[2] - 1) / threads[2];
    int ny = (n + threads[1] - 1) / threads[1];
    sycl::range<3> blocks(counts, ny, nx);
    stream.parallel_for<class _pack_tril_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _pack_tril(a_tril, a, n);
    });
#else
    dim3 threads(THREADS, THREADS);
    int nx = (n + threads.x - 1) / threads.x;
    int ny = (n + threads.y - 1) / threads.y;
    dim3 blocks(nx, ny, counts);
    _pack_tril<<<blocks, threads, 0, stream>>>(a_tril, a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
#endif
    return 0;
}

int unpack_tril(cudaStream_t stream, double *eri_tril, double *eri,
                int nao, int blk_size, int hermi)
{
#ifdef USE_SYCL
    sycl::range<3> threads(1, THREADS, THREADS);
    int nx = (nao + threads[2] - 1) / threads[2];
    int ny = (nao + threads[1] - 1) / threads[1];
    sycl::range<3> blocks(blk_size, ny, nx);
    stream.parallel_for<class _unpack_tril_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _unpack_tril(eri_tril, eri, nao);
    });
    stream.parallel_for<class _fill_triu_sycl2>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      _fill_triu(eri, nao, hermi);
    });
#else
    dim3 threads(THREADS, THREADS);
    int nx = (nao + threads.x - 1) / threads.x;
    int ny = (nao + threads.y - 1) / threads.y;
    dim3 blocks(nx, ny, blk_size);
    _unpack_tril<<<blocks, threads, 0, stream>>>(eri_tril, eri, nao);
    _fill_triu<<<blocks, threads, 0, stream>>>(eri, nao, hermi);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
#endif
    return 0;
}

int unpack_sparse(cudaStream_t stream, const double *cderi_sparse, const long *row, const long *col,
                double *eri, int nao, int nij, int naux, int p0, int p1)
{
    int blockx = (nij + THREADS - 1) / THREADS;
    int blocky = (p1 - p0 + THREADS - 1) / THREADS;

    #ifdef USE_SYCL
    sycl::range<2> threads(THREADS, THREADS);
    sycl::range<2> blocks(blocky, blockx);
    stream.parallel_for<class _unpack_sparse_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      _unpack_sparse(cderi_sparse, row, col, eri, nao, nij, naux, p0, p1);
    });
    #else
    dim3 threads(THREADS, THREADS);
    dim3 blocks(blockx, blocky);

    _unpack_sparse<<<blocks, threads, 0, stream>>>(cderi_sparse, row, col, eri, nao, nij, naux, p0, p1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    #endif
    return 0;
}

}
