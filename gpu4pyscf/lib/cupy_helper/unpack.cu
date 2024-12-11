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
#include <stdio.h>
#define THREADS       32
#define BDIM 32

__global__
void _unpack_tril(const double *eri_tril, double *eri, int nao){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int p = blockIdx.z;
    int stride = ((nao + 1) * nao) / 2;

    if(i >= nao || j >= nao || i < j){
        return;
    }
    int ptr = j + (i+1)*i/2;
    eri[p*nao*nao + j*nao + i] = eri_tril[ptr + p*stride];
}

__global__
void _unpack_triu(const double *eri_tril, double *eri, int nao){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int p = blockIdx.z;
    int stride = ((nao + 1) * nao) / 2;

    if(i >= nao || j >= nao || i > j){
        return;
    }
    int ptr = i + (j+1)*j/2;

    eri[p*nao*nao + j*nao + i] = eri_tril[ptr + p*stride];
}

__global__
void _unpack_sparse(const double *cderi_sparse, const long *row, const long *col,
                    double *out, int nao, int nij, int stride_sparse, int p0, int p1){
    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

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
int unpack_tril(cudaStream_t stream, const double *eri_tril, double *eri, int nao, int blk_size){
    dim3 threads(THREADS, THREADS);
    int nx = (nao + threads.x - 1) / threads.x;
    int ny = (nao + threads.y - 1) / threads.y;
    dim3 blocks(nx, ny, blk_size);
    _unpack_tril<<<blocks, threads, 0, stream>>>(eri_tril, eri, nao);
    _unpack_triu<<<blocks, threads, 0, stream>>>(eri_tril, eri, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int unpack_sparse(cudaStream_t stream, const double *cderi_sparse, const long *row, const long *col,
                double *eri, int nao, int nij, int naux, int p0, int p1){
    int blockx = (nij + THREADS - 1) / THREADS;
    int blocky = (p1 - p0 + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(blockx, blocky);

    _unpack_sparse<<<blocks, threads, 0, stream>>>(cderi_sparse, row, col, eri, nao, nij, naux, p0, p1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

}
