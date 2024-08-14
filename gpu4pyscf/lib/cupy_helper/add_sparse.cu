/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _add_sparse(double *a, const double *b, const int *indices, int n, int m, int k, int count)
{
	int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (row >= k || col >= k){
        return;
    }
    int idx_a = indices[row] * n + indices[col];
    int idx_b = row * m + col;
    for (int i = 0; i < count; i++){
        //a[idx_a + i*n*n] += b[idx_b + i*m*m];
        atomicAdd(a+idx_a+i*n*n, b[idx_b+i*m*m]);
    }
}

__global__
void _reduce_sparse(double *a, const double *b, const int *indices, int n, int m, int k, int count)
{
	int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (row >= k || col >= k){
        return;
    }

    for (int i = 0; i < count; i++){
        int idx_a = indices[row + i * k] * n + indices[col + i * k];
        int idx_b = row * m + col;
        //a[idx_a + i*n*n] += b[idx_b + i*m*m];
        atomicAdd(a+idx_a, b[idx_b+i*m*m]);
    }
}

extern "C" {
__host__
int add_sparse(cudaStream_t stream, double *a, const double *b, 
                const int *indices, int n, int m, int k, int count){
    int ntile = (k + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _add_sparse<<<blocks, threads, 0, stream>>>(a, b, indices, n, m, k, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

__host__
int reduce_sparse(cudaStream_t stream, double *a, const double *b, 
                const int *indices, int n, int m, int k, int count){
    int ntile = (k + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _reduce_sparse<<<blocks, threads, 0, stream>>>(a, b, indices, n, m, k, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
