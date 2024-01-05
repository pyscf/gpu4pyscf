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

#include <cuda_runtime.h>

#define THREADS        32
#define BLOCK_DIM   32

__global__
static void _dsymm_triu(double *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < j || i >= n || j >= n) {
        return;
    }
    size_t N = n;
    size_t off = N * N * blockIdx.z;
    a[off + j * N + i] = a[off + i * N + j];
}

__global__
void _transpose_sum(double *a, int n)
{
    if(blockIdx.x > blockIdx.y){
        return;
    }
	__shared__ double block[BLOCK_DIM][BLOCK_DIM+1];

    unsigned int blockx_off = blockIdx.x * BLOCK_DIM;
    unsigned int blocky_off = blockIdx.y * BLOCK_DIM;
	unsigned int x0 = blockx_off + threadIdx.x;
	unsigned int y0 = blocky_off + threadIdx.y;
    unsigned int x1 = blocky_off + threadIdx.x;
	unsigned int y1 = blockx_off + threadIdx.y;
    unsigned int z = blockIdx.z;

    unsigned int off = n * n * z;
    unsigned int xy0 = y0 * n + x0 + off;
    unsigned int xy1 = y1 * n + x1 + off;

    if (x0 < n && y0 < n){
        block[threadIdx.y][threadIdx.x] = a[xy0];
    }
    __syncthreads();
    if (x1 < n && y1 < n){
        block[threadIdx.x][threadIdx.y] += a[xy1];
    }
    __syncthreads();

    if(x0 < n && y0 < n){
        a[xy0] = block[threadIdx.y][threadIdx.x];
    }
    if(x1 < n && y1 < n){
        a[xy1] = block[threadIdx.x][threadIdx.y];
    }
}

extern "C" {
__host__
int CPdsymm_triu(double *a, int n, int counts)
{
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _dsymm_triu<<<blocks, threads>>>(a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

__host__
int transpose_sum(cudaStream_t stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _transpose_sum<<<blocks, threads, 0, stream>>>(a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
