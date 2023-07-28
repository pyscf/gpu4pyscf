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
	__shared__ double block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    unsigned int zIndex = blockIdx.z;
    unsigned int off = zIndex * n * n;

	if((xIndex < n) && (yIndex < n))
	{
		unsigned int index_in = yIndex * n + xIndex + off;
		block[threadIdx.y][threadIdx.x] = a[index_in];
	}

    // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < n) && (yIndex < n))
	{
		unsigned int index_out = yIndex * n + xIndex + off;
		a[index_out] += block[threadIdx.x][threadIdx.y];
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
int transpose_sum(double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _transpose_sum<<<blocks, threads>>>(a, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
