/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
#include <stdio.h>
#define THREADS        32

__global__
static void _take(double *a, const double *b, int *indices, int n)
{
    int i = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || k >= n) {
        return;
    }
    
    int j_b = indices[j];
    int k_b = indices[k];
    int off = i * n * n;

    a[off + j * n + k] = b[off + j_b * n + k_b];
}

extern "C" {
int take_last2d(cudaStream_t stream, double *a, const double *b, int *indices, int blk_size, int n)
{
    // reorder j and k in a[i,j,k] with indicies
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, blk_size);
    _take<<<blocks, threads, 0, stream>>>(a, b, indices, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
