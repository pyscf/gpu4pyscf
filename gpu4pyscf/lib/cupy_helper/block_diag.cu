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
#define THREADS        16

__global__
static void _block_diag(double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int r = blockIdx.x;
    
    if (r >= ndiags){
        return;
    }
    int m0 = rows[r+1] - rows[r];
    int n0 = cols[r+1] - cols[r];
    if (i >= m0 || j >= n0) {
        return;
    }
    out[(i+rows[r])*n + (j+cols[r])] = diags[offsets[r] + i*n0 + j];
}

extern "C" {
int block_diag(cudaStream_t stream, double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols)
{
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ndiags);
    _block_diag<<<blocks, threads, 0, stream>>>(out, m, n, diags, ndiags, offsets, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
