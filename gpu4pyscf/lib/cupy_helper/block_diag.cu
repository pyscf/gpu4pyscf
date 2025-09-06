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
#define THREADS        16

__global__
static void _block_diag(double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols)
{
    int r = blockIdx.x;

    if (r >= ndiags){
        return;
    }
    int m0 = rows[r+1] - rows[r];
    int n0 = cols[r+1] - cols[r];
    int diag_offset = offsets[r];
    int row_offset = rows[r];
    int col_offset = cols[r];
    
    for (int i = threadIdx.y; i < m0; i += THREADS){
        for (int j = threadIdx.x; j < n0; j += THREADS){
            out[(i+row_offset)*n + (j+col_offset)] = diags[diag_offset + i*n0 + j];
        }
    }
}

extern "C" {
int block_diag(cudaStream_t stream, double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols)
{
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ndiags);
    _block_diag<<<blocks, threads, 0, stream>>>(out, m, n, diags, ndiags, offsets, rows, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in block_diag: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
