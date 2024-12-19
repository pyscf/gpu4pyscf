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

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _add_sparse(double *a, double *b, int *indices, int n, int m, int count)
{
	int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (row >= m || col >= m){
        return;
    }
    int idx_a = indices[row] * n + indices[col];
    int idx_b = row * m + col;
    for (int i = 0; i < count; i++){
        a[idx_a + i*n*n] += b[idx_b + i*m*m];
    }
}

extern "C" {
__host__
int add_sparse(cudaStream_t stream, double *a, double *b, int *indices, int n, int m, int count){
    int ntile = (m + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _add_sparse<<<blocks, threads, 0, stream>>>(a, b, indices, n, m, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
