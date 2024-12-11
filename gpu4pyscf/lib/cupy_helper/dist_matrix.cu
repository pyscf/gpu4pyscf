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
#define THREADS        32

__global__
static void _calc_distances(double *dist, const double *x, const double *y, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n){
        return;
    }

    double dx = x[3*i]   - y[3*j];
    double dy = x[3*i+1] - y[3*j+1];
    double dz = x[3*i+2] - y[3*j+2];
    dist[i*n+j] = norm3d(dx, dy, dz);
}

extern "C" {
int dist_matrix(cudaStream_t stream, double *dist, const double *x, const double *y, int m, int n)
{
    int ntilex = (m + THREADS - 1) / THREADS;
    int ntiley = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntilex, ntiley);
    _calc_distances<<<blocks, threads, 0, stream>>>(dist, x, y, m, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
