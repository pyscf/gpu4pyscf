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

#define THREADS        16

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

extern "C" {
__host__
void CPdsymm_triu(double *a, int n, int counts)
{
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile, counts);
    _dsymm_triu<<<blocks, threads>>>(a, n);
}
}
