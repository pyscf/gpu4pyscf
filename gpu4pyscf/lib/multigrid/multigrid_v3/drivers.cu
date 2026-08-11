/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <gvhf-rys/rys_constant.cu>

__constant__ double c_lattice_vectors[9];
__constant__ double c_reciprocal_lattice_vectors[9];
__constant__ double c_dxyz_dabc[9];

__constant__ int c_nf[] = {
    1,
    3,
    6,
    10,
    15,
    21,
    28,
    36,
    45,
};

__constant__ float c_div_nf[] = {
    1.f,
    0.333334f,
    0.166667f,
    0.100001f,
    0.066667f,
    0.047620f,
    0.035715f,
    0.027778f,
    0.022223f,
};

// input[nc,nx,ny,nz], output[nc,mx,my,mz]
__global__ static
void fft_take_kernel(double2* __restrict__ out, double2* __restrict__ in,
                     int mx, int my, int mz, int nx, int ny, int nz, int nc)
{
    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int threadsx = blockDim.x;
    if (x >= mx || y >= my) return;

    int sx = x;
    int sy = y;
    // fftfreq indexing
    if (x > mx/2) sx = nx + x - mx;
    if (y > my/2) sy = ny + y - my;
    for (int z = tx; z < mz; z += threadsx) {
        int sz = z;
        if (z > mz/2) sz = nz + z - mz;

        for (int c = 0; c < nc; ++c) {
            size_t src = (((size_t)c*nx + sx)*ny + sy)*nz + sz;
            size_t dst = (((size_t)c*mx + x )*my + y )*mz + z;
            out[dst] = in[src];
        }
    }
}

// output[nc,nx,ny,nz], input[nc,mx,my,mz]
__global__ static
void fft_takebak_kernel(double2* __restrict__ out, double2* __restrict__ in,
                        int mx, int my, int mz, int nx, int ny, int nz, int nc)
{
    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int threadsx = blockDim.x;
    if (x >= mx || y >= my) return;

    int sx = x;
    int sy = y;
    // fftfreq indexing
    if (x > mx/2) sx = nx + x - mx;
    if (y > my/2) sy = ny + y - my;
    for (int z = tx; z < mz; z += threadsx) {
        int sz = z;
        if (z > mz/2) sz = nz + z - mz;

        for (int c = 0; c < nc; ++c) {
            size_t dst = (((size_t)c*nx + sx)*ny + sy)*nz + sz;
            size_t src = (((size_t)c*mx + x )*my + y )*mz + z;
            out[dst].x += in[src].x;
            out[dst].y += in[src].y;
        }
    }
}

extern "C" {
void update_lattice_vectors(double *lattice_vectors,
                            double *reciprocal_lattice_vectors,
                            double *reciprocal_norm)
{
    cudaMemcpyToSymbol(c_lattice_vectors, lattice_vectors, 9 * sizeof(double));
    cudaMemcpyToSymbol(c_reciprocal_lattice_vectors, reciprocal_lattice_vectors, 9 * sizeof(double));
}

void update_dxyz_dabc(double *dxyz_dabc) {
    cudaMemcpyToSymbol(c_dxyz_dabc, dxyz_dabc, 9 * sizeof(double));
}

int fft_take(double2 *out, double2 *in, int *out_shape, int *in_shape, int counts)
{
    int mx = out_shape[0];
    int my = out_shape[1];
    int mz = out_shape[2];
    dim3 threads(32, 16);
    dim3 grids(mx, (my+15)/16);
    fft_take_kernel<<<grids, threads>>>(
        out, in, mx, my, mz, in_shape[0], in_shape[1], in_shape[2], counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fft_take kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int fft_takebak(double2 *out, double2 *in, int *out_shape, int *in_shape, int counts)
{
    int mx = in_shape[0];
    int my = in_shape[1];
    int mz = in_shape[2];
    dim3 threads(32, 16);
    dim3 grids(mx, (my+15)/16);
    fft_takebak_kernel<<<grids, threads>>>(
        out, in, mx, my, mz, out_shape[0], out_shape[1], out_shape[2], counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fft_takebak kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
