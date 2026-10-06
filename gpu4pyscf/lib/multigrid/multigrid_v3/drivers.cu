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
#include <cuda_runtime.h>
#include "constant_objects.cuh"
#include "gsycl/gpu_compat.h"

TABLE_DEFINE(double, s_c_lattice_vectors, c_lattice_vectors, 9);
TABLE_DEFINE(double, s_c_reciprocal_lattice_vectors, c_reciprocal_lattice_vectors, 9);
TABLE_DEFINE(double, s_c_dxyz_dabc, c_dxyz_dabc, 9);



// c_nf/c_div_nf defined in constant_tables.cu (CUDA-only).


// input[nc,nx,ny,nz], output[nc,mx,my,mz]
__global__ static
void fft_take_kernel(double2* __restrict__ out, double2* __restrict__ in,
                     int mx, int my, int mz, int nx, int ny, int nz, int nc)
{
    setup_context();
    int x = blockIdx_x;
    int y = global_y;
    int tx = threadIdx_x;
    int threadsx = blockDim_x;
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
    setup_context();
    int x = blockIdx_x;
    int y = global_y;
    int tx = threadIdx_x;
    int threadsx = blockDim_x;
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
            out[dst] = double2{D2X(out[dst]) + D2X(in[src]),
                               D2Y(out[dst]) + D2Y(in[src])};
        }
    }
}

extern "C" {
void update_lattice_vectors(double *lattice_vectors,
                            double *reciprocal_lattice_vectors)
{
    TABLE_FILL(c_lattice_vectors, s_c_lattice_vectors, lattice_vectors, 9 * sizeof(double));
    TABLE_FILL(c_reciprocal_lattice_vectors, s_c_reciprocal_lattice_vectors, reciprocal_lattice_vectors, 9 * sizeof(double));
}

void update_dxyz_dabc(double *dxyz_dabc) {
    TABLE_FILL(c_dxyz_dabc, s_c_dxyz_dabc, dxyz_dabc, 9 * sizeof(double));
}

int fft_take(double2 *out, double2 *in, int *out_shape, int *in_shape, int counts)
{
    int mx = out_shape[0];
    int my = out_shape[1];
    int mz = out_shape[2];
    int nx = in_shape[0], ny = in_shape[1], nz = in_shape[2];
    auto threads = make_block(32, 16);
    auto grids = make_grid(mx, (my+15)/16);
    LAUNCH_KERNEL( fft_take_kernel, grids, threads, 0,
                    out, in, mx, my, mz, nx, ny, nz, counts);
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
    int nx = out_shape[0], ny = out_shape[1], nz = out_shape[2];
    auto threads = make_block(32, 16);
    auto grids = make_grid(mx, (my+15)/16);
    LAUNCH_KERNEL( fft_takebak_kernel, grids, threads, 0,
                    out, in, mx, my, mz, nx, ny, nz, counts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fft_takebak kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
