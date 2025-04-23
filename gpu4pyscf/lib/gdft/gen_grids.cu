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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else // USE_SYCL
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_SYCL


#define NATOM_PER_BLOCK        128
#define TILE    16

__global__
void GDFTgrid_weight_kernel(double *weight, double *coords, double *atm_coords, double *a,
                            int *atm_idx, int ngrids, int natm
			    #ifdef USE_SYCL
			    , sycl::nd_item<2> &item
			    #endif
    )
{
#ifdef USE_SYCL
    sycl::group thread_block = item.get_group();
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int blockIdx_x = thread_block.get_group_id(1);
    using tile_t = double[TILE];
    tile_t& atom_xi = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& atom_yi = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& atom_zi = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& atom_xj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& atom_yj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& atom_zj = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    using tile_tt = double[TILE*TILE];
    tile_tt& a_smem = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_tt>(thread_block);
    tile_tt& dij_smem = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_tt>(thread_block);
#else
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockIdx_x = blockIdx.x;
    __shared__ double atom_xi[TILE];
    __shared__ double atom_yi[TILE];
    __shared__ double atom_zi[TILE];
    __shared__ double atom_xj[TILE];
    __shared__ double atom_yj[TILE];
    __shared__ double atom_zj[TILE];
    __shared__ double a_smem[TILE*TILE];
    __shared__ double dij_smem[TILE*TILE];
#endif

    int thread_id = ty * TILE + tx;
    int grid_id = blockIdx_x * TILE*TILE + thread_id;
    double xg = 0.0;
    double yg = 0.0;
    double zg = 0.0;
    int atom_id = natm;
    if (grid_id < ngrids) {
        xg = coords[0*ngrids+grid_id];
        yg = coords[1*ngrids+grid_id];
        zg = coords[2*ngrids+grid_id];
        atom_id = atm_idx[grid_id];
    }
    double *atm_x = atm_coords;
    double *atm_y = atm_x + natm;
    double *atm_z = atm_y + natm;

    double becke_self = 0.;
    double becke_sum = 0.;
    for (int atom_i0 = 0; atom_i0 < natm; atom_i0 += TILE) {
        int i1 = min(natm-atom_i0, TILE);
        __syncthreads();
        if (ty == 0 && atom_i0 + tx < natm) {
            int atom_i = atom_i0 + tx;
            atom_xi[tx] = atm_x[atom_i];
            atom_yi[tx] = atm_y[atom_i];
            atom_zi[tx] = atm_z[atom_i];
        }
        double becke[TILE];
#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            becke[i] = 2.;
        }
        for (int atom_j0 = 0; atom_j0 < natm; atom_j0 += TILE) {
            __syncthreads();
            int atom_i = atom_i0 + ty;
            int atom_j = atom_j0 + tx;
            if (atom_i >= natm) atom_i = 0;
            if (atom_j >= natm) atom_j = 0;
            double xi = atom_xi[ty];
            double yi = atom_yi[ty];
            double zi = atom_zi[ty];
            double xj = atm_x[atom_j];
            double yj = atm_y[atom_j];
            double zj = atm_z[atom_j];
            // distance between atom i and atom j
            double dij_inv = rnorm3d(xi-xj, yi-yj, zi-zj);
            a_smem[thread_id] = a[atom_i * natm + atom_j];
            dij_smem[thread_id] = dij_inv;
            if (ty == 0) {
                atom_xj[tx] = xj;
                atom_yj[tx] = yj;
                atom_zj[tx] = zj;
            }
            __syncthreads();

            int j1 = min(natm-atom_j0, TILE);
            double djg[TILE];
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= j1) {
                    break;
                }
                double dx = xg - atom_xj[j];
                double dy = yg - atom_yj[j];
                double dz = zg - atom_zj[j];
                djg[j] = norm3d(dx, dy, dz);
            }

#pragma unroll
            for (int i = 0; i < TILE; ++i) {
                if (i >= i1) {
                    break;
                }
                double becke_i = becke[i];
                double dx = xg - atom_xi[i];
                double dy = yg - atom_yi[i];
                double dz = zg - atom_zi[i];
                double dig = norm3d(dx, dy, dz);
#pragma unroll
                for (int j = 0; j < TILE; ++j) {
                    if (j >= j1) {
                        break;
                    }
                    double dij = dij_smem[i*TILE+j];
                    double aij = a_smem[i*TILE+j];
                    double g = 0.;
                    if (atom_i0+i != atom_j0+j) {
                        g = (dig - djg[j]) * dij;
                    }

                    // atomic radii adjust function
                    double g1 = g*g - 1.0;
                    g += g1 * aij;

                    // becke scheme
                    g = (3.0 - g*g) * g * .5;
                    g = (3.0 - g*g) * g * .5;
                    g = (3.0 - g*g) * g * .5;

                    becke_i *= 0.5 * (1.0 - g);
                }
                becke[i] = becke_i;
            }
        }
        if (grid_id < ngrids) {
#pragma unroll
            for (int i = 0; i < TILE; ++i) {
                if (i >= i1) {
                    break;
                }
                becke_sum += becke[i];
                if (atom_i0+i == atom_id) {
                    becke_self = becke[i];
                }
            }
        }
    }
    if (grid_id < ngrids) {
        weight[grid_id] *= becke_self / becke_sum;
    }
}

__global__
void GDFTgroup_grids_kernel(int* group_ids, const double* atom_coords, const double* coords, int natm, int ngrids
#ifdef USE_SYCL
			    , sycl::nd_item<2> &item
#endif
    )
{
#ifdef USE_SYCL
    const int grid_id = item.get_global_id(1);
    const int tx = item.get_local_id(1);
    sycl::group thread_block = item.get_group();
    const int blockDim_x = item.get_group_range(1);
    using tile_t = double[NATOM_PER_BLOCK];
    tile_t& x_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& y_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
    tile_t& z_atom = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = threadIdx_x;
    const int blockDim_x = blockDim.x;
    double __shared__ x_atom[NATOM_PER_BLOCK];
    double __shared__ y_atom[NATOM_PER_BLOCK];
    double __shared__ z_atom[NATOM_PER_BLOCK];
#endif

    double xg = coords[grid_id];
    double yg = coords[grid_id + ngrids];
    double zg = coords[grid_id + 2*ngrids];

    double r2min = 1e30;
    int idx = 0;
    for (int j = 0; j < natm; j+=blockDim_x){
        int atom_idx = j + tx;
        if (atom_idx < natm){
            // distance between atom i and atom j
            x_atom[tx] = atom_coords[atom_idx];
            y_atom[tx] = atom_coords[atom_idx + natm];
            z_atom[tx] = atom_coords[atom_idx + 2*natm];
        }
        __syncthreads();

        for (int l = 0, M = min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
            int atom_j = j + l;
            double xa = x_atom[l] - xg;
            double ya = y_atom[l] - yg;
            double za = z_atom[l] - zg;
            double r2 = xa*xa + ya*ya + za*za;
            if (r2 < r2min){
                r2min = r2;
                idx = atom_j;
            }
        }
    }
    group_ids[grid_id] = idx;
}

extern "C"{
__host__
int GDFTbecke_partition_weights(double *weights, double *coords, double *atm_coords,
                                double *a, int *atm_idx, int ngrids, int natm)
{
#ifdef USE_SYCL
    sycl::range<2> threads(TILE, TILE);
    sycl::range<2> blocks(1, (ngrids+TILE*TILE-1)/(TILE*TILE));
    sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
	GDFTgrid_weight_kernel(weights, coords, atm_coords, a, atm_idx, ngrids, natm, item);
    });
#else
    dim3 threads(TILE, TILE);
    int blocks = (ngrids+TILE*TILE-1)/(TILE*TILE);
    GDFTgrid_weight_kernel<<<blocks, threads>>>(weights, coords, atm_coords, a,
                                                atm_idx, ngrids, natm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error in GDFTgrid_weight: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
    return 0;
}

__host__
int GDFTgroup_grids(cudaStream_t stream, int* group_ids, const double* atom_coords, const double* coords,
    int natm, int ngrids){
    if (ngrids % NATOM_PER_BLOCK != 0){
        fprintf(stderr, "SYCL/CUDA Error of gen grids: grids alignment must be %d.", NATOM_PER_BLOCK);
        return 1;
    }
#ifdef USE_SYCL
    sycl::range<2> threads(1, NATOM_PER_BLOCK);
    sycl::range<2> blocks(1, (ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
	GDFTgroup_grids_kernel(group_ids, atom_coords, coords, natm, ngrids, item);
    });
#else
    dim3 threads(NATOM_PER_BLOCK);
    dim3 blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    GDFTgroup_grids_kernel<<<blocks, threads, 0, stream>>>(group_ids, atom_coords, coords, natm, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error of group grids: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
    return 0;
}

}
