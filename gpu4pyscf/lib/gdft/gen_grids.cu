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
#include <cuda_runtime.h>

#define NATOM_PER_BLOCK        128

__global__
void GDFTgen_grid_kernel(double *pbecke, const double *coords, const double *atm_coords, const double *a,
int ngrids, int natm)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;
    double xg = 0.0;
    double yg = 0.0;
    double zg = 0.0;
    if(active){
        xg = coords[3*grid_id+0];
        yg = coords[3*grid_id+1];
        zg = coords[3*grid_id+2];
    }
    __shared__ double xj[NATOM_PER_BLOCK];
    __shared__ double yj[NATOM_PER_BLOCK];
    __shared__ double zj[NATOM_PER_BLOCK];
    __shared__ double a_smem[NATOM_PER_BLOCK];
    __shared__ double dij_smem[NATOM_PER_BLOCK];
    const int tx = threadIdx.x;

    for (int atom_i = 0; atom_i < natm; atom_i++){
        double xi = atm_coords[atom_i];
        double yi = atm_coords[atom_i + natm];
        double zi = atm_coords[atom_i + 2*natm];

        double becke = 2.0;
        double dx, dy, dz, dig;
        if (active){
            // distance between grids and atom i
            dx = xg - xi;
            dy = yg - yi;
            dz = zg - zi;
            dig = norm3d(dx, dy, dz);
        }
        for (int j = 0; j < natm; j+=blockDim.x){
            int atom_idx = j + tx;
            if (atom_idx < natm){
                double xj_t = atm_coords[atom_idx];
                double yj_t = atm_coords[atom_idx + natm];
                double zj_t = atm_coords[atom_idx + 2*natm];

                // distance between atom i and atom j
                dx = xi - xj_t;
                dy = yi - yj_t;
                dz = zi - zj_t;
                double dij = rnorm3d(dx, dy, dz);

                // distance between atom i and atom j
                dij_smem[tx] = dij;
                xj[tx] = xj_t;
                yj[tx] = yj_t;
                zj[tx] = zj_t;
                a_smem[tx] = a[atom_i * natm + atom_idx];
            }
            __syncthreads();

            for (int l = 0, M = min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
                int atom_j = j + l;
                // distance between grids and atom j
                dx = xg - xj[l];
                dy = yg - yj[l];
                dz = zg - zj[l];
                double djg = norm3d(dx, dy, dz);

                double dij = dij_smem[l];
                double aij = a_smem[l];
                double g = (atom_i == atom_j) ? 0.0 : (dig - djg) * dij;

                // atomic radii adjust function
                double g1 = g*g - 1.0;
                //g1 -= 1.0;
                g += g1 * aij;

                // becke scheme
                g = (3.0 - g*g) * g * .5;
                g = (3.0 - g*g) * g * .5;
                g = (3.0 - g*g) * g * .5;

                g = 0.5 * (1.0 - g);
                becke *= g;
            }
            __syncthreads();
        }
        if(active){
            pbecke[atom_i*ngrids + grid_id] = becke;
        }
    }
}

__global__
void GDFTgroup_grids_kernel(int* group_ids, const double* atom_coords, const double* coords, int natm, int ngrids){
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;

    double xg = coords[grid_id];
    double yg = coords[grid_id + ngrids];
    double zg = coords[grid_id + 2*ngrids];

    double r2min = 1e30;
    int idx = 0;
    const int tx = threadIdx.x;
    double __shared__ x_atom[NATOM_PER_BLOCK];
    double __shared__ y_atom[NATOM_PER_BLOCK];
    double __shared__ z_atom[NATOM_PER_BLOCK];
    for (int j = 0; j < natm; j+=blockDim.x){
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
int GDFTgen_grid_partition(cudaStream_t stream, double *pbecke,
const double *coords, const double *atm_coords, const double *a, int ngrids, int natm)
{
    dim3 threads(NATOM_PER_BLOCK);
    dim3 blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    GDFTgen_grid_kernel<<<blocks, threads, 0, stream>>>(pbecke, coords, atm_coords, a, ngrids, natm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error of gen grids: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

__host__
int GDFTgroup_grids(cudaStream_t stream, int* group_ids, const double* atom_coords, const double* coords,
    int natm, int ngrids){
    if (ngrids % NATOM_PER_BLOCK != 0){
        fprintf(stderr, "CUDA Error of gen grids: grids alignment must be %d.", NATOM_PER_BLOCK);
        return 1;
    }
    dim3 threads(NATOM_PER_BLOCK);
    dim3 blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    GDFTgroup_grids_kernel<<<blocks, threads, 0, stream>>>(group_ids, atom_coords, coords, natm, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error of group grids: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}
