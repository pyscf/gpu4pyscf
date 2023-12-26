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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NATOM_PER_BLOCK        128

__global__
void GDFTgen_grid_kernel(double *pbecke, const double *dist_ig, const double *dist_ij,
    const double *a, int ngrids, int natm)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;

    __shared__ double dij_smem[NATOM_PER_BLOCK];
    __shared__ double a_smem[NATOM_PER_BLOCK];
    const int tx = threadIdx.x;

    for (int atom_i = 0; atom_i < natm; atom_i++){
        double becke = 2.0;
        double dig = 0.0;
        if (active){
            // distance between grids and atom i
            dig = dist_ig[atom_i * ngrids + grid_id];
        }
        for (int j = 0; j < natm; j+=blockDim.x){
            int atom_idx = j + tx;
            if (atom_idx < natm){
                // distance between atom i and atom j
                dij_smem[tx] = dist_ij[atom_i * natm + atom_idx];
                a_smem[tx] = a[atom_i * natm + atom_idx];
            }
            __syncthreads();

            for (int l = 0, M = min(NATOM_PER_BLOCK, natm-j); l < M; ++l){
                int atom_j = j + l;
                // distance between grids and atom j
                double djg = 0;
                if (active){
                    djg = dist_ig[atom_j * ngrids + grid_id];
                }
                double dij = dij_smem[l];
                double aij = a_smem[l];
                double g = (atom_i == atom_j) ? 0.0 : (dig - djg) / dij;

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
    const double *dist_ig, const double *dist_ij,
    const double *a, int ngrids, int natm)
{
    dim3 threads(NATOM_PER_BLOCK);
    dim3 blocks((ngrids+NATOM_PER_BLOCK-1)/NATOM_PER_BLOCK);
    GDFTgen_grid_kernel<<<blocks, threads, 0, stream>>>(pbecke, dist_ig, dist_ij, a, ngrids, natm);
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