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

#define THREADS         256

__global__
void GDFTgen_grid_kernel(double *pbecke, double *coords, double *atm_coords, double *a, int ngrids, int natm)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    
    double xg = coords[3*grid_id+0];
    double yg = coords[3*grid_id+1];
    double zg = coords[3*grid_id+2];

    double dx, dy, dz;
    for (int atom_i = 0; atom_i < natm; atom_i++){
        double xi = atm_coords[3*atom_i + 0];
        double yi = atm_coords[3*atom_i + 1];
        double zi = atm_coords[3*atom_i + 2];

        // distance between grids and atom i
        dx = xg - xi;
        dy = yg - yi;
        dz = zg - zi;
        double dig = norm3d(dx, dy, dz);
        double becke = 2.0;
        for (int atom_j = 0; atom_j < natm; atom_j++){
            double xj = atm_coords[3*atom_j + 0];
            double yj = atm_coords[3*atom_j + 1];
            double zj = atm_coords[3*atom_j + 2];

            // distance between grids and atom j
            dx = xg - xj;
            dy = yg - yj;
            dz = zg - zj;
            double djg = norm3d(dx, dy, dz);
    
            // distance between atom i and atom j
            dx = xi - xj;
            dy = yi - yj;
            dz = zi - zj;
            double dij = norm3d(dx, dy, dz);
            double g = (atom_i == atom_j) ? 0.0 : (dig - djg) / dij;
    
            // atomic radii adjust function
            double g1 = g*g - 1.0;
            //g1 -= 1.0;
            g1 *= a[atom_i * natm + atom_j];
            g += g1;

            // becke scheme
            g = (3.0 - g*g) * g * .5;
            g = (3.0 - g*g) * g * .5;
            g = (3.0 - g*g) * g * .5;

            g = 0.5 * (1.0 - g);
            becke *= g;
        }
        pbecke[atom_i*ngrids + grid_id] = becke;
    }
}

extern "C"{
__host__
int GDFTgen_grid_partition(cudaStream_t stream, double *pbecke, double *coords, double *atm_coords, double *a, int ngrids, int natm)
{
    dim3 threads(THREADS);
    dim3 blocks((ngrids+THREADS-1)/THREADS);
    GDFTgen_grid_kernel<<<blocks, threads, 0, stream>>>(pbecke, coords, atm_coords, a, ngrids, natm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA Error of gen grids: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}