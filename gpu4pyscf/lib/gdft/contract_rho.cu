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

#include "contract_rho.cuh"
// TODO: improve this?
__global__
void GDFTcontract_rho_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    int ao_id;
    double v = 0;
    if (active){
        for (ao_id = threadIdx.y; ao_id < nao; ao_id += BLKSIZEY) {
            v += bra[grid_id + ao_id * Ngrids] * ket[grid_id + ao_id * Ngrids];
        }
    }
    
    __shared__ double buf[BLKSIZEX*(BLKSIZEY+1)];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ixy = ix + BLKSIZEX * iy;
    buf[ixy] = v;   __syncthreads();
    // assume block dim = 32 x 32
    if (blockDim.y >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; __syncthreads();
    if (blockDim.y >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  __syncthreads();
    if (blockDim.y >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  __syncthreads();
    if (blockDim.y >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  __syncthreads();
    if (blockDim.y >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  __syncthreads();

    if (iy == 0 && active) {
        rho[grid_id] = buf[ix];
    }
}

__global__
void GDFTscale_ao_kernel(double *out, double *ket, double *wv,
                         int ngrids, int nao, int nvar)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int ao_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;
    double val = 0;
    int n;
    for (n = 0; n < nvar; ++n) {
         val += ket[ixy + Nag * n] * wv[grid_id + ngrids * n];
    }
    out[ixy] = val;
}
