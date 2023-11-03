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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "contract_rho.cuh"
// TODO: improve this?
__global__
void GDFTcontract_rho_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    double v = 0;
    if (active){
        for (int ao_id = threadIdx.y; ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            v += bra[ket_idx] * ket[ket_idx];
        }
    }

    __shared__ double buf[BLKSIZEX*(BLKSIZEY+1)];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ixy = ix + BLKSIZEX * iy;
    buf[ixy] = v;   __syncthreads();

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
void GDFTcontract_rho4_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, int count)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;
    size_t ket_stride = nao * ngrids;
    size_t rho_stride = count * ngrids;

    __shared__ double buf[BLKSIZEX*(BLKSIZEY+1)];

    for (int ia = 0; ia < count; ia++){
        double v[4] = {0.0, 0.0, 0.0, 0.0};
        if (active){
            for (int ao_id = threadIdx.y; ao_id < nao; ao_id += BLKSIZEY) {
                int ket_idx = grid_id + ao_id * ngrids;
                double bra_tmp = bra[ket_idx + ia * ket_stride];
                v[0] += bra_tmp * ket[0*ket_stride + ket_idx];
                v[1] += bra_tmp * ket[1*ket_stride + ket_idx];
                v[2] += bra_tmp * ket[2*ket_stride + ket_idx];
                v[3] += bra_tmp * ket[3*ket_stride + ket_idx];
            }
        }

        int ix = threadIdx.x;
        int iy = threadIdx.y;
        int ixy = ix + BLKSIZEX * iy;
        for (int i = 0; i < 4; i++){
            buf[ixy] = v[i];   __syncthreads();
            if (blockDim.y >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; __syncthreads();
            if (blockDim.y >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  __syncthreads();
            if (blockDim.y >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  __syncthreads();
            if (blockDim.y >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  __syncthreads();
            if (blockDim.y >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  __syncthreads();

            if (iy == 0 && active) {
                rho[grid_id + ia * ngrids + rho_stride * i] = buf[ix];
            }
        }
    }
}

__global__
void GDFTcontract_rho_gga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[4] = {0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = threadIdx.y; ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp = bra[ket_idx];
            double ket_tmp = ket[ket_idx];

            v[0] += bra_tmp * ket_tmp;

            ket_idx += ket_stride;
            v[1] += bra_tmp * ket[ket_idx];
            v[1] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[2] += bra_tmp * ket[ket_idx];
            v[2] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[3] += bra_tmp * ket[ket_idx];
            v[3] += ket_tmp * bra[ket_idx];
        }
    }

    __shared__ double buf[BLKSIZEX*(BLKSIZEY+1)];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 4; i++){
        buf[ixy] = v[i];   __syncthreads();
        if (blockDim.y >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; __syncthreads();
        if (blockDim.y >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  __syncthreads();
        if (blockDim.y >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  __syncthreads();
        if (blockDim.y >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  __syncthreads();
        if (blockDim.y >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  __syncthreads();

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = 2.0 * buf[ix];
        }
    }
}


__global__
void GDFTcontract_rho_mgga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = threadIdx.y; ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp0 = bra[ket_idx];
            double ket_tmp0 = ket[ket_idx];

            v[0] += bra_tmp0 * ket_tmp0;

            ket_idx += ket_stride;
            double bra_tmp1 = bra[ket_idx];
            double ket_tmp1 = ket[ket_idx];
            v[1] += bra_tmp0 * ket_tmp1;
            v[1] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[2] += bra_tmp0 * ket_tmp1;
            v[2] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[3] += bra_tmp0 * ket_tmp1;
            v[3] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

        }
    }

    v[4] *= 0.5;

    __shared__ double buf[BLKSIZEX*(BLKSIZEY+1)];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 5; i++){
        buf[ixy] = v[i];   __syncthreads();
        if (blockDim.y >= 32 && iy < 16) buf[ixy] += buf[ixy + BLKSIZEX * 16]; __syncthreads();
        if (blockDim.y >= 16 && iy < 8)  buf[ixy] += buf[ixy + BLKSIZEX * 8];  __syncthreads();
        if (blockDim.y >= 8  && iy < 4)  buf[ixy] += buf[ixy + BLKSIZEX * 4];  __syncthreads();
        if (blockDim.y >= 4  && iy < 2)  buf[ixy] += buf[ixy + BLKSIZEX * 2];  __syncthreads();
        if (blockDim.y >= 2  && iy < 1)  buf[ixy] += buf[ixy + BLKSIZEX * 1];  __syncthreads();

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = 2.0 * buf[ix];
        }
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

__global__
void GDFT_make_dR_dao_w_kernel(double *out, double *ket, double *wv,
                         int ngrids, int nao)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int ao_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;

    double wv0 = wv[grid_id + ngrids * 0];
    double wv1 = wv[grid_id + ngrids * 1];
    double wv2 = wv[grid_id + ngrids * 2];
    double wv3 = wv[grid_id + ngrids * 3];

    double ket5 = ket[ixy + Nag * 5];
    double ket6 = ket[ixy + Nag * 6];
    double val;
    val = ket[ixy + Nag * 1] * wv0;
    val+= ket[ixy + Nag * 4] * wv1;
    val+= ket5 * wv2;
    val+= ket6 * wv3;
    out[ixy + Nag * 0] = val;

    double ket8 = ket[ixy + Nag * 8];
    val = ket[ixy + Nag * 2] * wv0;
    val+= ket5 * wv1;
    val+= ket[ixy + Nag * 7] * wv2;
    val+= ket8 * wv3;
    out[ixy + Nag * 1] = val;

    val = ket[ixy + Nag * 3] * wv0;
    val+= ket6 * wv1;
    val+= ket8 * wv2;
    val+= ket[ixy + Nag * 9] * wv3;
    out[ixy + Nag * 2] = val;
}


extern "C"{
__host__
int GDFTcontract_rho(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX);
    GDFTcontract_rho_kernel<<<blocks, threads, 0, stream>>>(rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho4(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao, int count)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX);
    GDFTcontract_rho4_kernel<<<blocks, threads, 0, stream>>>(rho, bra, ket, ngrids, nao, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho_gga(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX);
    GDFTcontract_rho_gga_kernel<<<blocks, threads, 0, stream>>>(rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho_gga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho_mgga(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX);
    GDFTcontract_rho_mgga_kernel<<<blocks, threads, 0, stream>>>(rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho_mgga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFT_make_dR_dao_w(cudaStream_t stream, double *out, double *ket, double *wv,
                 int ngrids, int nao)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX, (nao+BLKSIZEY-1)/BLKSIZEY);
    GDFT_make_dR_dao_w_kernel<<<blocks, threads, 0, stream>>>(out, ket, wv, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFT_make_dR_dao_w: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTscale_ao(cudaStream_t stream, double *out, double *ket, double *wv,
                 int ngrids, int nao, int nvar)
{
    dim3 threads(BLKSIZEX, BLKSIZEY);
    dim3 blocks((ngrids+BLKSIZEX-1)/BLKSIZEX, (nao+BLKSIZEY-1)/BLKSIZEY);
    GDFTscale_ao_kernel<<<blocks, threads, 0, stream>>>(out, ket, wv, ngrids, nao, nvar);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTscale_ao: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}