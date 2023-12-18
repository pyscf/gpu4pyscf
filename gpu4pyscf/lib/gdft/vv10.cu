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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gint/gint.h"
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define NG_PER_BLOCK      128
#define NG_PER_THREADS    1

__global__
static void vv10_kernel(double *Fvec, double *Uvec, double *Wvec,
    const double *vvcoords, const double *coords,
    const double *W0p, const double *W0, const double *K,
    const double *Kp, const double *RpW,
    int vvngrids, int ngrids)
{
    // grid id
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;
    double xi, yi, zi;
    double W0i, Ki;
    if (active){
        xi = coords[grid_id];
        yi = coords[ngrids + grid_id];
        zi = coords[2*ngrids + grid_id];
        W0i = W0[grid_id];
        Ki = K[grid_id];
    }

    double F = 0.0;
    double U = 0.0;
    double W = 0.0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    //__shared__ double xj_smem[NG_PER_BLOCK];
    //__shared__ double yj_smem[NG_PER_BLOCK];
    //__shared__ double zj_smem[NG_PER_BLOCK];
    //__shared__ double Kp_smem[NG_PER_BLOCK];
    //__shared__ double W0p_smem[NG_PER_BLOCK];
    //__shared__ double RpW_smem[NG_PER_BLOCK];

    __shared__ double3 xj_t[NG_PER_BLOCK];
    __shared__ double3 kp_t[NG_PER_BLOCK];

    const int tx = threadIdx.x;

    for (int j = 0; j < vvngrids; j+=blockDim.x) {
        int idx = j + tx;
        if (idx < vvngrids){
            //xj_smem[tx] = xj[idx];
            //yj_smem[tx] = yj[idx];
            //zj_smem[tx] = zj[idx];
            //Kp_smem[tx] = Kp[idx];
            //W0p_smem[tx] = W0p[idx];
            //RpW_smem[tx] = RpW[idx];

            xj_t[tx] = {xj[idx], yj[idx], zj[idx]};
            kp_t[tx] = {Kp[idx], W0p[idx], RpW[idx]};
        }
        __syncthreads();

        for (int l = 0, M = min(NG_PER_BLOCK, vvngrids - j); l < M; ++l){
            // about 24 operations for each pair
            //double DX = xj_smem[l] - xi;//xj_tmp.x - xi;
            //double DY = yj_smem[l] - yi;//xj_tmp.y - yi;
            //double DZ = zj_smem[l] - zi;//xj_tmp.z - zi;

            double3 xj_tmp = xj_t[l];
            double DX = xj_tmp.x - xi;
            double DY = xj_tmp.y - yi;
            double DZ = xj_tmp.z - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            double3 kp_tmp = kp_t[l]; // (Kpj, W0pj, RpWj)
            double gp = R2*kp_tmp.y + kp_tmp.x;
            //double gp = R2 * W0p_smem[l] + Kp_smem[l];//R2*kp_tmp.y + kp_tmp.x;
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggt = g*gt;
            double g_gt = g + gt;
            //double T = RpW_smem[l] / (gp*ggt*ggt);//kp_tmp.z / (gp*ggt*ggt);
            double T = kp_tmp.z / (gp*ggt*ggt);

            F += T * ggt;
            U += T * g_gt;
            W += T * R2 * g_gt;
            /*
            double ggt = g * gt;
            double ggt2 = ggt * ggt;
            double T = kp_tmp.z/(gp*ggt2);

            F += T * ggt;
            T *= (g + gt);
            U += T;
            W += T * R2;
            */
        }
        __syncthreads();
    }
    if(active){
        Fvec[grid_id] = F * -1.5;
        Uvec[grid_id] = U;
        Wvec[grid_id] = W;
    }

}

__global__
static void vv10_grad_kernel(double *Fvec, const double *vvcoords, const double *coords,
    const double *W0p, const double *W0,
    const double *K, const double *Kp, const double *RpW,
    int vvngrids, int ngrids)
{
    // grid id
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = grid_id < ngrids;
    double xi, yi, zi;
    double W0i, Ki;
    if (active){
        xi = coords[grid_id];
        yi = coords[ngrids + grid_id];
        zi = coords[2*ngrids + grid_id];
        W0i = W0[grid_id];
        Ki = K[grid_id];
    }
    double FX = 0;
    double FY = 0;
    double FZ = 0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    __shared__ double3 xj_t[NG_PER_BLOCK];
    __shared__ double3 kp_t[NG_PER_BLOCK];

    const int tx = threadIdx.x;
    for (int j = 0; j < vvngrids; j+=blockDim.x) {
        int idx = j + threadIdx.x;
        if (idx < vvngrids){
            xj_t[tx] = {xj[idx], yj[idx], zj[idx]};
            kp_t[tx] = {Kp[idx], W0p[idx], RpW[idx]};
        }
        __syncthreads();
        for (int l = 0, M = min(NG_PER_BLOCK, vvngrids - j); l < M; ++l){
            double3 xj_tmp = xj_t[l];
            double pjx = xj_tmp.x;
            double pjy = xj_tmp.y;
            double pjz = xj_tmp.z;

            // about 23 operations for each pair
            double DX = pjx - xi;
            double DY = pjy - yi;
            double DZ = pjz - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            double3 kp_tmp = kp_t[l];
            double Kpj = kp_tmp.x;
            double W0pj = kp_tmp.y;
            double RpWj = kp_tmp.z;

            double gp = R2*W0pj + Kpj;
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggp = g * gp;
            double ggt_gp = gt * ggp;
            double T = RpWj / (ggt_gp * ggt_gp);
            double Q = T * ((W0i*gp + W0pj*g)*gt + (W0i+W0pj)*ggp);

            FX += Q * DX;
            FY += Q * DY;
            FZ += Q * DZ;
        }
         __syncthreads();
    }
    if (active) {
        Fvec[0*ngrids + grid_id] = FX * -3;
        Fvec[1*ngrids + grid_id] = FY * -3;
        Fvec[2*ngrids + grid_id] = FZ * -3;
    }
}

extern "C" {
__host__
int VXC_vv10nlc(cudaStream_t stream, double *Fvec, double *Uvec, double *Wvec,
                 const double *vvcoords, const double *coords,
                 const double *W0p, const double *W0, const double *K,
                 const double *Kp, const double *RpW,
                 int vvngrids, int ngrids)
{
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids/NG_PER_THREADS+1+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_kernel<<<blocks, threads, 0, stream>>>(Fvec, Uvec, Wvec,
                 vvcoords, coords,
                 W0p, W0, K, Kp, RpW, vvngrids, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_grad(cudaStream_t stream, double *Fvec,
                    const double *vvcoords, const double *coords,
                    const double *W0p, const double *W0, const double *K,
                    const double *Kp, const double *RpW,
                    int vvngrids, int ngrids)
{
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_grad_kernel<<<blocks, threads, 0, stream>>>(Fvec, vvcoords, coords,
                      W0p, W0, K, Kp, RpW, vvngrids, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
