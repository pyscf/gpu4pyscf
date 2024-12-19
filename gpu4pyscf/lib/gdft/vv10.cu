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
            // about 23 operations for each pair
            double DX = xj_tmp.x - xi;
            double DY = xj_tmp.y - yi;
            double DZ = xj_tmp.z - zi;
            double R2 = DX*DX + DY*DY + DZ*DZ;

            double3 kp_tmp = kp_t[l];
            double gp = R2*kp_tmp.y + kp_tmp.x;
            double g  = R2*W0i + Ki;
            double gt = g + gp;
            double ggp = g * gp;
            double ggt_gp = gt * ggp;
            double T = kp_tmp.z / (ggt_gp * ggt_gp);
            double Q = T * ((W0i*gp + kp_tmp.y*g)*gt + (W0i+kp_tmp.y)*ggp);

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
