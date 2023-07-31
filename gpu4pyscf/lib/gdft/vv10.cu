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
#include <cint.h>
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define THREADS        128

__global__
static void vv10_kernel(double *Fvec, double *Uvec, double *Wvec,
    double *vvcoords, double *coords,
    double *W0p, double *W0, double *K, double *Kp, double *RpW,
    int vvngrids, int ngrids)
{
    // grid id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids) {
        return;
    }
    
    double DX, DY, DZ, R2;
    double gp, g, gt, T;
    double x = coords[i*3+0];
    double y = coords[i*3+1];
    double z = coords[i*3+2];
    double W0i = W0[i];
    double Ki = K[i];
    double F = 0;
    double U = 0;
    double W = 0;
    for (int j = 0; j < vvngrids; j++) {
        DX = vvcoords[j*3+0] - x;
        DY = vvcoords[j*3+1] - y;
        DZ = vvcoords[j*3+2] - z;
        R2 = DX*DX + DY*DY + DZ*DZ;
        gp = R2*W0p[j] + Kp[j];
        g  = R2*W0i + Ki;
        gt = g + gp;
        T = RpW[j] / (g*gp*gt);
        F += T;
        T *= 1./g + 1./gt;
        U += T;
        W += T * R2;
    }
    Fvec[i] = F * -1.5;
    Uvec[i] = U;
    Wvec[i] = W;
}

__global__
static void vv10_grad_kernel(double *Fvec, double *vvcoords, double *coords,
    double *W0p, double *W0, double *K, double *Kp, double *RpW,
    int vvngrids, int ngrids)
{
    // grid id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids) {
        return;
    }

    double DX, DY, DZ, R2;
    double gp, g, gt, T, Q;
    double FX = 0;
    double FY = 0;
    double FZ = 0;
    for (int j = 0; j < vvngrids; j++) {
        DX = vvcoords[j*3+0] - coords[i*3+0];
        DY = vvcoords[j*3+1] - coords[i*3+1];
        DZ = vvcoords[j*3+2] - coords[i*3+2];
        R2 = DX*DX + DY*DY + DZ*DZ;
        gp = R2*W0p[j] + Kp[j];
        g  = R2*W0[i] + K[i];
        gt = g + gp;
        T = RpW[j] / (g*gp*gt);
        Q = T * (W0[i]/g + W0p[j]/gp + (W0[i]+W0p[j])/gt);
        FX += Q * DX;
        FY += Q * DY;
        FZ += Q * DZ;
    }
    Fvec[i*3+0] = FX * -3;
    Fvec[i*3+1] = FY * -3;
    Fvec[i*3+2] = FZ * -3;
}

extern "C" {
__host__
int VXC_vv10nlc(cudaStream_t stream, double *Fvec, double *Uvec, double *Wvec,
                 double *vvcoords, double *coords,
                 double *W0p, double *W0, double *K, double *Kp, double *RpW,
                 int vvngrids, int ngrids)
{
    dim3 threads(THREADS);
    dim3 blocks((ngrids+THREADS-1)/THREADS);
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
int VXC_vv10nlc_grad(cudaStream_t stream, double *Fvec, double *vvcoords, double *coords,
                      double *W0p, double *W0, double *K, double *Kp, double *RpW,
                      int vvngrids, int ngrids)
{
    dim3 threads(THREADS);
    dim3 blocks((ngrids+THREADS-1)/THREADS);
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
