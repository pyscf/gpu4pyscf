/*
 * Copyright 2024 The PySCF Developers. All Rights Reserved.
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
#include <cuda_runtime.h>
#include "pbc.cuh"
#include "ft_ao.cuh"

#define GOUT_WIDTH      20
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2

#define AUXL            6
#define AUXNF           ((AUXL+1)*(AUXL+2)/2)

__global__ static
void ft_ao_bdiv_kernel(double *out, PBCIntEnvVars envs, int nGv, double *grids)
{
    int sh_block_id = gridDim.x - blockIdx.x - 1;
    int Gv_block_id = blockIdx.y;
    int nsh_per_block = FT_AO_THREADS / NG_PER_BLOCK;
    int sh_id_in_block = threadIdx.y;
    int Gv_id_in_block = threadIdx.x;
    int sh_id = sh_block_id * nsh_per_block + sh_id_in_block;
    if (sh_id >= envs.cell0_nbas) {
        return;
    }

    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    int li = bas[sh_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int16_t *idx = c_pair_idx + nfi * li;
    int16_t *idy = idx + nfi;
    int16_t *idz = idy + nfi;
    int iprim = bas[sh_id*BAS_SLOTS+NPRIM_OF];
    int Gv_id = Gv_block_id * NG_PER_BLOCK + Gv_id_in_block;
    double *Gv = grids + Gv_id;
    double kx = Gv[0];
    double ky = Gv[nGv];
    double kz = Gv[nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    int gx_len = (AUXL+1) * FT_AO_THREADS;
    __shared__ double g[(AUXL+1)*FT_AO_THREADS * 6];
    double *gxR = g + (AUXL+1) * NG_PER_BLOCK * sh_id_in_block + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxI + gx_len;
    double *gyI = gyR + gx_len;
    double *gzR = gyI + gx_len;
    double *gzI = gzR + gx_len;

    double goutR[AUXNF];
    double goutI[AUXNF];
#pragma unroll
    for (int n = 0; n < AUXNF; ++n) {
        goutR[n] = 0.;
        goutI[n] = 0.;
    }
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;
    double s0yR, s1yR, s2yR;
    double s0yI, s1yI, s2yI;
    double s0zR, s1zR, s2zR;
    double s0zI, s1zI, s2zI;

    int ia = bas[sh_id*BAS_SLOTS+ATOM_OF];
    double *expi = env + bas[sh_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[sh_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + atm[ia*ATM_SLOTS+PTR_COORD];
    for (int ip = 0; ip < iprim; ++ip) {
        __syncthreads();
        double ai = expi[ip];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double kR = kx * xi + ky * yi + kz * zi;
        sincos(-kR, gzI, gzR);
        double Kab = exp(-.25/ai*kk);
        gxR[0] = OVERLAP_FAC * ci[ip] / (ai * sqrt(ai));
        gxI[0] = 0.;
        gyR[0] = 1.;
        gyI[0] = 0.;
        gzR[0] *= Kab;
        gzI[0] *= Kab;

        if (li > 0) {
            double a2 = .5 / ai;
            double xpaI = -a2 * kx;
            double ypaI = -a2 * ky;
            double zpaI = -a2 * kz;
            s0xR = gxR[0];
            s0xI = gxI[0];
            s0yR = gyR[0];
            s0yI = gyI[0];
            s0zR = gzR[0];
            s0zI = gzI[0];
            s1xR = -xpaI * s0xI;
            s1xI =  xpaI * s0xR;
            s1yR = -ypaI * s0yI;
            s1yI =  ypaI * s0yR;
            s1zR = -zpaI * s0zI;
            s1zI =  zpaI * s0zR;
            gxR[NG_PER_BLOCK] = s1xR;
            gxI[NG_PER_BLOCK] = s1xI;
            gyR[NG_PER_BLOCK] = s1yR;
            gyI[NG_PER_BLOCK] = s1yI;
            gzR[NG_PER_BLOCK] = s1zR;
            gzI[NG_PER_BLOCK] = s1zI;
            for (int i = 2; i <= AUXL; i++) {
                if (i > li) break;
                double ia2 = (i-1) * a2;
                s2xR = ia2 * s0xR - xpaI * s1xI;
                s2xI = ia2 * s0xI + xpaI * s1xR;
                s2yR = ia2 * s0yR - ypaI * s1yI;
                s2yI = ia2 * s0yI + ypaI * s1yR;
                s2zR = ia2 * s0zR - zpaI * s1zI;
                s2zI = ia2 * s0zI + zpaI * s1zR;
                gxR[i*NG_PER_BLOCK] = s2xR;
                gxI[i*NG_PER_BLOCK] = s2xI;
                gyR[i*NG_PER_BLOCK] = s2yR;
                gyI[i*NG_PER_BLOCK] = s2yI;
                gzR[i*NG_PER_BLOCK] = s2zR;
                gzI[i*NG_PER_BLOCK] = s2zI;
                s0xR = s1xR;
                s0xI = s1xI;
                s0yR = s1yR;
                s0yI = s1yI;
                s0zR = s1zR;
                s0zI = s1zI;
                s1xR = s2xR;
                s1xI = s2xI;
                s1yR = s2yR;
                s1yI = s2yI;
                s1zR = s2zR;
                s1zI = s2zI;
            }
        }

        __syncthreads();
#pragma unroll
        for (int n = 0; n < AUXNF; ++n) {
            if (n >= nfi) break;
            int addrx = idx[n] * NG_PER_BLOCK;
            int addry = idy[n] * NG_PER_BLOCK;
            int addrz = idz[n] * NG_PER_BLOCK;
            double xR = gxR[addrx];
            double xI = gxI[addrx];
            double yR = gyR[addry];
            double yI = gyI[addry];
            double zR = gzR[addrz];
            double zI = gzI[addrz];
            double xyR = xR * yR - xI * yI;
            double xyI = xR * yI + xI * yR;
            goutR[n] += xyR * zR - xyI * zI;
            goutI[n] += xyR * zI + xyI * zR;
        }
    }

    if (Gv_id < nGv) {
        int stride = nGv * OF_COMPLEX;
        double *aft_tensor = out + (envs.ao_loc[sh_id] * nGv + Gv_id) * OF_COMPLEX;
#pragma unroll
        for (int n = 0; n < AUXNF; ++n) {
            if (n >= nfi) break;
            aft_tensor[n*stride  ] = goutR[n];
            aft_tensor[n*stride+1] = goutI[n];
        }
    }
}

extern "C" {
int build_ft_ao(double *out, PBCIntEnvVars *envs, int ngrids, double *grids,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    int nsh_per_block = FT_AO_THREADS/NG_PER_BLOCK;
    dim3 threads(NG_PER_BLOCK, nsh_per_block);
    int nbatches_grids = (ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK;
    int nbatches_shls = (nbas + nsh_per_block - 1) / nsh_per_block;
    dim3 blocks(nbatches_shls, nbatches_grids);
    ft_ao_bdiv_kernel<<<blocks, threads>>>(out, *envs, ngrids, grids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_bdiv_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
