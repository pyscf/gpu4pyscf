/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include "pbc.cuh"

#define LMAX 4
#define THREADS 256

template <int ANG> __device__ __forceinline__
void _cart_gto_ip2(double gto[], double gx[], double gy[], double gz[],
                   double a2, double rx, double ry, double rz)
{
#pragma unroll
    for (int i = 0; i < ANG+2; ++i) {
        gx[(i+1)] = gx[i] * rx;
        gy[(i+1)] = gy[i] * ry;
        gz[(i+1)] = gz[i] * rz;
    }
#pragma unroll
    for (int i = 0, lx = ANG; lx >= 0; lx--){
#pragma unroll
        for (int ly = ANG - lx; ly >= 0; ly--, i++){
            int lz = ANG - lx - ly;
            double fx0 = gx[lx];
            double fy0 = gy[ly];
            double fz0 = gz[lz];
            double fx1 = a2 * gx[lx+1];
            double fy1 = a2 * gy[ly+1];
            double fz1 = a2 * gz[lz+1];
            if (lx > 0) fx1 += lx*gx[lx-1];
            if (ly > 0) fy1 += ly*gy[ly-1];
            if (lz > 0) fz1 += lz*gz[lz-1];
            double fx2 = a2 * ((lx*2+1)*fx0 + a2*gx[lx+2]);
            double fy2 = a2 * ((ly*2+1)*fy0 + a2*gy[ly+2]);
            double fz2 = a2 * ((lz*2+1)*fz0 + a2*gz[lz+2]);
            if (lx > 1) fx2 += lx*(lx-1)*gx[lx-2];
            if (ly > 1) fy2 += ly*(ly-1)*gy[ly-2];
            if (lz > 1) fz2 += lz*(lz-1)*gz[lz-2];
            gto[i*6+0] += fx2 * fy0 * fz0;
            gto[i*6+1] += fx1 * fy1 * fz0;
            gto[i*6+2] += fx1 * fy0 * fz1;
            gto[i*6+3] += fx0 * fy2 * fz0;
            gto[i*6+4] += fx0 * fy1 * fz1;
            gto[i*6+5] += fx0 * fy0 * fz2;
        }
    }
}

__global__
static void _cart_deriv0_kernel(double *out, PBCIntEnvVars envs, double *grids,
                                size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto[n_cart_max];
    for (int n = 0; n < n_cart_max; ++n) {
        gto[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    int nimgs = envs.nimgs;
    for (int img = 0; img < nimgs; ++img) {
        double ce = 0;
        double rx = xi - img_coords[img*3+0];
        double ry = yi - img_coords[img*3+1];
        double rz = zi - img_coords[img*3+2];
        double rr = rx * rx + ry * ry + rz * rz;
        if (rr > rrcutoff) continue;
        for (int ip = 0; ip < nprim; ++ip) {
            ce += ci[ip] * exp(-expi[ip] * rr);
        }
        if (fabs(ce) < 1e-18) continue;
        switch (li) {
        case 0:
            gto[0] += ce;
            break;
        case 1:
            gto[0] += ce * rx;
            gto[1] += ce * ry;
            gto[2] += ce * rz;
            break;
        case 2:
            gto[0] += ce * rx * rx;
            gto[1] += ce * rx * ry;
            gto[2] += ce * rx * rz;
            gto[3] += ce * ry * ry;
            gto[4] += ce * ry * rz;
            gto[5] += ce * rz * rz;
            break;
        case 3:
            gto[0] += ce * rx * rx * rx;
            gto[1] += ce * rx * rx * ry;
            gto[2] += ce * rx * rx * rz;
            gto[3] += ce * rx * ry * ry;
            gto[4] += ce * rx * ry * rz;
            gto[5] += ce * rx * rz * rz;
            gto[6] += ce * ry * ry * ry;
            gto[7] += ce * ry * ry * rz;
            gto[8] += ce * ry * rz * rz;
            gto[9] += ce * rz * rz * rz;
            break;
        case 4:
            gto[0 ] += ce * rx * rx * rx * rx;
            gto[1 ] += ce * rx * rx * rx * ry;
            gto[2 ] += ce * rx * rx * rx * rz;
            gto[3 ] += ce * rx * rx * ry * ry;
            gto[4 ] += ce * rx * rx * ry * rz;
            gto[5 ] += ce * rx * rx * rz * rz;
            gto[6 ] += ce * rx * ry * ry * ry;
            gto[7 ] += ce * rx * ry * ry * rz;
            gto[8 ] += ce * rx * ry * rz * rz;
            gto[9 ] += ce * rx * rz * rz * rz;
            gto[10] += ce * ry * ry * ry * ry;
            gto[11] += ce * ry * ry * ry * rz;
            gto[12] += ce * ry * ry * rz * rz;
            gto[13] += ce * ry * rz * rz * rz;
            gto[14] += ce * rz * rz * rz * rz;
            break;
        }
    }
    int *ao_loc = envs.ao_loc;
    int nf = (li + 1) * (li + 2) / 2;
    out += ao_loc[bas_id] * ngrids;
    for (int n = 0; n < n_cart_max; ++n) {
        if (n >= nf) break;
        out[n*ngrids+grid_id] = gto[n];
    }
}

__global__
static void _cart_deriv1_kernel(double *out, PBCIntEnvVars envs, double *grids,
                                size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto [n_cart_max];
    double gtox[n_cart_max];
    double gtoy[n_cart_max];
    double gtoz[n_cart_max];
    for (int n = 0; n < n_cart_max; ++n) {
        gto [n] = 0;
        gtox[n] = 0;
        gtoy[n] = 0;
        gtoz[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    int nimgs = envs.nimgs;
    for (int img = 0; img < nimgs; ++img) {
        double ce = 0;
        double ce_2a = 0;
        double rx = xi - img_coords[img*3+0];
        double ry = yi - img_coords[img*3+1];
        double rz = zi - img_coords[img*3+2];
        double rr = rx * rx + ry * ry + rz * rz;
        if (rr > rrcutoff) continue;
        for (int ip = 0; ip < nprim; ++ip) {
            double ai = expi[ip];
            double c_exp = ci[ip] * exp(-ai * rr);
            ce += c_exp;
            ce_2a -= c_exp * ai * 2;
        }
        if (fabs(ce) < 1e-18) continue;
        switch (li) {
        case 0:
            gto [0] += ce;
            gtox[0] += ce_2a * rx;
            gtoy[0] += ce_2a * ry;
            gtoz[0] += ce_2a * rz;
            break;
        case 1: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto [0] += ce * rx;
            gto [1] += ce * ry;
            gto [2] += ce * rz;
            gtox[0] += ax * rx + ce;
            gtox[1] += ax * ry;
            gtox[2] += ax * rz;
            gtoy[0] += ay * rx;
            gtoy[1] += ay * ry + ce;
            gtoy[2] += ay * rz;
            gtoz[0] += az * rx;
            gtoz[1] += az * ry;
            gtoz[2] += az * rz + ce;
        } break;
        case 2: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto [0] += ce * rx * rx;
            gto [1] += ce * rx * ry;
            gto [2] += ce * rx * rz;
            gto [3] += ce * ry * ry;
            gto [4] += ce * ry * rz;
            gto [5] += ce * rz * rz;
            gtox[0] += (ax * rx + 2 * ce) * rx;
            gtox[1] += (ax * rx +     ce) * ry;
            gtox[2] += (ax * rx +     ce) * rz;
            gtox[3] += ax * ry * ry;
            gtox[4] += ax * ry * rz;
            gtox[5] += ax * rz * rz;
            gtoy[0] += ay * rx * rx;
            gtoy[1] += (ay * ry +     ce) * rx;
            gtoy[2] += ay * rx * rz;
            gtoy[3] += (ay * ry + 2 * ce) * ry;
            gtoy[4] += (ay * ry +     ce) * rz;
            gtoy[5] += ay * rz * rz;
            gtoz[0] += az * rx * rx;
            gtoz[1] += az * rx * ry;
            gtoz[2] += (az * rz +     ce) * rx;
            gtoz[3] += az * ry * ry;
            gtoz[4] += (az * rz +     ce) * ry;
            gtoz[5] += (az * rz + 2 * ce) * rz;
        } break;
        case 3: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto [0] += ce * rx * rx * rx;
            gto [1] += ce * rx * rx * ry;
            gto [2] += ce * rx * rx * rz;
            gto [3] += ce * rx * ry * ry;
            gto [4] += ce * rx * ry * rz;
            gto [5] += ce * rx * rz * rz;
            gto [6] += ce * ry * ry * ry;
            gto [7] += ce * ry * ry * rz;
            gto [8] += ce * ry * rz * rz;
            gto [9] += ce * rz * rz * rz;
            gtox[0] += (ax * rx + 3 * ce) * rx * rx;
            gtox[1] += (ax * rx + 2 * ce) * rx * ry;
            gtox[2] += (ax * rx + 2 * ce) * rx * rz;
            gtox[3] += (ax * rx +     ce) * ry * ry;
            gtox[4] += (ax * rx +     ce) * ry * rz;
            gtox[5] += (ax * rx +     ce) * rz * rz;
            gtox[6] += ax * ry * ry * ry;
            gtox[7] += ax * ry * ry * rz;
            gtox[8] += ax * ry * rz * rz;
            gtox[9] += ax * rz * rz * rz;
            gtoy[0] += ay * rx * rx * rx;
            gtoy[1] += (ay * ry +     ce) * rx * rx;
            gtoy[2] += ay * rx * rx * rz;
            gtoy[3] += (ay * ry + 2 * ce) * rx * ry;
            gtoy[4] += (ay * ry +     ce) * rx * rz;
            gtoy[5] += ay * rx * rz * rz;
            gtoy[6] += (ay * ry + 3 * ce) * ry * ry;
            gtoy[7] += (ay * ry + 2 * ce) * ry * rz;
            gtoy[8] += (ay * ry +     ce) * rz * rz;
            gtoy[9] += ay * rz * rz * rz;
            gtoz[0] += az * rx * rx * rx;
            gtoz[1] += az * rx * rx * ry;
            gtoz[2] += (az * rz +     ce) * rx * rx;
            gtoz[3] += az * rx * ry * ry;
            gtoz[4] += (az * rz +     ce) * rx * ry;
            gtoz[5] += (az * rz + 2 * ce) * rx * rz;
            gtoz[6] += az * ry * ry * ry;
            gtoz[7] += (az * rz +     ce) * ry * ry;
            gtoz[8] += (az * rz + 2 * ce) * ry * rz;
            gtoz[9] += (az * rz + 3 * ce) * rz * rz;
        } break;
        case 4: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            double bxxx = ce * rx * rx * rx;
            double bxxy = ce * rx * rx * ry;
            double bxxz = ce * rx * rx * rz;
            double bxyy = ce * rx * ry * ry;
            double bxyz = ce * rx * ry * rz;
            double bxzz = ce * rx * rz * rz;
            double byyy = ce * ry * ry * ry;
            double byyz = ce * ry * ry * rz;
            double byzz = ce * ry * rz * rz;
            double bzzz = ce * rz * rz * rz;
            gto [0 ] += ce * rx * rx * rx * rx;
            gto [1 ] += ce * rx * rx * rx * ry;
            gto [2 ] += ce * rx * rx * rx * rz;
            gto [3 ] += ce * rx * rx * ry * ry;
            gto [4 ] += ce * rx * rx * ry * rz;
            gto [5 ] += ce * rx * rx * rz * rz;
            gto [6 ] += ce * rx * ry * ry * ry;
            gto [7 ] += ce * rx * ry * ry * rz;
            gto [8 ] += ce * rx * ry * rz * rz;
            gto [9 ] += ce * rx * rz * rz * rz;
            gto [10] += ce * ry * ry * ry * ry;
            gto [11] += ce * ry * ry * ry * rz;
            gto [12] += ce * ry * ry * rz * rz;
            gto [13] += ce * ry * rz * rz * rz;
            gto [14] += ce * rz * rz * rz * rz;
            gtox[0 ] += ax * rx * rx * rx * rx + 4 * bxxx;
            gtox[1 ] += ax * rx * rx * rx * ry + 3 * bxxy;
            gtox[2 ] += ax * rx * rx * rx * rz + 3 * bxxz;
            gtox[3 ] += ax * rx * rx * ry * ry + 2 * bxyy;
            gtox[4 ] += ax * rx * rx * ry * rz + 2 * bxyz;
            gtox[5 ] += ax * rx * rx * rz * rz + 2 * bxzz;
            gtox[6 ] += ax * rx * ry * ry * ry +     byyy;
            gtox[7 ] += ax * rx * ry * ry * rz +     byyz;
            gtox[8 ] += ax * rx * ry * rz * rz +     byzz;
            gtox[9 ] += ax * rx * rz * rz * rz +     bzzz;
            gtox[10] += ax * ry * ry * ry * ry;
            gtox[11] += ax * ry * ry * ry * rz;
            gtox[12] += ax * ry * ry * rz * rz;
            gtox[13] += ax * ry * rz * rz * rz;
            gtox[14] += ax * rz * rz * rz * rz;
            gtoy[0 ] += ay * rx * rx * rx * rx;
            gtoy[1 ] += ay * rx * rx * rx * ry +     bxxx;
            gtoy[2 ] += ay * rx * rx * rx * rz;
            gtoy[3 ] += ay * rx * rx * ry * ry + 2 * bxxy;
            gtoy[4 ] += ay * rx * rx * ry * rz +     bxxz;
            gtoy[5 ] += ay * rx * rx * rz * rz;
            gtoy[6 ] += ay * rx * ry * ry * ry + 3 * bxyy;
            gtoy[7 ] += ay * rx * ry * ry * rz + 2 * bxyz;
            gtoy[8 ] += ay * rx * ry * rz * rz +     bxzz;
            gtoy[9 ] += ay * rx * rz * rz * rz;
            gtoy[10] += ay * ry * ry * ry * ry + 4 * byyy;
            gtoy[11] += ay * ry * ry * ry * rz + 3 * byyz;
            gtoy[12] += ay * ry * ry * rz * rz + 2 * byzz;
            gtoy[13] += ay * ry * rz * rz * rz +     bzzz;
            gtoy[14] += ay * rz * rz * rz * rz;
            gtoz[0 ] += az * rx * rx * rx * rx;
            gtoz[1 ] += az * rx * rx * rx * ry;
            gtoz[2 ] += az * rx * rx * rx * rz +     bxxx;
            gtoz[3 ] += az * rx * rx * ry * ry;
            gtoz[4 ] += az * rx * rx * ry * rz +     bxxy;
            gtoz[5 ] += az * rx * rx * rz * rz + 2 * bxxz;
            gtoz[6 ] += az * rx * ry * ry * ry;
            gtoz[7 ] += az * rx * ry * ry * rz +     bxyy;
            gtoz[8 ] += az * rx * ry * rz * rz + 2 * bxyz;
            gtoz[9 ] += az * rx * rz * rz * rz + 3 * bxzz;
            gtoz[10] += az * ry * ry * ry * ry;
            gtoz[11] += az * ry * ry * ry * rz +     byyy;
            gtoz[12] += az * ry * ry * rz * rz + 2 * byyz;
            gtoz[13] += az * ry * rz * rz * rz + 3 * byzz;
            gtoz[14] += az * rz * rz * rz * rz + 4 * bzzz;
        } }
    }
    int *ao_loc = envs.ao_loc;
    int nf = (li + 1) * (li + 2) / 2;
    out += ao_loc[bas_id] * ngrids + grid_id;
    double *outx = out + 1 * nao * ngrids;
    double *outy = out + 2 * nao * ngrids;
    double *outz = out + 3 * nao * ngrids;
    for (int n = 0; n < n_cart_max; ++n) {
        if (n >= nf) break;
        out [n*ngrids] = gto [n];
        outx[n*ngrids] = gtox[n];
        outy[n*ngrids] = gtoy[n];
        outz[n*ngrids] = gtoz[n];
    }
}

__global__
static void _cart_ip2_kernel(double *out, PBCIntEnvVars envs, double *grids,
                             size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    double gx[LMAX+3];
    double gy[LMAX+3];
    double gz[LMAX+3];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto[6*n_cart_max];
    for (int n = 0; n < 6*n_cart_max; ++n) {
        gto[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    gy[0] = 1.;
    gz[0] = 1.;
    int nimgs = envs.nimgs;
    for (int ip = 0; ip < nprim; ++ip) {
        double c = ci[ip];
        double ai = expi[ip];
        double a2 = -2 * ai;
        for (int img = 0; img < nimgs; ++img) {
            double rx = xi - img_coords[img*3+0];
            double ry = yi - img_coords[img*3+1];
            double rz = zi - img_coords[img*3+2];
            double rr = rx * rx + ry * ry + rz * rz;
            if (rr > rrcutoff) continue;
            double ce = c * exp(-ai * rr);
            if (fabs(ce) < 1e-18) continue;
            gx[0] = ce;
            switch (li) {
            case 0: {
                gx[1] = gx[0] * rx;
                gy[1] = gy[0] * ry;
                gz[1] = gz[0] * rz;
                gx[2] = gx[1] * rx;
                gy[2] = gy[1] * ry;
                gz[2] = gz[1] * rz;
                double fx0 = gx[0];
                double fy0 = gy[0];
                double fz0 = gz[0];
                double fx1 = a2 * gx[1];
                double fy1 = a2 * gy[1];
                double fz1 = a2 * gz[1];
                double fx2 = a2 * (fx0 + a2*gx[2]);
                double fy2 = a2 * (fy0 + a2*gy[2]);
                double fz2 = a2 * (fz0 + a2*gz[2]);
                gto[0] += fx2 * fy0 * fz0;
                gto[1] += fx1 * fy1 * fz0;
                gto[2] += fx1 * fy0 * fz1;
                gto[3] += fx0 * fy2 * fz0;
                gto[4] += fx0 * fy1 * fz1;
                gto[5] += fx0 * fy0 * fz2;
            } break;
            case 1:
                _cart_gto_ip2<1>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 2:
                _cart_gto_ip2<2>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 3:
                _cart_gto_ip2<3>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 4:
                _cart_gto_ip2<4>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            }
        }
    }

    int *ao_loc = envs.ao_loc;
    int nf = (li + 1) * (li + 2) / 2;
    double *out_x = out + ao_loc[bas_id] * ngrids + grid_id;
    for (int n = 0; n < n_cart_max; ++n) {
        if (n >= nf) break;
        for (int ix = 0; ix < 6; ++ix) {
            out_x[(ix*nao+n)*ngrids] = gto[n*6+ix];
        }
    }
}

__global__
static void _sph_deriv0_kernel(double *out, PBCIntEnvVars envs, double *grids,
                               size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto[n_cart_max];
    for (int n = 0; n < n_cart_max; ++n) {
        gto[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    int nimgs = envs.nimgs;
    for (int img = 0; img < nimgs; ++img) {
        double ce = 0;
        double rx = xi - img_coords[img*3+0];
        double ry = yi - img_coords[img*3+1];
        double rz = zi - img_coords[img*3+2];
        double rr = rx * rx + ry * ry + rz * rz;
        if (rr > rrcutoff) continue;
        for (int ip = 0; ip < nprim; ++ip) {
            ce += ci[ip] * exp(-expi[ip] * rr);
        }
        if (fabs(ce) < 1e-18) continue;
        switch (li) {
        case 0:
            gto[0] += ce;
            break;
        case 1:
            gto[0] += ce * rx;
            gto[1] += ce * ry;
            gto[2] += ce * rz;
            break;
        case 2:
            gto[0] += ce * rx * rx;
            gto[1] += ce * rx * ry;
            gto[2] += ce * rx * rz;
            gto[3] += ce * ry * ry;
            gto[4] += ce * ry * rz;
            gto[5] += ce * rz * rz;
            break;
        case 3:
            gto[0] += ce * rx * rx * rx;
            gto[1] += ce * rx * rx * ry;
            gto[2] += ce * rx * rx * rz;
            gto[3] += ce * rx * ry * ry;
            gto[4] += ce * rx * ry * rz;
            gto[5] += ce * rx * rz * rz;
            gto[6] += ce * ry * ry * ry;
            gto[7] += ce * ry * ry * rz;
            gto[8] += ce * ry * rz * rz;
            gto[9] += ce * rz * rz * rz;
            break;
        case 4:
            gto[0 ] += ce * rx * rx * rx * rx;
            gto[1 ] += ce * rx * rx * rx * ry;
            gto[2 ] += ce * rx * rx * rx * rz;
            gto[3 ] += ce * rx * rx * ry * ry;
            gto[4 ] += ce * rx * rx * ry * rz;
            gto[5 ] += ce * rx * rx * rz * rz;
            gto[6 ] += ce * rx * ry * ry * ry;
            gto[7 ] += ce * rx * ry * ry * rz;
            gto[8 ] += ce * rx * ry * rz * rz;
            gto[9 ] += ce * rx * rz * rz * rz;
            gto[10] += ce * ry * ry * ry * ry;
            gto[11] += ce * ry * ry * ry * rz;
            gto[12] += ce * ry * ry * rz * rz;
            gto[13] += ce * ry * rz * rz * rz;
            gto[14] += ce * rz * rz * rz * rz;
            break;
        }
    }
    int *ao_loc = envs.ao_loc;
    out += ao_loc[bas_id] * ngrids;
    switch (li) {
    case 0:
        out[grid_id] = gto[0];
        break;
    case 1:
        out[         grid_id] = gto[0];
        out[  ngrids+grid_id] = gto[1];
        out[2*ngrids+grid_id] = gto[2];
        break;
    case 2:
        out[         grid_id] = 1.092548430592079070 * gto[1];
        out[  ngrids+grid_id] = 1.092548430592079070 * gto[4];
        out[2*ngrids+grid_id] = 0.630783130505040012 * gto[5] - 0.315391565252520002 * (gto[0] + gto[3]);
        out[3*ngrids+grid_id] = 1.092548430592079070 * gto[2];
        out[4*ngrids+grid_id] = 0.546274215296039535 * (gto[0] - gto[3]);
        break;
    case 3:
        out[         grid_id] += 1.770130769779930531 * gto[1] - 0.590043589926643510 * gto[6];
        out[  ngrids+grid_id] += 2.890611442640554055 * gto[4];
        out[2*ngrids+grid_id] += 1.828183197857862944 * gto[8] - 0.457045799464465739 * (gto[1] + gto[6]);
        out[3*ngrids+grid_id] += 0.746352665180230782 * gto[9] - 1.119528997770346170 * (gto[2] + gto[7]);
        out[4*ngrids+grid_id] += 1.828183197857862944 * gto[5] - 0.457045799464465739 * (gto[0] + gto[3]);
        out[5*ngrids+grid_id] += 1.445305721320277020 * (gto[2] - gto[7]);
        out[6*ngrids+grid_id] += 0.590043589926643510 * gto[0] - 1.770130769779930530 * gto[3];
        break;
    case 4:          
        out[         grid_id] += 2.503342941796704538 * (gto[1] - gto[6]) ;
        out[  ngrids+grid_id] += 5.310392309339791593 * gto[4] - 1.770130769779930530 * gto[11];
        out[2*ngrids+grid_id] += 5.677048174545360108 * gto[8] - 0.946174695757560014 * (gto[1] + gto[6]);
        out[3*ngrids+grid_id] += 2.676186174229156671 * gto[13]- 2.007139630671867500 * (gto[4] + gto[11]);
        out[4*ngrids+grid_id] += 0.317356640745612911 * (gto[0] + gto[10]) + 0.634713281491225822 * gto[3] - 2.538853125964903290 * (gto[5] + gto[12]) + 0.846284375321634430 * gto[14];
        out[5*ngrids+grid_id] += 2.676186174229156671 * gto[9] - 2.007139630671867500 * (gto[2] + gto[7]);
        out[6*ngrids+grid_id] += 2.838524087272680054 * (gto[5] - gto[12]) + 0.473087347878780009 * (gto[10]- gto[0]);
        out[7*ngrids+grid_id] += 1.770130769779930531 * gto[2] - 5.310392309339791590 * gto[7];
        out[8*ngrids+grid_id] += 0.625835735449176134 * (gto[0] + gto[10]) - 3.755014412695056800 * gto[3];
        break;
    }
}

__global__
static void _sph_deriv1_kernel(double *out, PBCIntEnvVars envs, double *grids,
                               size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto [4*n_cart_max];
    for (int n = 0; n < 4*n_cart_max; ++n) {
        gto [n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    int nimgs = envs.nimgs;
    for (int img = 0; img < nimgs; ++img) {
        double ce = 0;
        double ce_2a = 0;
        double rx = xi - img_coords[img*3+0];
        double ry = yi - img_coords[img*3+1];
        double rz = zi - img_coords[img*3+2];
        double rr = rx * rx + ry * ry + rz * rz;
        if (rr > rrcutoff) continue;
        for (int ip = 0; ip < nprim; ++ip) {
            double ai = expi[ip];
            double c_exp = ci[ip] * exp(-ai * rr);
            ce += c_exp;
            ce_2a -= c_exp * ai * 2;
        }
        if (fabs(ce) < 1e-18) continue;
        switch (li) {
        case 0:
            gto[0] += ce;
            gto[1] += ce_2a * rx;
            gto[2] += ce_2a * ry;
            gto[3] += ce_2a * rz;
            break;
        case 1: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto[0 ] += ce * rx;
            gto[1 ] += ce * ry;
            gto[2 ] += ce * rz;
            gto[3 ] += ax * rx + ce;
            gto[4 ] += ax * ry;
            gto[5 ] += ax * rz;
            gto[6 ] += ay * rx;
            gto[7 ] += ay * ry + ce;
            gto[8 ] += ay * rz;
            gto[9 ] += az * rx;
            gto[10] += az * ry;
            gto[11] += az * rz + ce;
        } break;
        case 2: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto[0 ] += ce * rx * rx;
            gto[1 ] += ce * rx * ry;
            gto[2 ] += ce * rx * rz;
            gto[3 ] += ce * ry * ry;
            gto[4 ] += ce * ry * rz;
            gto[5 ] += ce * rz * rz;
            gto[6 ] += (ax * rx + 2 * ce) * rx;
            gto[7 ] += (ax * rx +     ce) * ry;
            gto[8 ] += (ax * rx +     ce) * rz;
            gto[9 ] += ax * ry * ry;
            gto[10] += ax * ry * rz;
            gto[11] += ax * rz * rz;
            gto[12] += ay * rx * rx;
            gto[13] += (ay * ry +     ce) * rx;
            gto[14] += ay * rx * rz;
            gto[15] += (ay * ry + 2 * ce) * ry;
            gto[16] += (ay * ry +     ce) * rz;
            gto[17] += ay * rz * rz;
            gto[18] += az * rx * rx;
            gto[19] += az * rx * ry;
            gto[20] += (az * rz +     ce) * rx;
            gto[21] += az * ry * ry;
            gto[22] += (az * rz +     ce) * ry;
            gto[23] += (az * rz + 2 * ce) * rz;
        } break;
        case 3: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            gto[0 ] += ce * rx * rx * rx;
            gto[1 ] += ce * rx * rx * ry;
            gto[2 ] += ce * rx * rx * rz;
            gto[3 ] += ce * rx * ry * ry;
            gto[4 ] += ce * rx * ry * rz;
            gto[5 ] += ce * rx * rz * rz;
            gto[6 ] += ce * ry * ry * ry;
            gto[7 ] += ce * ry * ry * rz;
            gto[8 ] += ce * ry * rz * rz;
            gto[9 ] += ce * rz * rz * rz;
            gto[10] += (ax * rx + 3 * ce) * rx * rx;
            gto[11] += (ax * rx + 2 * ce) * rx * ry;
            gto[12] += (ax * rx + 2 * ce) * rx * rz;
            gto[13] += (ax * rx +     ce) * ry * ry;
            gto[14] += (ax * rx +     ce) * ry * rz;
            gto[15] += (ax * rx +     ce) * rz * rz;
            gto[16] += ax * ry * ry * ry;
            gto[17] += ax * ry * ry * rz;
            gto[18] += ax * ry * rz * rz;
            gto[19] += ax * rz * rz * rz;
            gto[20] += ay * rx * rx * rx;
            gto[21] += (ay * ry +     ce) * rx * rx;
            gto[22] += ay * rx * rx * rz;
            gto[23] += (ay * ry + 2 * ce) * rx * ry;
            gto[24] += (ay * ry +     ce) * rx * rz;
            gto[25] += ay * rx * rz * rz;
            gto[26] += (ay * ry + 3 * ce) * ry * ry;
            gto[27] += (ay * ry + 2 * ce) * ry * rz;
            gto[28] += (ay * ry +     ce) * rz * rz;
            gto[29] += ay * rz * rz * rz;
            gto[30] += az * rx * rx * rx;
            gto[31] += az * rx * rx * ry;
            gto[32] += (az * rz +     ce) * rx * rx;
            gto[33] += az * rx * ry * ry;
            gto[34] += (az * rz +     ce) * rx * ry;
            gto[35] += (az * rz + 2 * ce) * rx * rz;
            gto[36] += az * ry * ry * ry;
            gto[37] += (az * rz +     ce) * ry * ry;
            gto[38] += (az * rz + 2 * ce) * ry * rz;
            gto[39] += (az * rz + 3 * ce) * rz * rz;
        } break;
        case 4: {
            double ax = ce_2a * rx;
            double ay = ce_2a * ry;
            double az = ce_2a * rz;
            double bxxx = ce * rx * rx * rx;
            double bxxy = ce * rx * rx * ry;
            double bxxz = ce * rx * rx * rz;
            double bxyy = ce * rx * ry * ry;
            double bxyz = ce * rx * ry * rz;
            double bxzz = ce * rx * rz * rz;
            double byyy = ce * ry * ry * ry;
            double byyz = ce * ry * ry * rz;
            double byzz = ce * ry * rz * rz;
            double bzzz = ce * rz * rz * rz;
            gto[0 ] += ce * rx * rx * rx * rx;
            gto[1 ] += ce * rx * rx * rx * ry;
            gto[2 ] += ce * rx * rx * rx * rz;
            gto[3 ] += ce * rx * rx * ry * ry;
            gto[4 ] += ce * rx * rx * ry * rz;
            gto[5 ] += ce * rx * rx * rz * rz;
            gto[6 ] += ce * rx * ry * ry * ry;
            gto[7 ] += ce * rx * ry * ry * rz;
            gto[8 ] += ce * rx * ry * rz * rz;
            gto[9 ] += ce * rx * rz * rz * rz;
            gto[10] += ce * ry * ry * ry * ry;
            gto[11] += ce * ry * ry * ry * rz;
            gto[12] += ce * ry * ry * rz * rz;
            gto[13] += ce * ry * rz * rz * rz;
            gto[14] += ce * rz * rz * rz * rz;
            gto[15] += ax * rx * rx * rx * rx + 4 * bxxx;
            gto[16] += ax * rx * rx * rx * ry + 3 * bxxy;
            gto[17] += ax * rx * rx * rx * rz + 3 * bxxz;
            gto[18] += ax * rx * rx * ry * ry + 2 * bxyy;
            gto[19] += ax * rx * rx * ry * rz + 2 * bxyz;
            gto[20] += ax * rx * rx * rz * rz + 2 * bxzz;
            gto[21] += ax * rx * ry * ry * ry +     byyy;
            gto[22] += ax * rx * ry * ry * rz +     byyz;
            gto[23] += ax * rx * ry * rz * rz +     byzz;
            gto[24] += ax * rx * rz * rz * rz +     bzzz;
            gto[25] += ax * ry * ry * ry * ry;
            gto[26] += ax * ry * ry * ry * rz;
            gto[27] += ax * ry * ry * rz * rz;
            gto[28] += ax * ry * rz * rz * rz;
            gto[29] += ax * rz * rz * rz * rz;
            gto[30] += ay * rx * rx * rx * rx;
            gto[31] += ay * rx * rx * rx * ry +     bxxx;
            gto[32] += ay * rx * rx * rx * rz;
            gto[33] += ay * rx * rx * ry * ry + 2 * bxxy;
            gto[34] += ay * rx * rx * ry * rz +     bxxz;
            gto[35] += ay * rx * rx * rz * rz;
            gto[36] += ay * rx * ry * ry * ry + 3 * bxyy;
            gto[37] += ay * rx * ry * ry * rz + 2 * bxyz;
            gto[38] += ay * rx * ry * rz * rz +     bxzz;
            gto[39] += ay * rx * rz * rz * rz;
            gto[40] += ay * ry * ry * ry * ry + 4 * byyy;
            gto[41] += ay * ry * ry * ry * rz + 3 * byyz;
            gto[42] += ay * ry * ry * rz * rz + 2 * byzz;
            gto[43] += ay * ry * rz * rz * rz +     bzzz;
            gto[44] += ay * rz * rz * rz * rz;
            gto[45] += az * rx * rx * rx * rx;
            gto[46] += az * rx * rx * rx * ry;
            gto[47] += az * rx * rx * rx * rz +     bxxx;
            gto[48] += az * rx * rx * ry * ry;
            gto[49] += az * rx * rx * ry * rz +     bxxy;
            gto[50] += az * rx * rx * rz * rz + 2 * bxxz;
            gto[51] += az * rx * ry * ry * ry;
            gto[52] += az * rx * ry * ry * rz +     bxyy;
            gto[53] += az * rx * ry * rz * rz + 2 * bxyz;
            gto[54] += az * rx * rz * rz * rz + 3 * bxzz;
            gto[55] += az * ry * ry * ry * ry;
            gto[56] += az * ry * ry * ry * rz +     byyy;
            gto[57] += az * ry * ry * rz * rz + 2 * byyz;
            gto[58] += az * ry * rz * rz * rz + 3 * byzz;
            gto[59] += az * rz * rz * rz * rz + 4 * bzzz;
        } }
    }
    int *ao_loc = envs.ao_loc;
    out += ao_loc[bas_id] * ngrids + grid_id;
    switch (li) {
    case 0:
        for (int n = 0; n < 4; ++n) {
            out[n * nao * ngrids] = gto[n];
        }
        break;
    case 1:
        for (int n = 0; n < 4; ++n) {
            out[(n*nao+0)*ngrids] = gto[n*3+0];
            out[(n*nao+1)*ngrids] = gto[n*3+1];
            out[(n*nao+2)*ngrids] = gto[n*3+2];
        }
        break;
    case 2:
        for (int n = 0; n < 4; ++n) {
            out[(n*nao+0)*ngrids] = 1.092548430592079070 * gto[n*6+1];
            out[(n*nao+1)*ngrids] = 1.092548430592079070 * gto[n*6+4];
            out[(n*nao+2)*ngrids] = 0.630783130505040012 * gto[n*6+5] - 0.315391565252520002 * (gto[n*6+0] + gto[n*6+3]);
            out[(n*nao+3)*ngrids] = 1.092548430592079070 * gto[n*6+2];
            out[(n*nao+4)*ngrids] = 0.546274215296039535 * (gto[n*6+0] - gto[n*6+3]);
        }
        break;
    case 3:
        for (int n = 0; n < 4; ++n) {
            out[(n*nao+0)*ngrids] += 1.770130769779930531 * gto[n*10+1] - 0.590043589926643510 * gto[n*10+6];
            out[(n*nao+1)*ngrids] += 2.890611442640554055 * gto[n*10+4];
            out[(n*nao+2)*ngrids] += 1.828183197857862944 * gto[n*10+8] - 0.457045799464465739 * (gto[n*10+1] + gto[n*10+6]);
            out[(n*nao+3)*ngrids] += 0.746352665180230782 * gto[n*10+9] - 1.119528997770346170 * (gto[n*10+2] + gto[n*10+7]);
            out[(n*nao+4)*ngrids] += 1.828183197857862944 * gto[n*10+5] - 0.457045799464465739 * (gto[n*10+0] + gto[n*10+3]);
            out[(n*nao+5)*ngrids] += 1.445305721320277020 * (gto[n*10+2] - gto[n*10+7]);
            out[(n*nao+6)*ngrids] += 0.590043589926643510 * gto[n*10+0] - 1.770130769779930530 * gto[n*10+3];
        }
        break;
    case 4:          
        for (int n = 0; n < 4; ++n) {
            out[(n*nao+0)*ngrids] += 2.503342941796704538 * (gto[n*15+1] - gto[n*15+6]) ;
            out[(n*nao+1)*ngrids] += 5.310392309339791593 * gto[n*15+4] - 1.770130769779930530 * gto[n*15+11];
            out[(n*nao+2)*ngrids] += 5.677048174545360108 * gto[n*15+8] - 0.946174695757560014 * (gto[n*15+1] + gto[n*15+6]);
            out[(n*nao+3)*ngrids] += 2.676186174229156671 * gto[n*15+13]- 2.007139630671867500 * (gto[n*15+4] + gto[n*15+11]);
            out[(n*nao+4)*ngrids] += 0.317356640745612911 * (gto[n*15+0] + gto[n*15+10]) + 0.634713281491225822 * gto[n*15+3] - 2.538853125964903290 * (gto[n*15+5] + gto[n*15+12]) + 0.846284375321634430 * gto[n*15+14];
            out[(n*nao+5)*ngrids] += 2.676186174229156671 * gto[n*15+9] - 2.007139630671867500 * (gto[n*15+2] + gto[n*15+7]);
            out[(n*nao+6)*ngrids] += 2.838524087272680054 * (gto[n*15+5] - gto[n*15+12]) + 0.473087347878780009 * (gto[n*15+10]- gto[n*15+0]);
            out[(n*nao+7)*ngrids] += 1.770130769779930531 * gto[n*15+2] - 5.310392309339791590 * gto[n*15+7];
            out[(n*nao+8)*ngrids] += 0.625835735449176134 * (gto[n*15+0] + gto[n*15+10]) - 3.755014412695056800 * gto[n*15+3];
        }
        break;
    }
}

__global__
static void _sph_ip2_kernel(double *out, PBCIntEnvVars envs, double *grids,
                            size_t ngrids, int nao, double *rcut)
{
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id] - ri[0];
    double yi = gridy[grid_id] - ri[1];
    double zi = gridz[grid_id] - ri[2];
    double gx[LMAX+3];
    double gy[LMAX+3];
    double gz[LMAX+3];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto[6*n_cart_max];
    for (int n = 0; n < 6*n_cart_max; ++n) {
        gto[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    gy[0] = 1.;
    gz[0] = 1.;
    int nimgs = envs.nimgs;
    for (int ip = 0; ip < nprim; ++ip) {
        double c = ci[ip];
        double ai = expi[ip];
        double a2 = -2 * ai;
        for (int img = 0; img < nimgs; ++img) {
            double rx = xi - img_coords[img*3+0];
            double ry = yi - img_coords[img*3+1];
            double rz = zi - img_coords[img*3+2];
            double rr = rx * rx + ry * ry + rz * rz;
            if (rr > rrcutoff) continue;
            double ce = c * exp(-ai * rr);
            if (fabs(ce) < 1e-18) continue;
            gx[0] = ce;
            switch (li) {
            case 0: {
                gx[1] = gx[0] * rx;
                gy[1] = gy[0] * ry;
                gz[1] = gz[0] * rz;
                gx[2] = gx[1] * rx;
                gy[2] = gy[1] * ry;
                gz[2] = gz[1] * rz;
                double fx0 = gx[0];
                double fy0 = gy[0];
                double fz0 = gz[0];
                double fx1 = a2 * gx[1];
                double fy1 = a2 * gy[1];
                double fz1 = a2 * gz[1];
                double fx2 = a2 * (fx0 + a2*gx[2]);
                double fy2 = a2 * (fy0 + a2*gy[2]);
                double fz2 = a2 * (fz0 + a2*gz[2]);
                gto[0] += fx2 * fy0 * fz0;
                gto[1] += fx1 * fy1 * fz0;
                gto[2] += fx1 * fy0 * fz1;
                gto[3] += fx0 * fy2 * fz0;
                gto[4] += fx0 * fy1 * fz1;
                gto[5] += fx0 * fy0 * fz2;
            } break;
            case 1:
                _cart_gto_ip2<1>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 2:
                _cart_gto_ip2<2>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 3:
                _cart_gto_ip2<3>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            case 4:
                _cart_gto_ip2<4>(gto, gx, gy, gz, a2, rx, ry, rz);
                break;
            }
        }
    }

    int *ao_loc = envs.ao_loc;
    out += ao_loc[bas_id] * ngrids + grid_id;
    switch (li) {
    case 0:
        for (int n = 0; n < 6; ++n) {
            out[n * nao * ngrids] = gto[n];
        }
        break;
    case 1:
        for (int n = 0; n < 6; ++n) {
            out[(n*nao+0)*ngrids] = gto[0*6+n];
            out[(n*nao+1)*ngrids] = gto[1*6+n];
            out[(n*nao+2)*ngrids] = gto[2*6+n];
        }
        break;
    case 2:
        for (int n = 0; n < 6; ++n) {
            out[(n*nao+0)*ngrids] = 1.092548430592079070 * gto[1*6+n];
            out[(n*nao+1)*ngrids] = 1.092548430592079070 * gto[4*6+n];
            out[(n*nao+2)*ngrids] = 0.630783130505040012 * gto[5*6+n] - 0.315391565252520002 * (gto[0*6+n] + gto[3*6+n]);
            out[(n*nao+3)*ngrids] = 1.092548430592079070 * gto[2*6+n];
            out[(n*nao+4)*ngrids] = 0.546274215296039535 * (gto[0*6+n] - gto[3*6+n]);
        }
        break;
    case 3:
        for (int n = 0; n < 6; ++n) {
            out[(n*nao+0)*ngrids] += 1.770130769779930531 * gto[1*6+n] - 0.590043589926643510 * gto[6*6+n];
            out[(n*nao+1)*ngrids] += 2.890611442640554055 * gto[4*6+n];
            out[(n*nao+2)*ngrids] += 1.828183197857862944 * gto[8*6+n] - 0.457045799464465739 * (gto[1*6+n] + gto[6*6+n]);
            out[(n*nao+3)*ngrids] += 0.746352665180230782 * gto[9*6+n] - 1.119528997770346170 * (gto[2*6+n] + gto[7*6+n]);
            out[(n*nao+4)*ngrids] += 1.828183197857862944 * gto[5*6+n] - 0.457045799464465739 * (gto[0*6+n] + gto[3*6+n]);
            out[(n*nao+5)*ngrids] += 1.445305721320277020 * (gto[2*6+n] - gto[7*6+n]);
            out[(n*nao+6)*ngrids] += 0.590043589926643510 * gto[0*6+n] - 1.770130769779930530 * gto[3*6+n];
        }
        break;
    case 4:          
        for (int n = 0; n < 6; ++n) {
            out[(n*nao+0)*ngrids] += 2.503342941796704538 * (gto[1*6+n] - gto[6*6+n]) ;
            out[(n*nao+1)*ngrids] += 5.310392309339791593 * gto[4*6+n] - 1.770130769779930530 * gto[11];
            out[(n*nao+2)*ngrids] += 5.677048174545360108 * gto[8*6+n] - 0.946174695757560014 * (gto[1*6+n] + gto[6*6+n]);
            out[(n*nao+3)*ngrids] += 2.676186174229156671 * gto[13]- 2.007139630671867500 * (gto[4*6+n] + gto[11]);
            out[(n*nao+4)*ngrids] += 0.317356640745612911 * (gto[0*6+n] + gto[10]) + 0.634713281491225822 * gto[3*6+n] - 2.538853125964903290 * (gto[5*6+n] + gto[12]) + 0.846284375321634430 * gto[14];
            out[(n*nao+5)*ngrids] += 2.676186174229156671 * gto[9*6+n] - 2.007139630671867500 * (gto[2*6+n] + gto[7*6+n]);
            out[(n*nao+6)*ngrids] += 2.838524087272680054 * (gto[5*6+n] - gto[12]) + 0.473087347878780009 * (gto[10]- gto[0*6+n]);
            out[(n*nao+7)*ngrids] += 1.770130769779930531 * gto[2*6+n] - 5.310392309339791590 * gto[7*6+n];
            out[(n*nao+8)*ngrids] += 0.625835735449176134 * (gto[0*6+n] + gto[10]) - 3.755014412695056800 * gto[3*6+n];
        }
        break;
    }
}

extern "C" {
int PBCeval_gto_deriv(double *out, PBCIntEnvVars *envs,
                      double *grids, int ngrids, int nao, int nbas,
                      int deriv, int cart, double *rcut)
{
    constexpr int ngrids_per_block = THREADS;
    int threads = ngrids_per_block;
    dim3 blocks((ngrids+ngrids_per_block-1)/ngrids_per_block, nbas);
    switch (deriv) {
    case 0:
        if (cart) {
            _cart_deriv0_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        } else {
            _sph_deriv0_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        }
        break;
    case 1:
        if (cart) {
            _cart_deriv1_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        } else {
            _sph_deriv1_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        }
        break;
    case 2:
        if (cart) {
            _cart_deriv1_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
            _cart_ip2_kernel<<<blocks, threads>>>(out+4*nao*ngrids, *envs, grids, ngrids, nao, rcut);
        } else {
            _sph_deriv1_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
            _sph_ip2_kernel<<<blocks, threads>>>(out+4*nao*ngrids, *envs, grids, ngrids, nao, rcut);
        }
        break;
    default:
        fprintf(stderr, "deriv = %d not supported\n", deriv);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCeval_gto: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
