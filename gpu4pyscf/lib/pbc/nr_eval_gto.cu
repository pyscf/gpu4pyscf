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
#include "gvhf-rys/vhf.cuh"

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

template <int ANG> __device__ __forceinline__
void _cart_deriv1_strain_tensor(
        double ao[], double gx[], double gy[], double gz[],
        double a2, double rx, double ry, double rz,
        double Rx, double Ry, double Rz, int n0)
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
            if (i == n0 || i == n0+1) {
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
                double fyz0 = fy0 * fz0;
                double fxz0 = fx0 * fz0;
                double fxy0 = fx0 * fy0;
                double vx = fx1 * fyz0;
                double vy = fy1 * fxz0;
                double vz = fz1 * fxy0;
                double vxx = fx2 * fyz0;
                double vyy = fy2 * fxz0;
                double vzz = fz2 * fxy0;
                double vxy = fx1 * fy1 * fz0;
                double vxz = fx1 * fy0 * fz1;
                double vyz = fx0 * fy1 * fz1;
                ao[(i-n0)*36+0 ] -= vx * Rx;
                ao[(i-n0)*36+1 ] -= vx * Ry;
                ao[(i-n0)*36+2 ] -= vx * Rz;
                ao[(i-n0)*36+3 ] -= vy * Rx;
                ao[(i-n0)*36+4 ] -= vy * Ry;
                ao[(i-n0)*36+5 ] -= vy * Rz;
                ao[(i-n0)*36+6 ] -= vz * Rx;
                ao[(i-n0)*36+7 ] -= vz * Ry;
                ao[(i-n0)*36+8 ] -= vz * Rz;
                ao[(i-n0)*36+9 ] -= vxx * Rx;
                ao[(i-n0)*36+10] -= vxx * Ry;
                ao[(i-n0)*36+11] -= vxx * Rz;
                ao[(i-n0)*36+12] -= vxy * Rx;
                ao[(i-n0)*36+13] -= vxy * Ry;
                ao[(i-n0)*36+14] -= vxy * Rz;
                ao[(i-n0)*36+15] -= vxz * Rx;
                ao[(i-n0)*36+16] -= vxz * Ry;
                ao[(i-n0)*36+17] -= vxz * Rz;
                ao[(i-n0)*36+18] -= vxy * Rx;
                ao[(i-n0)*36+19] -= vxy * Ry;
                ao[(i-n0)*36+20] -= vxy * Rz;
                ao[(i-n0)*36+21] -= vyy * Rx;
                ao[(i-n0)*36+22] -= vyy * Ry;
                ao[(i-n0)*36+23] -= vyy * Rz;
                ao[(i-n0)*36+24] -= vyz * Rx;
                ao[(i-n0)*36+25] -= vyz * Ry;
                ao[(i-n0)*36+26] -= vyz * Rz;
                ao[(i-n0)*36+27] -= vxz * Rx;
                ao[(i-n0)*36+28] -= vxz * Ry;
                ao[(i-n0)*36+29] -= vxz * Rz;
                ao[(i-n0)*36+30] -= vyz * Rx;
                ao[(i-n0)*36+31] -= vyz * Ry;
                ao[(i-n0)*36+32] -= vyz * Rz;
                ao[(i-n0)*36+33] -= vzz * Rx;
                ao[(i-n0)*36+34] -= vzz * Ry;
                ao[(i-n0)*36+35] -= vzz * Rz;
            }
        }
    }
}

template <int ANG> __device__
void _eval_cart_deriv1_strain_tensor(
        double *out, double *img_coords, double *env,
        double xi, double yi, double zi, double rrcutoff,
        int *bas, int nimgs, int nao, int ngrids)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int bas_id = item.get_group(0);
#else
    int bas_id = blockIdx.y;
#endif
    int nprim = bas[NPRIM_OF+bas_id*BAS_SLOTS];
    double *expi = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[bas_id*BAS_SLOTS+PTR_BAS_COORD];
    double cell0_Rx = ri[0];
    double cell0_Ry = ri[1];
    double cell0_Rz = ri[2];
    double gx[LMAX+3];
    double gy[LMAX+3];
    double gz[LMAX+3];
    double ao[72];
    gy[0] = 1.;
    gz[0] = 1.;
    constexpr int n_cart = (ANG+1)*(ANG+2)/2;
    size_t naog = nao * ngrids;
#pragma unroll
    for (int n0 = 0; n0 < n_cart; n0 += 2) {
        for (int n = 0; n < min(n_cart-n0, 2)*36; ++n) {
            ao[n] = 0;
        }
        for (int ip = 0; ip < nprim; ++ip) {
            double c = ci[ip];
            double ai = expi[ip];
            double a2 = -2 * ai;
            for (int img = 0; img < nimgs; ++img) {
                double Rx = img_coords[img*3+0] + cell0_Rx;
                double Ry = img_coords[img*3+1] + cell0_Ry;
                double Rz = img_coords[img*3+2] + cell0_Rz;
                double rx = xi - Rx;
                double ry = yi - Ry;
                double rz = zi - Rz;
                double rr = rx * rx + ry * ry + rz * rz;
                if (rr > rrcutoff) continue;
                double ce = c * exp(-ai * rr);
                if (fabs(ce) < 1e-18) continue;
                gx[0] = ce;
                _cart_deriv1_strain_tensor<ANG>(
                        ao, gx, gy, gz, a2, rx, ry, rz, Rx, Ry, Rz, n0);
            }
        }
        for (int n = 0; n < min(n_cart-n0, 2); ++n) {
            for (int x = 0; x < 9; ++x) {
                for (int s = 0; s < 4; ++s) {
                    out[(x*4+s)*naog+(n0+n)*ngrids] = ao[n*36+s*9+x];
                }
            }
        }
    }
}

__global__
static void _cart_deriv0_kernel(double *out, PBCIntEnvVars envs, double *grids,
                                size_t ngrids, int nao, double *rcut)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
        double ax = ce_2a * rx;
        double ay = ce_2a * ry;
        double az = ce_2a * rz;
        switch (li) {
        case 0:
            gto [0] += ce;
            gtox[0] += ax;
            gtoy[0] += ay;
            gtoz[0] += az;
            break;
        case 1: {
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
            gtox[0 ] += (ax * rx + 4 * ce) * rx * rx * rx;
            gtox[1 ] += (ax * rx + 3 * ce) * rx * rx * ry;
            gtox[2 ] += (ax * rx + 3 * ce) * rx * rx * rz;
            gtox[3 ] += (ax * rx + 2 * ce) * rx * ry * ry;
            gtox[4 ] += (ax * rx + 2 * ce) * rx * ry * rz;
            gtox[5 ] += (ax * rx + 2 * ce) * rx * rz * rz;
            gtox[6 ] += (ax * rx +     ce) * ry * ry * ry;
            gtox[7 ] += (ax * rx +     ce) * ry * ry * rz;
            gtox[8 ] += (ax * rx +     ce) * ry * rz * rz;
            gtox[9 ] += (ax * rx +     ce) * rz * rz * rz;
            gtox[10] += ax * ry * ry * ry * ry;
            gtox[11] += ax * ry * ry * ry * rz;
            gtox[12] += ax * ry * ry * rz * rz;
            gtox[13] += ax * ry * rz * rz * rz;
            gtox[14] += ax * rz * rz * rz * rz;
            gtoy[0 ] += ay * rx * rx * rx * rx;
            gtoy[1 ] += (ay * ry +     ce) * rx * rx * rx;
            gtoy[2 ] += ay * rx * rx * rx * rz;
            gtoy[3 ] += (ay * ry + 2 * ce) * rx * rx * ry;
            gtoy[4 ] += (ay * ry +     ce) * rx * rx * rz;
            gtoy[5 ] += ay * rx * rx * rz * rz;
            gtoy[6 ] += (ay * ry + 3 * ce) * rx * ry * ry;
            gtoy[7 ] += (ay * ry + 2 * ce) * rx * ry * rz;
            gtoy[8 ] += (ay * ry +     ce) * rx * rz * rz;
            gtoy[9 ] += ay * rx * rz * rz * rz;
            gtoy[10] += (ay * ry + 4 * ce) * ry * ry * ry;
            gtoy[11] += (ay * ry + 3 * ce) * ry * ry * rz;
            gtoy[12] += (ay * ry + 2 * ce) * ry * rz * rz;
            gtoy[13] += (ay * ry +     ce) * rz * rz * rz;
            gtoy[14] += ay * rz * rz * rz * rz;
            gtoz[0 ] += az * rx * rx * rx * rx;
            gtoz[1 ] += az * rx * rx * rx * ry;
            gtoz[2 ] += (az * rz +     ce) * rx * rx * rx;
            gtoz[3 ] += az * rx * rx * ry * ry;
            gtoz[4 ] += (az * rz +     ce) * rx * rx * ry;
            gtoz[5 ] += (az * rz + 2 * ce) * rx * rx * rz;
            gtoz[6 ] += az * rx * ry * ry * ry;
            gtoz[7 ] += (az * rz +     ce) * rx * ry * ry;
            gtoz[8 ] += (az * rz + 2 * ce) * rx * ry * rz;
            gtoz[9 ] += (az * rz + 3 * ce) * rx * rz * rz;
            gtoz[10] += az * ry * ry * ry * ry;
            gtoz[11] += (az * ry * rz +     ce) * ry * ry;
            gtoz[12] += (az * ry * rz + 2 * ce) * ry * rz;
            gtoz[13] += (az * ry * rz + 3 * ce) * rz * rz;
            gtoz[14] += (az * rz * rz + 4 * ce) * rz * rz;
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
        double ax = ce_2a * rx;
        double ay = ce_2a * ry;
        double az = ce_2a * rz;
        switch (li) {
        case 0:
            gto[0] += ce;
            gto[1] += ax;
            gto[2] += ay;
            gto[3] += az;
            break;
        case 1: {
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
            gto[15] += (ax * rx + 4 * ce) * rx * rx * rx;
            gto[16] += (ax * rx + 3 * ce) * rx * rx * ry;
            gto[17] += (ax * rx + 3 * ce) * rx * rx * rz;
            gto[18] += (ax * rx + 2 * ce) * rx * ry * ry;
            gto[19] += (ax * rx + 2 * ce) * rx * ry * rz;
            gto[20] += (ax * rx + 2 * ce) * rx * rz * rz;
            gto[21] += (ax * rx +     ce) * ry * ry * ry;
            gto[22] += (ax * rx +     ce) * ry * ry * rz;
            gto[23] += (ax * rx +     ce) * ry * rz * rz;
            gto[24] += (ax * rx +     ce) * rz * rz * rz;
            gto[25] += ax * ry * ry * ry * ry;
            gto[26] += ax * ry * ry * ry * rz;
            gto[27] += ax * ry * ry * rz * rz;
            gto[28] += ax * ry * rz * rz * rz;
            gto[29] += ax * rz * rz * rz * rz;
            gto[30] += ay * rx * rx * rx * rx;
            gto[31] += (ay * ry +     ce) * rx * rx * rx;
            gto[32] += ay * rx * rx * rx * rz;
            gto[33] += (ay * ry + 2 * ce) * rx * rx * ry;
            gto[34] += (ay * ry +     ce) * rx * rx * rz;
            gto[35] += ay * rx * rx * rz * rz;
            gto[36] += (ay * ry + 3 * ce) * rx * ry * ry;
            gto[37] += (ay * ry + 2 * ce) * rx * ry * rz;
            gto[38] += (ay * ry +     ce) * rx * rz * rz;
            gto[39] += ay * rx * rz * rz * rz;
            gto[40] += (ay * ry + 4 * ce) * ry * ry * ry;
            gto[41] += (ay * ry + 3 * ce) * ry * ry * rz;
            gto[42] += (ay * ry + 2 * ce) * ry * rz * rz;
            gto[43] += (ay * ry +     ce) * rz * rz * rz;
            gto[44] += ay * rz * rz * rz * rz;
            gto[45] += az * rx * rx * rx * rx;
            gto[46] += az * rx * rx * rx * ry;
            gto[47] += (az * rz +     ce) * rx * rx * rx;
            gto[48] += az * rx * rx * ry * ry;
            gto[49] += (az * rz +     ce) * rx * rx * ry;
            gto[50] += (az * rz + 2 * ce) * rx * rx * rz;
            gto[51] += az * rx * ry * ry * ry;
            gto[52] += (az * rz +     ce) * rx * ry * ry;
            gto[53] += (az * rz + 2 * ce) * rx * ry * rz;
            gto[54] += (az * rz + 3 * ce) * rx * rz * rz;
            gto[55] += az * ry * ry * ry * ry;
            gto[56] += (az * ry * rz +     ce) * ry * ry;
            gto[57] += (az * ry * rz + 2 * ce) * ry * rz;
            gto[58] += (az * ry * rz + 3 * ce) * rz * rz;
            gto[59] += (az * rz * rz + 4 * ce) * rz * rz;
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
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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

__global__
static void _cart_deriv0_strain_tensor_kernel(
        double *out, PBCIntEnvVars envs, double *grids,
        size_t ngrids, int nao, double *rcut)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
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
    double xi = gridx[grid_id];
    double yi = gridy[grid_id];
    double zi = gridz[grid_id];
    double cell0_Rx = ri[0];
    double cell0_Ry = ri[1];
    double cell0_Rz = ri[2];
    constexpr int n_cart_max = (LMAX+1)*(LMAX+2)/2;
    double gto[n_cart_max];
    double ao[90];
    for (int n = 0; n < 90; ++n) {
        ao[n] = 0;
    }
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;

    int nimgs = envs.nimgs;
    for (int img = 0; img < nimgs; ++img) {
        double ce = 0;
        double ce_2a = 0;
        double Rx = img_coords[img*3+0] + cell0_Rx;
        double Ry = img_coords[img*3+1] + cell0_Ry;
        double Rz = img_coords[img*3+2] + cell0_Rz;
        double rx = xi - Rx;
        double ry = yi - Ry;
        double rz = zi - Rz;
        double rr = rx * rx + ry * ry + rz * rz;
        if (rr > rrcutoff) continue;
        for (int ip = 0; ip < nprim; ++ip) {
            double ai = expi[ip];
            double c_exp = ci[ip] * exp(-ai * rr);
            ce += c_exp;
            ce_2a -= c_exp * ai * 2;
        }
        if (fabs(ce) < 1e-18) continue;
        double ax = ce_2a * rx;
        double ay = ce_2a * ry;
        double az = ce_2a * rz;
        switch (li) {
        case 0:
            ao[0] -= ax * Rx;
            ao[1] -= ax * Ry;
            ao[2] -= ax * Rz;
            ao[3] -= ay * Rx;
            ao[4] -= ay * Ry;
            ao[5] -= ay * Rz;
            ao[6] -= az * Rx;
            ao[7] -= az * Ry;
            ao[8] -= az * Rz;
            break;
        case 1: {
            gto[0] = ax * rx + ce;
            gto[1] = ax * ry;
            gto[2] = ax * rz;
            gto[3] = ay * rx;
            gto[4] = ay * ry + ce;
            gto[5] = ay * rz;
            gto[6] = az * rx;
            gto[7] = az * ry;
            gto[8] = az * rz + ce;
            for (int n = 0; n < 3; n++) {
                ao[0+9*n] -= gto[0*3+n] * Rx;
                ao[1+9*n] -= gto[0*3+n] * Ry;
                ao[2+9*n] -= gto[0*3+n] * Rz;
                ao[3+9*n] -= gto[1*3+n] * Rx;
                ao[4+9*n] -= gto[1*3+n] * Ry;
                ao[5+9*n] -= gto[1*3+n] * Rz;
                ao[6+9*n] -= gto[2*3+n] * Rx;
                ao[7+9*n] -= gto[2*3+n] * Ry;
                ao[8+9*n] -= gto[2*3+n] * Rz;
            }
        } break;
        case 2: {
            gto[0] = (ax * rx + 2 * ce) * rx;
            gto[1] = (ax * rx +     ce) * ry;
            gto[2] = (ax * rx +     ce) * rz;
            gto[3] = ax * ry * ry;
            gto[4] = ax * ry * rz;
            gto[5] = ax * rz * rz;
            for (int n = 0; n < 6; n++) {
                ao[0+9*n] -= gto[n] * Rx;
                ao[1+9*n] -= gto[n] * Ry;
                ao[2+9*n] -= gto[n] * Rz;
            }
            gto[0] = ay * rx * rx;
            gto[1] = (ay * ry +     ce) * rx;
            gto[2] = ay * rx * rz;
            gto[3] = (ay * ry + 2 * ce) * ry;
            gto[4] = (ay * ry +     ce) * rz;
            gto[5] = ay * rz * rz;
            for (int n = 0; n < 6; n++) {
                ao[3+9*n] -= gto[n] * Rx;
                ao[4+9*n] -= gto[n] * Ry;
                ao[5+9*n] -= gto[n] * Rz;
            }
            gto[0] = az * rx * rx;
            gto[1] = az * rx * ry;
            gto[2] = (az * rz +     ce) * rx;
            gto[3] = az * ry * ry;
            gto[4] = (az * rz +     ce) * ry;
            gto[5] = (az * rz + 2 * ce) * rz;
            for (int n = 0; n < 6; n++) {
                ao[6+9*n] -= gto[n] * Rx;
                ao[7+9*n] -= gto[n] * Ry;
                ao[8+9*n] -= gto[n] * Rz;
            }
        } break;
        case 3: {
            gto[0] = (ax * rx + 3 * ce) * rx * rx;
            gto[1] = (ax * rx + 2 * ce) * rx * ry;
            gto[2] = (ax * rx + 2 * ce) * rx * rz;
            gto[3] = (ax * rx +     ce) * ry * ry;
            gto[4] = (ax * rx +     ce) * ry * rz;
            gto[5] = (ax * rx +     ce) * rz * rz;
            gto[6] = ax * ry * ry * ry;
            gto[7] = ax * ry * ry * rz;
            gto[8] = ax * ry * rz * rz;
            gto[9] = ax * rz * rz * rz;
            for (int n = 0; n < 10; n++) {
                ao[0+9*n] -= gto[n] * Rx;
                ao[1+9*n] -= gto[n] * Ry;
                ao[2+9*n] -= gto[n] * Rz;
            }
            gto[0] = ay * rx * rx * rx;
            gto[1] = (ay * ry +     ce) * rx * rx;
            gto[2] = ay * rx * rx * rz;
            gto[3] = (ay * ry + 2 * ce) * rx * ry;
            gto[4] = (ay * ry +     ce) * rx * rz;
            gto[5] = ay * rx * rz * rz;
            gto[6] = (ay * ry + 3 * ce) * ry * ry;
            gto[7] = (ay * ry + 2 * ce) * ry * rz;
            gto[8] = (ay * ry +     ce) * rz * rz;
            gto[9] = ay * rz * rz * rz;
            for (int n = 0; n < 10; n++) {
                ao[3+9*n] -= gto[n] * Rx;
                ao[4+9*n] -= gto[n] * Ry;
                ao[5+9*n] -= gto[n] * Rz;
            }
            gto[0] = az * rx * rx * rx;
            gto[1] = az * rx * rx * ry;
            gto[2] = (az * rz +     ce) * rx * rx;
            gto[3] = az * rx * ry * ry;
            gto[4] = (az * rz +     ce) * rx * ry;
            gto[5] = (az * rz + 2 * ce) * rx * rz;
            gto[6] = az * ry * ry * ry;
            gto[7] = (az * rz +     ce) * ry * ry;
            gto[8] = (az * rz + 2 * ce) * ry * rz;
            gto[9] = (az * rz + 3 * ce) * rz * rz;
            for (int n = 0; n < 10; n++) {
                ao[6+9*n] -= gto[n] * Rx;
                ao[7+9*n] -= gto[n] * Ry;
                ao[8+9*n] -= gto[n] * Rz;
            }
        } break;
        case 4: {
            gto[0 ] = (ax * rx + 4 * ce) * rx * rx * rx;
            gto[1 ] = (ax * rx + 3 * ce) * rx * rx * ry;
            gto[2 ] = (ax * rx + 3 * ce) * rx * rx * rz;
            gto[3 ] = (ax * rx + 2 * ce) * rx * ry * ry;
            gto[4 ] = (ax * rx + 2 * ce) * rx * ry * rz;
            gto[5 ] = (ax * rx + 2 * ce) * rx * rz * rz;
            gto[6 ] = (ax * rx +     ce) * ry * ry * ry;
            gto[7 ] = (ax * rx +     ce) * ry * ry * rz;
            gto[8 ] = (ax * rx +     ce) * ry * rz * rz;
            gto[9 ] = (ax * rx +     ce) * rz * rz * rz;
            gto[10] = ax * ry * ry * ry * ry;
            gto[11] = ax * ry * ry * ry * rz;
            gto[12] = ax * ry * ry * rz * rz;
            gto[13] = ax * ry * rz * rz * rz;
            gto[14] = ax * rz * rz * rz * rz;
            for (int n = 0; n < 15; n++) {
                ao[0+6*n] -= gto[n] * Rx;
                ao[1+6*n] -= gto[n] * Ry;
                ao[2+6*n] -= gto[n] * Rz;
            }
            gto[0 ] = ay * rx * rx * rx * rx;
            gto[1 ] = (ay * ry +     ce) * rx * rx * rx;
            gto[2 ] = ay * rx * rx * rx * rz;
            gto[3 ] = (ay * ry + 2 * ce) * rx * rx * ry;
            gto[4 ] = (ay * ry +     ce) * rx * rx * rz;
            gto[5 ] = ay * rx * rx * rz * rz;
            gto[6 ] = (ay * ry + 3 * ce) * rx * ry * ry;
            gto[7 ] = (ay * ry + 2 * ce) * rx * ry * rz;
            gto[8 ] = (ay * ry +     ce) * rx * rz * rz;
            gto[9 ] = ay * rx * rz * rz * rz;
            gto[10] = (ay * ry + 4 * ce) * ry * ry * ry;
            gto[11] = (ay * ry + 3 * ce) * ry * ry * rz;
            gto[12] = (ay * ry + 2 * ce) * ry * rz * rz;
            gto[13] = (ay * ry +     ce) * rz * rz * rz;
            gto[14] = ay * rz * rz * rz * rz;
            for (int n = 0; n < 15; n++) {
                ao[3+6*n] -= gto[n] * Rx;
                ao[4+6*n] -= gto[n] * Ry;
                ao[5+6*n] -= gto[n] * Rz;
            }
        } }
    }
    if (li < 4) {
        int *ao_loc = envs.ao_loc;
        int nf = (li + 1) * (li + 2) / 2;
        out += ao_loc[bas_id] * ngrids + grid_id;
        size_t naog = nao * ngrids;
        for (int n = 0; n < 10; ++n) {
            if (n >= nf) break;
            for (int x = 0; x < 9; ++x) {
                out[x*naog+n*ngrids] = ao[n*9+x];
            }
        }
    } else {
        int *ao_loc = envs.ao_loc;
        int nf = (li + 1) * (li + 2) / 2;
        out += ao_loc[bas_id] * ngrids + grid_id;
        size_t naog = nao * ngrids;
        for (int n = 0; n < 15; ++n) {
            if (n >= nf) break;
            for (int x = 0; x < 6; ++x) {
                out[x*naog+n*ngrids] = ao[n*6+x];
            }
        }
        out += 6 * naog; // To process zx, zy, zz

        for (int n = 0; n < 45; ++n) {
            ao[n] = 0;
        }
        for (int img = 0; img < nimgs; ++img) {
            double ce = 0;
            double ce_2a = 0;
            double Rx = img_coords[img*3+0] + cell0_Rx;
            double Ry = img_coords[img*3+1] + cell0_Ry;
            double Rz = img_coords[img*3+2] + cell0_Rz;
            double rx = xi - Rx;
            double ry = yi - Ry;
            double rz = zi - Rz;
            double rr = rx * rx + ry * ry + rz * rz;
            if (rr > rrcutoff) continue;
            for (int ip = 0; ip < nprim; ++ip) {
                double ai = expi[ip];
                double c_exp = ci[ip] * exp(-ai * rr);
                ce += c_exp;
                ce_2a -= c_exp * ai * 2;
            }
            if (fabs(ce) < 1e-18) continue;
            double az = ce_2a * rz;
            gto[0 ] = az * rx * rx * rx * rx;
            gto[1 ] = az * rx * rx * rx * ry;
            gto[2 ] = (az * rz +     ce) * rx * rx * rx;
            gto[3 ] = az * rx * rx * ry * ry;
            gto[4 ] = (az * rz +     ce) * rx * rx * ry;
            gto[5 ] = (az * rz + 2 * ce) * rx * rx * rz;
            gto[6 ] = az * rx * ry * ry * ry;
            gto[7 ] = (az * rz +     ce) * rx * ry * ry;
            gto[8 ] = (az * rz + 2 * ce) * rx * ry * rz;
            gto[9 ] = (az * rz + 3 * ce) * rx * rz * rz;
            gto[10] = az * ry * ry * ry * ry;
            gto[11] = (az * ry * rz +     ce) * ry * ry;
            gto[12] = (az * ry * rz + 2 * ce) * ry * rz;
            gto[13] = (az * ry * rz + 3 * ce) * rz * rz;
            gto[14] = (az * rz * rz + 4 * ce) * rz * rz;
            for (int n = 0; n < 15; n++) {
                ao[0+3*n] -= gto[n] * Rx;
                ao[1+3*n] -= gto[n] * Ry;
                ao[2+3*n] -= gto[n] * Rz;
            }
        }
    }
}

__global__
static void _cart_deriv1_strain_tensor_kernel(
        double *out, PBCIntEnvVars envs, double *grids,
        size_t ngrids, int nao, double *rcut)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int grid_id = item.get_global_id(1);
    int bas_id = item.get_group(0);
#else
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bas_id = blockIdx.y;
#endif
    if (grid_id >= ngrids) {
        return;
    }
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF+bas_id*BAS_SLOTS];
    double *gridx = grids;
    double *gridy = grids + ngrids;
    double *gridz = grids + ngrids * 2;
    double xi = gridx[grid_id];
    double yi = gridy[grid_id];
    double zi = gridz[grid_id];
    double _rcut = rcut[bas_id % envs.cell0_nbas];
    double rrcutoff = _rcut * _rcut;
    int *ao_loc = envs.ao_loc;
    out += ao_loc[bas_id] * ngrids + grid_id;
    int nimgs = envs.nimgs;

    switch (li) {
    case 0: _eval_cart_deriv1_strain_tensor<0>(out, img_coords, env,
                    xi, yi, zi, rrcutoff, bas, nimgs, nao, ngrids);
            break;
    case 1: _eval_cart_deriv1_strain_tensor<1>(out, img_coords, env,
                    xi, yi, zi, rrcutoff, bas, nimgs, nao, ngrids);
            break;
    case 2: _eval_cart_deriv1_strain_tensor<2>(out, img_coords, env,
                    xi, yi, zi, rrcutoff, bas, nimgs, nao, ngrids);
            break;
    case 3: _eval_cart_deriv1_strain_tensor<3>(out, img_coords, env,
                    xi, yi, zi, rrcutoff, bas, nimgs, nao, ngrids);
            break;
    case 4: _eval_cart_deriv1_strain_tensor<4>(out, img_coords, env,
                    xi, yi, zi, rrcutoff, bas, nimgs, nao, ngrids);
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

    #ifdef USE_SYCL
    sycl::range<2> thread(1, threads);
    sycl::range<2> blocks(nbas, (ngrids+ngrids_per_block-1)/ngrids_per_block);
    auto dev_envs = *envs;
    switch (deriv) {
    case 0:
      if (cart) {
        sycl_get_queue()->parallel_for<class _cart_deriv0_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_deriv0_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
      } else {
        sycl_get_queue()->parallel_for<class _sph_deriv0_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _sph_deriv0_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
      }
      break;
    case 1:
      if (cart) {
        sycl_get_queue()->parallel_for<class _cart_deriv1_kernel1_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_deriv1_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
      } else {
        sycl_get_queue()->parallel_for<class _sph_deriv1_kernel1_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _sph_deriv1_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
      }
      break;
    case 2:
      if (cart) {
        sycl_get_queue()->parallel_for<class _cart_deriv1_kernel2_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_deriv1_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
        sycl_get_queue()->parallel_for<class _cart_ip2_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_ip2_kernel(out+4*nao*ngrids, dev_envs, grids, ngrids, nao, rcut); });
      } else {
        sycl_get_queue()->parallel_for<class _sph_deriv1_kernel2_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _sph_deriv1_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
        sycl_get_queue()->parallel_for<class _sph_ip2_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _sph_ip2_kernel(out+4*nao*ngrids, dev_envs, grids, ngrids, nao, rcut); });
      }
      break;
    default:
      fprintf(stderr, "PBCeval_gto deriv = %d not supported\n", deriv);
      return 1;
    }
    #else
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
        fprintf(stderr, "PBCeval_gto deriv = %d not supported\n", deriv);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCeval_gto: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif    
    return 0;
}

int PBCeval_gto_strain_tensor(double *out, PBCIntEnvVars *envs,
                      double *grids, int ngrids, int nao, int nbas,
                      int deriv, int cart, double *rcut)
{
    if (!cart) {
        fprintf(stderr, "PBCeval_gto_strain_tensor does not support spherical GTOs\n");
        return 1;
    }
    constexpr int ngrids_per_block = THREADS;
    int threads = ngrids_per_block;
    #ifdef USE_SYCL
    sycl::range<2> thread(1, threads);
    sycl::range<2> blocks(nbas, (ngrids+ngrids_per_block-1)/ngrids_per_block);
    auto dev_envs = *envs;
    switch (deriv) {
    case 0:
        sycl_get_queue()->parallel_for<class _cart_deriv0_strain_tensor_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_deriv0_strain_tensor_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
        break;
    case 1:
        sycl_get_queue()->parallel_for<class _cart_deriv1_strain_tensor_kernel_sycl>(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) { _cart_deriv1_strain_tensor_kernel(out, dev_envs, grids, ngrids, nao, rcut); });
        break;
    default:
        fprintf(stderr, "PBCeval_gto_strain_tensor deriv = %d not supported\n", deriv);
        return 1;
    }
    #else
    dim3 blocks((ngrids+ngrids_per_block-1)/ngrids_per_block, nbas);
    switch (deriv) {
    case 0:
        _cart_deriv0_strain_tensor_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        break;
    case 1:
        _cart_deriv1_strain_tensor_kernel<<<blocks, threads>>>(out, *envs, grids, ngrids, nao, rcut);
        break;
    default:
        fprintf(stderr, "PBCeval_gto_strain_tensor deriv = %d not supported\n", deriv);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCeval_gto_strain_tensor: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif    
    return 0;
}
}
