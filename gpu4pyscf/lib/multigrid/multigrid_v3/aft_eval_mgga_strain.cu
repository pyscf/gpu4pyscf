/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"
#include "constant_objects.cuh"
#include "utils.cuh"
#include "aft_recursion.cuh"

#define WARP_SIZE       32
#define WARPS           8
#define THREADS         256
#define NGV_PER_BLOCK   16
#define DENSITY_WIDTH   16
#define TILES_PER_BATCH 64
#define REMOTE_THRESHOLD 50
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787

__device__ __forceinline__
void mul_add(double aR, double aI, double bR, double bI,
             double cR, double cI, double dR, double dI,
             double &outR, double &outI)
{
    outR = aR * bR - aI * bI;
    outI = aR * bI + aI * bR;
    outR += cR * dR - cI * dI;
    outI += cR * dI + cI * dR;
}

__global__ static
void orth_mgga_strain_kernel(double *out, double *dm,
                             cuDoubleComplex *vrhoG, cuDoubleComplex *vtauG,
                             PBCIntEnvVars envs, int64_t *bas_ij_idx,
                             double *G_bases, double *L_bases,
                             int *mesh_cum, int *nimgs_cum,
                             int npair, int ntiles_x, int ntiles_y, int ntiles_z,
                             double factor)
{
    int thread_id = threadIdx.x;
    int x_id = thread_id / NGV_PER_BLOCK;
    int Gv_id = thread_id % NGV_PER_BLOCK;
    __shared__ int tile_batch;
    int pair_id = blockIdx.x % npair;
    if (thread_id == 0) {
        tile_batch = blockIdx.x / npair;
    }
    extern __shared__ double shared_memory[];
    __shared__ int mesh_start[3];
    __shared__ int ri, rj, li, lj;
    __shared__ double ai, aj;

    int mesh_x = mesh_cum[1] - mesh_cum[0];
    int mesh_y = mesh_cum[2] - mesh_cum[1];
    int mesh_z = mesh_cum[3] - mesh_cum[2];
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    if (thread_id == 0) {
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    }
    __syncthreads();
    double *gx = shared_memory;
    // Xgx = xjxi * gx
    double *Xgx = gx + NGV_PER_BLOCK*3*2*(li+1)*(lj+1);
    double *swap = shared_memory + NGV_PER_BLOCK*3*2*(li+1)*(lj+1) * 2;
    double *dm_cache = swap + NGV_PER_BLOCK*3*2*(li+lj+2);

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    for (int n = thread_id; n < nfi * nfj; n += THREADS) {
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double aij = ai + aj;
        double fac = OVERLAP_FAC * env[ci] * env[cj] / (aij * sqrt(aij)) * factor;
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        }
        uint32_t nao = envs.ao_loc[nbas];
        int i0 = envs.ao_loc[ish_cell0];
        int j0 = envs.ao_loc[jsh_cell0];
        int i = n * c_div_nf[lj];
        int j = n - nfj * i;
        dm_cache[n] = dm[bvk_cell_id*nao*nao + (i0+i)*nao + j0+j] * fac;
    }

    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_xz = 0;
    double sigma_yx = 0;
    double sigma_yy = 0;
    double sigma_yz = 0;
    double sigma_zx = 0;
    double sigma_zy = 0;
    double sigma_zz = 0;

    int ntiles = ntiles_x * ntiles_y * ntiles_z;
    int tile0 = tile_batch * TILES_PER_BATCH;
    int tile1 = min(tile0 + TILES_PER_BATCH, ntiles);
    for (int tile_id = tile0; tile_id < tile1; tile_id++) {
        __syncthreads();
        int tile_z = tile_id % ntiles_z;
        int tile_xy = tile_id / ntiles_z;
        int tile_y = tile_xy % ntiles_y;
        int tile_x = tile_xy / ntiles_y;
        if (thread_id == 0) {
            mesh_start[0] = tile_x * NGV_PER_BLOCK;
            mesh_start[1] = tile_y * NGV_PER_BLOCK;
            mesh_start[2] = tile_z * NGV_PER_BLOCK;
        }

        constexpr int stride_i = NGV_PER_BLOCK * 6;
        int stride_j = stride_i * (li + 3);
        for (int n = thread_id; n < stride_j*(lj+2); n += THREADS) {
            gx[n] = 0;
            Xgx[n] = 0;
        }
        __syncthreads();

        if (x_id < 3) {
            double aij = ai + aj;
            double a2 = .5 / aij;
            double aj_aij = aj * 2 * a2;
            double theta_ij = ai * aj_aij;
            double kx = 0;
            int _Gv_id = mesh_cum[x_id] + mesh_start[x_id] + Gv_id;
            if (_Gv_id < mesh_cum[x_id+1]) {
                kx = G_bases[_Gv_id];
            }
            int addrR = x_id * NGV_PER_BLOCK*2 + Gv_id;
            for (int img = nimgs_cum[x_id]; img < nimgs_cum[x_id+1]; ++img) {
                double Lx = L_bases[img];
                double xi = env[ri+x_id];
                double xjxi = env[rj+x_id] + Lx - xi;
                double theta_rr = theta_ij * xjxi * xjxi + .5*a2 * kx * kx;
                if (theta_rr > REMOTE_THRESHOLD) continue;
                int addrI = addrR + NGV_PER_BLOCK;
                int lij = li + lj + 3;
                double xpa = xjxi * aj_aij;
                double xij = xpa + xi;
                double kR = kx * xij;
                double s0xR, s1xR, s2xR;
                double s0xI, s1xI, s2xI;
                sincos(-kR, &s0xI, &s0xR);
                double Kab = exp(-theta_rr);
                s0xR *= Kab;
                s0xI *= Kab;
                swap[addrR] = s0xR;
                swap[addrI] = s0xI;
                gx[addrR] += s0xR;
                gx[addrI] += s0xI;
                Xgx[addrR] += xjxi * s0xR;
                Xgx[addrI] += xjxi * s0xI;
                double RpaR = xpa;
                double RpaI = -a2 * kx;
                s1xR = RpaR * s0xR - RpaI * s0xI;
                s1xI = RpaR * s0xI + RpaI * s0xR;
                swap[addrR+stride_i] = s1xR;
                swap[addrI+stride_i] = s1xI;
                gx[addrR+stride_i] += s1xR;
                gx[addrI+stride_i] += s1xI;
                Xgx[addrR+stride_i] += xjxi * s1xR;
                Xgx[addrI+stride_i] += xjxi * s1xI;
                for (int i = 2; i <= lij; i++) {
                    double ia2 = (i-1) * a2;
                    s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                    s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                    swap[addrR+i*stride_i] = s2xR;
                    swap[addrI+i*stride_i] = s2xI;
                    if (i <= li+2) {
                        int i_ = i * stride_i;
                        gx[addrR+i_] += s2xR;
                        gx[addrI+i_] += s2xI;
                        Xgx[addrR+i_] += xjxi * s2xR;
                        Xgx[addrI+i_] += xjxi * s2xI;
                    }
                    s0xR = s1xR;
                    s0xI = s1xI;
                    s1xR = s2xR;
                    s1xI = s2xI;
                }
                for (int j = 1; j <= lj+1; ++j) {
                    int i = lij - j;
                    s1xR = swap[addrR+(i+1)*stride_i];
                    s1xI = swap[addrI+(i+1)*stride_i];
                    for (; i >= 0; --i) {
                        s0xR = swap[addrR+i*stride_i];
                        s0xI = swap[addrI+i*stride_i];
                        s2xR = s1xR - xjxi * s0xR;
                        s2xI = s1xI - xjxi * s0xI;
                        swap[addrR+i*stride_i] = s2xR;
                        swap[addrI+i*stride_i] = s2xI;
                        if (i <= li+2) {
                            int ij = i * stride_i + j * stride_j;
                            gx[addrR+ij] += s2xR;
                            gx[addrI+ij] += s2xI;
                            Xgx[addrR+ij] += xjxi * s2xR;
                            Xgx[addrI+ij] += xjxi * s2xI;
                        }
                        s1xR = s0xR;
                        s1xI = s0xI;
                    }
                }
            }
        }
        __syncthreads();

        int y_in_tile = thread_id / NGV_PER_BLOCK;
        int z_in_tile = Gv_id;
        int y = mesh_start[1] + y_in_tile;
        int z = mesh_start[2] + z_in_tile;
        if (y < mesh_y && z < mesh_z) {
            double vG_R[DENSITY_WIDTH];
            double vG_I[DENSITY_WIDTH];
#pragma unroll
            for (int n = 0; n < DENSITY_WIDTH; ++n) {
                int x = mesh_start[0] + n;
                if (x >= mesh_x) break;
                size_t addr = (x * mesh_y + y) * (size_t)mesh_z + z;
                cuDoubleComplex val = vrhoG[addr];
                vG_R[n] = val.x;
                vG_I[n] = val.y;
            }

            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int idx_i = lex_xyz_offset(li);
            int idx_j = lex_xyz_offset(lj);
            for (int i = 0; i < nfi; ++i) {
            for (int j = 0; j < nfj; ++j) {
                int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                int addrx = ix*stride_i + jx*stride_j;
                int addry = iy*stride_i + jy*stride_j + NGV_PER_BLOCK*2 + y_in_tile;
                int addrz = iz*stride_i + jz*stride_j + NGV_PER_BLOCK*4 + z_in_tile;
                double dm_fac = dm_cache[i*nfj+j];
                double *gxR = gx;
                double *gxI = gxR + NGV_PER_BLOCK;
                double yR0 = gxR[addry];
                double yI0 = gxI[addry];
                double zR0 = gxR[addrz] * dm_fac;
                double zI0 = gxI[addrz] * dm_fac;
                double YR0 = Xgx[addry              ];
                double YI0 = Xgx[addry+NGV_PER_BLOCK];
                double ZR0 = Xgx[addrz              ] * dm_fac;
                double ZI0 = Xgx[addrz+NGV_PER_BLOCK] * dm_fac;
                double yzR00, yzI00; multiply(yR0, yI0, zR0, zI0, yzR00, yzI00);
                double YzR00, YzI00; multiply(YR0, YI0, zR0, zI0, YzR00, YzI00);
                double yZR00, yZI00; multiply(yR0, yI0, ZR0, ZI0, yZR00, yZI00);

                double ai2 = ai * -2;
                double yR1, yI1; dI_gx(gxR, addry, stride_i, iy, ai2, yR1, yI1);
                double YR1, YI1; dI_gx(Xgx, addry, stride_i, iy, ai2, YR1, YI1);
                double zR1, zI1; dI_gx(gxR, addrz, stride_i, iz, ai2, zR1, zI1);
                double ZR1, ZI1; dI_gx(Xgx, addrz, stride_i, iz, ai2, ZR1, ZI1);
                zR1 *= dm_fac;
                zI1 *= dm_fac;
                ZR1 *= dm_fac;
                ZI1 *= dm_fac;
                double yzR10, yzI10; multiply(yR1, yI1, zR0, zI0, yzR10, yzI10);
                double YzR10, YzI10; multiply(YR1, YI1, zR0, zI0, YzR10, YzI10);
                double yZR10, yZI10; multiply(yR1, yI1, ZR0, ZI0, yZR10, yZI10);
                double yzR01, yzI01; multiply(yR0, yI0, zR1, zI1, yzR01, yzI01);
                double YzR01, YzI01; multiply(YR0, YI0, zR1, zI1, YzR01, YzI01);
                double yZR01, yZI01; multiply(yR0, yI0, ZR1, ZI1, yZR01, yZI01);
#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    int x_in_Gv_base = mesh_start[0] + x;
                    if (x_in_Gv_base >= mesh_x) break;
                    int addr = addrx + x;
                    double xR0 = gxR[addr];
                    double xI0 = gxI[addr];
                    double XR0 = Xgx[addr              ];
                    double XI0 = Xgx[addr+NGV_PER_BLOCK];
                    double xR1, xI1; dI_gx(gxR, addr, stride_i, ix, ai2, xR1, xI1);
                    double XR1, XI1; dI_gx(Xgx, addr, stride_i, ix, ai2, XR1, XI1);
                    double xyzR, xyzI;
                    multiply(XR1, XI1, yzR00, yzI00, xyzR, xyzI); sigma_xx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR1, xI1, YzR00, YzI00, xyzR, xyzI); sigma_xy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR1, xI1, yZR00, yZI00, xyzR, xyzI); sigma_xz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(XR0, XI0, yzR10, yzI10, xyzR, xyzI); sigma_yx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR0, xI0, YzR10, YzI10, xyzR, xyzI); sigma_yy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR0, xI0, yZR10, yZI10, xyzR, xyzI); sigma_yz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(XR0, XI0, yzR01, yzI01, xyzR, xyzI); sigma_zx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR0, xI0, YzR01, YzI01, xyzR, xyzI); sigma_zy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    multiply(xR0, xI0, yZR01, yZI01, xyzR, xyzI); sigma_zz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                }
            } }

#pragma unroll
            for (int n = 0; n < DENSITY_WIDTH; ++n) {
                int x = mesh_start[0] + n;
                if (x >= mesh_x) break;
                size_t addr = (x * mesh_y + y) * (size_t)mesh_z + z;
                cuDoubleComplex val = vtauG[addr];
                vG_R[n] = val.x / 2;
                vG_I[n] = val.y / 2;
            }
//            for (int i = 0; i < nfi; ++i) {
//            for (int j = 0; j < nfj; ++j) {
//                int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
//                int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
//                int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
//                int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
//                int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
//                int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
//                int addrx = ix*stride_i + jx*stride_j;
//                int addry = iy*stride_i + jy*stride_j + NGV_PER_BLOCK*2 + y_in_tile;
//                int addrz = iz*stride_i + jz*stride_j + NGV_PER_BLOCK*4 + z_in_tile;
//                double dm_fac = dm_cache[i*nfj+j];
//                double *gxR = gx;
//                double *gxI = gxR + NGV_PER_BLOCK;
//                double yR0 = gxR[addry];
//                double yI0 = gxI[addry];
//                double zR0 = gxR[addrz] * dm_fac;
//                double zI0 = gxI[addrz] * dm_fac;
//                double YR0 = Xgx[addry              ];
//                double YI0 = Xgx[addry+NGV_PER_BLOCK];
//                double ZR0 = Xgx[addrz              ] * dm_fac;
//                double ZI0 = Xgx[addrz+NGV_PER_BLOCK] * dm_fac;
//                double yzR00, yzI00; multiply(yR0, yI0, zR0, zI0, yzR00, yzI00);
//                double YzR00, YzI00; multiply(YR0, YI0, zR0, zI0, YzR00, YzI00);
//                double yZR00, yZI00; multiply(yR0, yI0, ZR0, ZI0, yZR00, yZI00);
//
//                double ai2 = ai * -2;
//                double aj2 = aj * -2;
//                double yR1, yI1; dI_gx(gxR, addry, stride_i, iy, ai2, yR1, yI1);
//                double YR1, YI1; dI_gx(Xgx, addry, stride_i, iy, ai2, YR1, YI1);
//                double zR1, zI1; dI_gx(gxR, addrz, stride_i, iz, ai2, zR1, zI1);
//                double ZR1, ZI1; dI_gx(Xgx, addrz, stride_i, iz, ai2, ZR1, ZI1);
//                zR1 *= dm_fac;
//                zI1 *= dm_fac;
//                ZR1 *= dm_fac;
//                ZI1 *= dm_fac;
//                double yzR10, yzI10; multiply(yR1, yI1, zR0, zI0, yzR10, yzI10);
//                double yzR01, yzI01; multiply(yR0, yI0, zR1, zI1, yzR01, yzI01);
//                double YzR10, YzI10; multiply(YR1, YI1, zR0, zI0, YzR10, YzI10);
//                double yZR10, yZI10; multiply(yR1, yI1, ZR0, ZI0, yZR10, yZI10);
//                double YzR01, YzI01; multiply(YR0, YI0, zR1, zI1, YzR01, YzI01);
//                double yZR01, yZI01; multiply(yR0, yI0, ZR1, ZI1, yZR01, yZI01);
//
//                // yz_00 = f0y * f0z 
//                // yz_20 = f2y * f0z + f0y * f2z
//                // yz_10 = f1y * f0z 
//                // yz_12 = f3y * f0z + f1y * f2z
//                // yz_01 = f0y * f1z 
//                // yz_21 = f0y * f3z + f2y * f1z
//                double yR2, yI2; dIdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR2, yI2);
//                double zR2, zI2; dIdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR2, zI2);
//                zR2 *= dm_fac;
//                zI2 *= dm_fac;
//                double yzR20, yzI20; mul_add(yR2, yI2, zR0, zI0,  yR0, yI0, zR2, zI2, yzR20, yzI20);
//
//                double YR2, YI2; dIdJ_gx(Xgx, addry, stride_i, stride_j, iy, jy, ai2, aj2, YR2, YI2);
//                double ZR2, ZI2; dIdJ_gx(Xgx, addrz, stride_i, stride_j, iz, jz, ai2, aj2, ZR2, ZI2);
//                ZR2 *= dm_fac;
//                ZI2 *= dm_fac;
//                double yZR20, yZI20; mul_add(yR2, yI2, ZR0, ZI0,  yR0, yI0, ZR2, ZI2, yZR20, yZI20);
//                double YzR20, YzI20; mul_add(YR2, YI2, zR0, zI0,  YR0, YI0, zR2, zI2, YzR20, YzI20);
//
//                double yR3, yI3; d2IdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR3, yI3);
//                double zR3, zI3; d2IdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR3, zI3);
//                zR3 *= dm_fac;
//                zI3 *= dm_fac;
//                double yzR21, yzI21; mul_add(yR2, yI2, zR1, zI1,  yR3, yI3, zR0, zI0, yzR21, yzI21);
//                double yzR12, yzI12; mul_add(yR1, yI1, zR2, zI2,  yR0, yI0, zR3, zI3, yzR12, yzI12);
//
//                double YR3, YI3; d2IdJ_gx(Xgx, addry, stride_i, stride_j, iy, jy, ai2, aj2, YR3, YI3);
//                double ZR3, ZI3; d2IdJ_gx(Xgx, addrz, stride_i, stride_j, iz, jz, ai2, aj2, ZR3, ZI3);
//                ZR3 *= dm_fac;
//                ZI3 *= dm_fac;
//                double YzR21, YzI21; mul_add(YR2, YI2, zR1, zI1,  YR3, YI3, zR0, zI0, YzR21, YzI21);
//                double yZR21, yZI21; mul_add(yR2, yI2, ZR1, ZI1,  yR3, yI3, ZR0, ZI0, yZR21, yZI21);
//                double YzR12, YzI12; mul_add(YR1, YI1, zR2, zI2,  YR0, YI0, zR3, zI3, YzR12, YzI12);
//                double yZR12, yZI12; mul_add(yR1, yI1, ZR2, ZI2,  yR0, yI0, ZR3, ZI3, yZR12, yZI12);
//#pragma unroll
//                for (int n = 0; n < DENSITY_WIDTH; ++n) {
//                    int x = n;
//                    int x_in_Gv_base = mesh_start[0] + x;
//                    if (x_in_Gv_base >= mesh_x) break;
//                    int addr = addrx + x;
//                    double xR0 = gxR[addr];
//                    double xI0 = gxI[addr];
//                    double XR0 = Xgx[addr              ];
//                    double XI0 = Xgx[addr+NGV_PER_BLOCK];
//                    double xR2, xI2; dIdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, xR2, xI2);
//                    double XR2, XI2; dIdJ_gx(Xgx, addr, stride_i, stride_j, ix, jx, ai2, aj2, XR2, XI2);
//                    double xyzR, xyzI;
//                    mul_add(XR2, XI2, yzR10, yzI10, XR0, XI0, yzR12, yzI12, xyzR, xyzI); sigma_xx -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(XR2, XI2, yzR01, yzI01, XR0, XI0, yzR21, yzI21, xyzR, xyzI); sigma_yx -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR2, xI2, YzR10, YzI10, xR0, xI0, YzR12, YzI12, xyzR, xyzI); sigma_xy -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR2, xI2, yZR10, yZI10, xR0, xI0, yZR12, yZI12, xyzR, xyzI); sigma_xz -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR2, xI2, YzR01, YzI01, xR0, xI0, YzR21, YzI21, xyzR, xyzI); sigma_yy -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR2, xI2, yZR01, yZI01, xR0, xI0, yZR21, yZI21, xyzR, xyzI); sigma_yz -= xyzR * vG_R[n] - xyzI * vG_I[n];
//
//                    double xR1, xI1; dI_gx(gxR, addr, stride_i, ix, ai2, xR1, xI1);
//                    double XR1, XI1; dI_gx(Xgx, addr, stride_i, ix, ai2, XR1, XI1);
//                    double xR3, xI3; d2IdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, xR3, xI3);
//                    double XR3, XI3; d2IdJ_gx(Xgx, addr, stride_i, stride_j, ix, jx, ai2, aj2, XR3, XI3);
//                    mul_add(XR3, XI3, yzR00, yzI00, XR1, XI1, yzR20, yzI20, xyzR, xyzI); sigma_zx -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR3, xI3, YzR00, YzI00, xR1, xI1, YzR20, YzI20, xyzR, xyzI); sigma_zy -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                    mul_add(xR3, xI3, yZR00, yZI00, xR1, xI1, yZR20, yZI20, xyzR, xyzI); sigma_zz -= xyzR * vG_R[n] - xyzI * vG_I[n];
//                }
//            } }
            for (int i = 0; i < nfi; ++i) {
            for (int j = 0; j < nfj; ++j) {
                int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                int addrx = ix*stride_i + jx*stride_j;
                int addry = iy*stride_i + jy*stride_j + NGV_PER_BLOCK*2 + y_in_tile;
                int addrz = iz*stride_i + jz*stride_j + NGV_PER_BLOCK*4 + z_in_tile;
                double dm_fac = dm_cache[i*nfj+j];
                double *gxR = gx;
                double *gxI = gxR + NGV_PER_BLOCK;
                double yR0 = gxR[addry];
                double yI0 = gxI[addry];
                double zR0 = gxR[addrz] * dm_fac;
                double zI0 = gxI[addrz] * dm_fac;
                double yzR00, yzI00; multiply(yR0, yI0, zR0, zI0, yzR00, yzI00);

                double ai2 = ai * -2;
                double aj2 = aj * -2;
                double yR1, yI1; dI_gx(gxR, addry, stride_i, iy, ai2, yR1, yI1);
                double zR1, zI1; dI_gx(gxR, addrz, stride_i, iz, ai2, zR1, zI1);
                zR1 *= dm_fac;
                zI1 *= dm_fac;
                double yzR10, yzI10; multiply(yR1, yI1, zR0, zI0, yzR10, yzI10);
                double yzR01, yzI01; multiply(yR0, yI0, zR1, zI1, yzR01, yzI01);

                // yz_00 = f0y * f0z 
                // yz_20 = f2y * f0z + f0y * f2z
                // yz_10 = f1y * f0z 
                // yz_12 = f3y * f0z + f1y * f2z
                // yz_01 = f0y * f1z 
                // yz_21 = f0y * f3z + f2y * f1z
                double yR2, yI2; dIdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR2, yI2);
                double zR2, zI2; dIdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR2, zI2);
                zR2 *= dm_fac;
                zI2 *= dm_fac;
                double yzR20, yzI20; mul_add(yR2, yI2, zR0, zI0,  yR0, yI0, zR2, zI2, yzR20, yzI20);

                double yR3, yI3; d2IdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR3, yI3);
                double yzR21, yzI21; mul_add(yR2, yI2, zR1, zI1,  yR3, yI3, zR0, zI0, yzR21, yzI21);

                double zR3, zI3; d2IdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR3, zI3);
                zR3 *= dm_fac;
                zI3 *= dm_fac;
                double yzR12, yzI12; mul_add(yR1, yI1, zR2, zI2,  yR0, yI0, zR3, zI3, yzR12, yzI12);
#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    int x_in_Gv_base = mesh_start[0] + x;
                    if (x_in_Gv_base >= mesh_x) break;
                    int addr = addrx + x;
                    double XR0 = Xgx[addr              ];
                    double XI0 = Xgx[addr+NGV_PER_BLOCK];
                    double XR2, XI2; dIdJ_gx(Xgx, addr, stride_i, stride_j, ix, jx, ai2, aj2, XR2, XI2);
                    double xyzR, xyzI;
                    mul_add(XR2, XI2, yzR10, yzI10, XR0, XI0, yzR12, yzI12, xyzR, xyzI); sigma_xx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    mul_add(XR2, XI2, yzR01, yzI01, XR0, XI0, yzR21, yzI21, xyzR, xyzI); sigma_yx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    double XR1, XI1; dI_gx(Xgx, addr, stride_i, ix, ai2, XR1, XI1);
                    double XR3, XI3; d2IdJ_gx(Xgx, addr, stride_i, stride_j, ix, jx, ai2, aj2, XR3, XI3);
                    mul_add(XR3, XI3, yzR00, yzI00, XR1, XI1, yzR20, yzI20, xyzR, xyzI); sigma_zx -= xyzR * vG_R[n] - xyzI * vG_I[n];
                }
            } }

            for (int i = 0; i < nfi; ++i) {
            for (int j = 0; j < nfj; ++j) {
                int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                int addrx = ix*stride_i + jx*stride_j;
                int addry = iy*stride_i + jy*stride_j + NGV_PER_BLOCK*2 + y_in_tile;
                int addrz = iz*stride_i + jz*stride_j + NGV_PER_BLOCK*4 + z_in_tile;
                double dm_fac = dm_cache[i*nfj+j];
                double *gxR = gx;
                double *gxI = gxR + NGV_PER_BLOCK;
                double yR0 = gxR[addry];
                double yI0 = gxI[addry];
                double zR0 = gxR[addrz] * dm_fac;
                double zI0 = gxI[addrz] * dm_fac;
                double YR0 = Xgx[addry              ];
                double YI0 = Xgx[addry+NGV_PER_BLOCK];
                double ZR0 = Xgx[addrz              ] * dm_fac;
                double ZI0 = Xgx[addrz+NGV_PER_BLOCK] * dm_fac;
                double YzR00, YzI00; multiply(YR0, YI0, zR0, zI0, YzR00, YzI00);
                double yZR00, yZI00; multiply(yR0, yI0, ZR0, ZI0, yZR00, yZI00);

                double ai2 = ai * -2;
                double aj2 = aj * -2;
                double yR1, yI1; dI_gx(gxR, addry, stride_i, iy, ai2, yR1, yI1);
                double YR1, YI1; dI_gx(Xgx, addry, stride_i, iy, ai2, YR1, YI1);
                double zR1, zI1; dI_gx(gxR, addrz, stride_i, iz, ai2, zR1, zI1);
                double ZR1, ZI1; dI_gx(Xgx, addrz, stride_i, iz, ai2, ZR1, ZI1);
                zR1 *= dm_fac;
                zI1 *= dm_fac;
                ZR1 *= dm_fac;
                ZI1 *= dm_fac;
                double YzR10, YzI10; multiply(YR1, YI1, zR0, zI0, YzR10, YzI10);
                double yZR10, yZI10; multiply(yR1, yI1, ZR0, ZI0, yZR10, yZI10);
                double YzR01, YzI01; multiply(YR0, YI0, zR1, zI1, YzR01, YzI01);
                double yZR01, yZI01; multiply(yR0, yI0, ZR1, ZI1, yZR01, yZI01);

                double YR2, YI2; dIdJ_gx(Xgx, addry, stride_i, stride_j, iy, jy, ai2, aj2, YR2, YI2);
                double zR2, zI2; dIdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR2, zI2);
                zR2 *= dm_fac;
                zI2 *= dm_fac;
                double YzR20, YzI20; mul_add(YR2, YI2, zR0, zI0,  YR0, YI0, zR2, zI2, YzR20, YzI20);

                double yR2, yI2; dIdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR2, yI2);
                double ZR2, ZI2; dIdJ_gx(Xgx, addrz, stride_i, stride_j, iz, jz, ai2, aj2, ZR2, ZI2);
                ZR2 *= dm_fac;
                ZI2 *= dm_fac;
                double yZR20, yZI20; mul_add(yR2, yI2, ZR0, ZI0,  yR0, yI0, ZR2, ZI2, yZR20, yZI20);

                double yR3, yI3; d2IdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, yR3, yI3);
                double YR3, YI3; d2IdJ_gx(Xgx, addry, stride_i, stride_j, iy, jy, ai2, aj2, YR3, YI3);
                double YzR21, YzI21; mul_add(YR2, YI2, zR1, zI1,  YR3, YI3, zR0, zI0, YzR21, YzI21);
                double yZR21, yZI21; mul_add(yR2, yI2, ZR1, ZI1,  yR3, yI3, ZR0, ZI0, yZR21, yZI21);

                double zR3, zI3; d2IdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, zR3, zI3);
                zR3 *= dm_fac;
                zI3 *= dm_fac;
                double ZR3, ZI3; d2IdJ_gx(Xgx, addrz, stride_i, stride_j, iz, jz, ai2, aj2, ZR3, ZI3);
                ZR3 *= dm_fac;
                ZI3 *= dm_fac;
                double YzR12, YzI12; mul_add(YR1, YI1, zR2, zI2,  YR0, YI0, zR3, zI3, YzR12, YzI12);
                double yZR12, yZI12; mul_add(yR1, yI1, ZR2, ZI2,  yR0, yI0, ZR3, ZI3, yZR12, yZI12);

#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    int x_in_Gv_base = mesh_start[0] + x;
                    if (x_in_Gv_base >= mesh_x) break;
                    int addr = addrx + x;
                    double xR0 = gxR[addr];
                    double xI0 = gxI[addr];
                    double xR2, xI2; dIdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, xR2, xI2);
                    double xyzR, xyzI;
                    mul_add(xR2, xI2, YzR10, YzI10, xR0, xI0, YzR12, YzI12, xyzR, xyzI); sigma_xy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    mul_add(xR2, xI2, yZR10, yZI10, xR0, xI0, yZR12, yZI12, xyzR, xyzI); sigma_xz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    mul_add(xR2, xI2, YzR01, YzI01, xR0, xI0, YzR21, YzI21, xyzR, xyzI); sigma_yy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    mul_add(xR2, xI2, yZR01, yZI01, xR0, xI0, yZR21, yZI21, xyzR, xyzI); sigma_yz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    double xR1, xI1; dI_gx(gxR, addr, stride_i, ix, ai2, xR1, xI1);
                    double xR3, xI3; d2IdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, xR3, xI3);
                    mul_add(xR3, xI3, YzR00, YzI00, xR1, xI1, YzR20, YzI20, xyzR, xyzI); sigma_zy -= xyzR * vG_R[n] - xyzI * vG_I[n];
                    mul_add(xR3, xI3, yZR00, yZI00, xR1, xI1, yZR20, yZI20, xyzR, xyzI); sigma_zz -= xyzR * vG_R[n] - xyzI * vG_I[n];
                }
            } }
        }
    }

    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        sigma_xx += __shfl_down_sync(0xffffffff, sigma_xx, offset);
        sigma_xy += __shfl_down_sync(0xffffffff, sigma_xy, offset);
        sigma_xz += __shfl_down_sync(0xffffffff, sigma_xz, offset);
        sigma_yx += __shfl_down_sync(0xffffffff, sigma_yx, offset);
        sigma_yy += __shfl_down_sync(0xffffffff, sigma_yy, offset);
        sigma_yz += __shfl_down_sync(0xffffffff, sigma_yz, offset);
        sigma_zx += __shfl_down_sync(0xffffffff, sigma_zx, offset);
        sigma_zy += __shfl_down_sync(0xffffffff, sigma_zy, offset);
        sigma_zz += __shfl_down_sync(0xffffffff, sigma_zz, offset);
    }
    int lane = thread_id % WARP_SIZE;
    if (lane == 0) {
        atomicAdd(out+0, sigma_xx);
        atomicAdd(out+1, sigma_xy);
        atomicAdd(out+2, sigma_xz);
        atomicAdd(out+3, sigma_yx);
        atomicAdd(out+4, sigma_yy);
        atomicAdd(out+5, sigma_yz);
        atomicAdd(out+6, sigma_zx);
        atomicAdd(out+7, sigma_zy);
        atomicAdd(out+8, sigma_zz);
    }
}

extern "C" {
int orth_aft_mgga_strain(double *out, double *dm,
                       cuDoubleComplex *vrhoG, cuDoubleComplex *vtauG,
                       PBCIntEnvVars *envs, int shm_size, int64_t *bas_ij_idx,
                       double *G_bases, double *L_bases, int *mesh_cum,
                       int *nimgs_cum, int *mesh, int npair, double factor)
{
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int ntiles_x = (mesh_x + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_y = (mesh_y + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_z = (mesh_z + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles = ntiles_x * ntiles_y * ntiles_z;
    int ntile_batch = (ntiles + TILES_PER_BATCH-1) / TILES_PER_BATCH;
    orth_mgga_strain_kernel<<<ntile_batch*npair, THREADS, shm_size>>>(
        out, dm, vrhoG, vtauG, *envs, bas_ij_idx, G_bases, L_bases,
        mesh_cum, nimgs_cum, npair, ntiles_x, ntiles_y, ntiles_z, factor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in orth_mgga_strain_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
