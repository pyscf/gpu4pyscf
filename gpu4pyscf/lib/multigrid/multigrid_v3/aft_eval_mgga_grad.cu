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

__global__ static
void orth_mgga_grad_kernel(double *out, double *dm,
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
    __shared__ double gx[NGV_PER_BLOCK*3*2*(LMAX+3)*(LMAX+2)];
    __shared__ int mesh_start[3];
    __shared__ double dm_cache[NCART_MAX*NCART_MAX];
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

    double v_ix = 0;
    double v_iy = 0;
    double v_iz = 0;
    double v_jx = 0;
    double v_jy = 0;
    double v_jz = 0;

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
        constexpr int stride_j = stride_i * (LMAX+3);
        for (int n = thread_id; n < stride_j*(lj+2); n += THREADS) {
            gx[n] = 0;
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
                double xpa = xjxi * aj_aij;
                double xij = xpa + xi;
                double kR = kx * xij;
                double g00R;
                double g00I;
                sincos(-kR, &g00I, &g00R);
                double Kab = exp(-theta_rr);
                g00R *= Kab;
                g00I *= Kab;
                double RpaR = xpa;
                double RpaI = -a2 * kx;
                switch (li*LMAX1+lj) {
                case (0*LMAX1+0): vrr_hrr<2,1>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (1*LMAX1+0): vrr_hrr<3,1>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (1*LMAX1+1): vrr_hrr<3,2>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (2*LMAX1+0): vrr_hrr<4,1>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (2*LMAX1+1): vrr_hrr<4,2>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (2*LMAX1+2): vrr_hrr<4,3>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (3*LMAX1+0): vrr_hrr<5,1>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (3*LMAX1+1): vrr_hrr<5,2>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (3*LMAX1+2): vrr_hrr<5,3>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (3*LMAX1+3): vrr_hrr<5,4>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (4*LMAX1+0): vrr_hrr<6,1>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (4*LMAX1+1): vrr_hrr<6,2>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (4*LMAX1+2): vrr_hrr<6,3>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (4*LMAX1+3): vrr_hrr<6,4>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                case (4*LMAX1+4): vrr_hrr<6,5>(gx, addrR, stride_j, a2, xjxi, RpaR, RpaI, g00R, g00I); break;
                }
            }
        }
        __syncthreads();

        int y_in_tile = thread_id / NGV_PER_BLOCK;
        int z_in_tile = Gv_id;
        int y = mesh_start[1] + y_in_tile;
        int z = mesh_start[2] + z_in_tile;
        if (y < mesh_y && z < mesh_z) {
            double vrho_R[DENSITY_WIDTH];
            double vrho_I[DENSITY_WIDTH];
            double vtau_R[DENSITY_WIDTH];
            double vtau_I[DENSITY_WIDTH];
#pragma unroll
            for (int n = 0; n < DENSITY_WIDTH; ++n) {
                int x = mesh_start[0] + n;
                if (x >= mesh_x) break;
                size_t addr = (x * mesh_y + y) * (size_t)mesh_z + z;
                cuDoubleComplex val = vrhoG[addr];
                vrho_R[n] = val.x;
                vrho_I[n] = val.y;
                val = vtauG[addr];
                vtau_R[n] = val.x / 2;
                vtau_I[n] = val.y / 2;
            }

            double ky = G_bases[mesh_cum[1] + y];
            double kz = G_bases[mesh_cum[2] + z];
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
                double yR = gxR[addry];
                double yI = gxI[addry];
                double zR = gxR[addrz];
                double zI = gxI[addrz];
                // yz_0 =  f0y * f0z 
                // yz_2 =  f2y * f0z + f0y * f2z
                // yz_10 = f1y * f0z 
                // yz_12 = f3y * f0z + f1y * f2z
                // yz_01 = f0y * f1z 
                // yz_21 = f0y * f3z + f2y * f1z
                double yzR00, yzI00;
                multiply(yR, yI, zR, zI, yzR00, yzI00);

                double ai2 = ai * -2;
                double aj2 = aj * -2;
                double f1yR, f1yI;
                dI_gx(gxR, addry, stride_i, iy, ai2, f1yR, f1yI);
                double yzR10, yzI10;
                multiply(f1yR, f1yI, zR, zI, yzR10, yzI10);

                double f1zR, f1zI;
                dI_gx(gxR, addry, stride_i, iz, ai2, f1zR, f1zI);
                double yzR01, yzI01;
                multiply(yR, yI, f1zR, f1zI, yzR01, yzI01);

                double f2yR, f2yI;
                dIdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, f2yR, f2yI);
                double yzR20, yzI20;
                multiply(f2yR, f2yI, zR, zI, yzR20, yzI20);
                double yzR21, yzI21;
                multiply(f2yR, f2yI, f1zR, f1zI, yzR21, yzI21);

                double f2zR, f2zI;
                dIdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, f2zR, f2zI);
                double tmpR, tmpI;
                multiply(yR, yI, f2zR, f2zI, tmpR, tmpI);
                yzR20 += tmpR;
                yzI20 += tmpI;
                double yzR12, yzI12;
                multiply(f1yR, f1yI, f2zR, f2zI, yzR12, yzI12);

                double f3yR, f3yI;
                d2IdJ_gx(gxR, addry, stride_i, stride_j, iy, jy, ai2, aj2, f3yR, f3yI);
                multiply(f3yR, f3yI, zR, zI, tmpR, tmpI);
                yzR21 += tmpR;
                yzI21 += tmpI;

                double f3zR, f3zI;
                d2IdJ_gx(gxR, addrz, stride_i, stride_j, iz, jz, ai2, aj2, f3zR, f3zI);
                multiply(yR, yI, f3zR, f3zI, tmpR, tmpI);
                yzR12 += tmpR;
                yzI12 += tmpI;

                yzR00 *= dm_fac;
                yzR20 *= dm_fac;
                yzR01 *= dm_fac;
                yzR10 *= dm_fac;
                yzR21 *= dm_fac;
                yzR12 *= dm_fac;
                yzI00 *= dm_fac;
                yzI20 *= dm_fac;
                yzI01 *= dm_fac;
                yzI10 *= dm_fac;
                yzI21 *= dm_fac;
                yzI12 *= dm_fac;

#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    int x_in_Gv_base = mesh_start[0] + x;
                    if (x_in_Gv_base >= mesh_x) break;
                    int addr = addrx + x;
                    // gout0 = f2x * f0y * f0z + f0x * f2y * f0z + f0x * f0y * f2z;
                    // goutx = f3x * f0y * f0z + f1x * f2y * f0z + f1x * f0y * f2z;
                    // gouty = f2x * f1y * f0z + f0x * f3y * f0z + f0x * f1y * f2z;
                    // goutz = f2x * f0y * f1z + f0x * f2y * f1z + f0x * f0y * f3z;
                    double xR = gxR[addr];
                    double xI = gxI[addr];
                    double xyzR, xyzI, tmpR, tmpI;
                    multiply(xR, xI, yzR20, yzI20, xyzR, xyzI);
                    double f2xR, f2xI;
                    dIdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, f2xR, f2xI);
                    multiply(f2xR, f2xI, yzR00, yzI00, tmpR, tmpI);
                    xyzR += tmpR;
                    xyzI += tmpI;
                    double gout0I = xyzR * vtau_I[n] + xyzI * vtau_R[n];
                    multiply(xR, xI, yzR00, yzI00, xyzR, xyzI);
                    gout0I += xyzR * vrho_I[n] + xyzI * vrho_R[n];

                    multiply(f2xR, f2xI, yzR10, yzI10, xyzR, xyzI);
                    multiply(xR, xI, yzR12, yzI12, tmpR, tmpI);
                    xyzR += tmpR;
                    xyzI += tmpI;
                    double gouty = xyzR * vtau_R[n] - xyzI * vtau_I[n];
                    multiply(xR, xI, yzR10, yzI10, xyzR, xyzI);
                    gouty += xyzR * vrho_R[n] - xyzI * vrho_I[n];
                    v_iy += gouty;
                    v_jy -= gout0I * ky + gouty;

                    multiply(f2xR, f2xI, yzR01, yzI01, xyzR, xyzI);
                    multiply(xR, xI, yzR21, yzI21, tmpR, tmpI);
                    xyzR += tmpR;
                    xyzI += tmpI;
                    double goutz = xyzR * vtau_R[n] - xyzI * vtau_I[n];
                    multiply(xR, xI, yzR01, yzI01, xyzR, xyzI);
                    goutz += xyzR * vrho_R[n] - xyzI * vrho_I[n];
                    v_iz += goutz;
                    v_jz -= gout0I * kz + goutz;

                    double f1xR, f1xI;
                    dI_gx(gxR, addr, stride_i, ix, ai2, f1xR, f1xI);
                    multiply(f1xR, f1xI, yzR00, yzI00, xyzR, xyzI);
                    double goutx = xyzR * vrho_R[n] - xyzI * vrho_I[n];
                    multiply(f1xR, f1xI, yzR20, yzI20, xyzR, xyzI);
                    double f3xR, f3xI;
                    d2IdJ_gx(gxR, addr, stride_i, stride_j, ix, jx, ai2, aj2, f3xR, f3xI);
                    multiply(f3xR, f3xI, yzR00, yzI00, tmpR, tmpI);
                    xyzR += tmpR;
                    xyzI += tmpI;
                    goutx += xyzR * vtau_R[n] - xyzI * vtau_I[n];
                    v_ix += goutx;
                    double kx = G_bases[mesh_cum[0] + x_in_Gv_base];
                    v_jx -= gout0I * kx + goutx;
                }
            } }
        }
    }

    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        v_ix += __shfl_down_sync(0xffffffff, v_ix, offset);
        v_iy += __shfl_down_sync(0xffffffff, v_iy, offset);
        v_iz += __shfl_down_sync(0xffffffff, v_iz, offset);
        v_jx += __shfl_down_sync(0xffffffff, v_jx, offset);
        v_jy += __shfl_down_sync(0xffffffff, v_jy, offset);
        v_jz += __shfl_down_sync(0xffffffff, v_jz, offset);
    }
    int lane = thread_id % WARP_SIZE;
    int ish_cell0 = ish;
    int bvk_cell_id = jsh / nbas;
    int jsh_cell0 = jsh - nbas * bvk_cell_id;
    int ia = bas[ish_cell0*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh_cell0*BAS_SLOTS+ATOM_OF];
    if (lane == 0) {
        atomicAdd(out+ia*3+0, v_ix);
        atomicAdd(out+ia*3+1, v_iy);
        atomicAdd(out+ia*3+2, v_iz);
        atomicAdd(out+ja*3+0, v_jx);
        atomicAdd(out+ja*3+1, v_jy);
        atomicAdd(out+ja*3+2, v_jz);
    }
}

extern "C" {
int orth_aft_mgga_grad(double *out, double *dm,
                       cuDoubleComplex *vrhoG, cuDoubleComplex *vtauG,
                       PBCIntEnvVars *envs, int64_t *bas_ij_idx,
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
    orth_mgga_grad_kernel<<<ntile_batch*npair, THREADS>>>(
        out, dm, vrhoG, vtauG, *envs, bas_ij_idx, G_bases, L_bases,
        mesh_cum, nimgs_cum, npair, ntiles_x, ntiles_y, ntiles_z, factor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in orth_mgga_grad_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
