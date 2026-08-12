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

#define THREADS         256
#define NGV_PER_BLOCK   16
#define DENSITY_WIDTH   16
#define REMOTE_THRESHOLD 50
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787

__global__ static
void orth_ft_tau_dm_kernel(double *densityR, double *densityI, double *tauR, double *tauI,
                           double *dm, PBCIntEnvVars envs, int *shl_pair_offsets,
                           int64_t *bas_ij_idx, double *G_bases, double *L_bases,
                           int *mesh_cum, int *nimgs_cum, int ntiles, double factor)
{
    int thread_id = threadIdx.x;
    int x_id = thread_id / NGV_PER_BLOCK;
    int Gv_id = thread_id % NGV_PER_BLOCK;
    int sp_block_id = blockIdx.x / ntiles;
    int tile_id = blockIdx.x % ntiles;
    __shared__ double gx[NGV_PER_BLOCK*3*2*(LMAX1+1)*(LMAX1+1)];
    __shared__ double swap[NGV_PER_BLOCK*3*2*(LMAX+LMAX+3)];
    __shared__ int mesh_start[3];
    __shared__ int ri, rj;
    __shared__ uint32_t ij_offset;
    __shared__ double fac, ai, aj;

    int *bas = envs.bas;
    int nbas = envs.nbas;
    double *env = envs.env;

    int mesh_x = mesh_cum[1] - mesh_cum[0];
    int mesh_y = mesh_cum[2] - mesh_cum[1];
    int mesh_z = mesh_cum[3] - mesh_cum[2];

    if (thread_id == 0) {
        int ntiles_y = (mesh_y + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
        int ntiles_z = (mesh_z + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
        int tile_z = tile_id % ntiles_z;
        int tile_xy = tile_id / ntiles_z;
        int tile_y = tile_xy % ntiles_y;
        int tile_x = tile_xy / ntiles_y;
        mesh_start[0] = tile_x * NGV_PER_BLOCK;
        mesh_start[1] = tile_y * NGV_PER_BLOCK;
        mesh_start[2] = tile_z * NGV_PER_BLOCK;
    }

    double rho_R[DENSITY_WIDTH];
    double rho_I[DENSITY_WIDTH];
    double tau_R[DENSITY_WIDTH];
    double tau_I[DENSITY_WIDTH];
#pragma unroll
    for (int n = 0; n < DENSITY_WIDTH; ++n) {
        rho_R[n] = 0.;
        rho_I[n] = 0.;
        tau_R[n] = 0.;
        tau_I[n] = 0.;
    }

    int shl_pair0 = shl_pair_offsets[sp_block_id];
    int shl_pair1 = shl_pair_offsets[sp_block_id+1];
    for (int pair_idx = shl_pair0; pair_idx < shl_pair1; pair_idx++) {
        __syncthreads();
        int64_t bas_ij = bas_ij_idx[pair_idx];
        int ish = bas_ij / NBAS_MAX;
        int jsh = bas_ij % NBAS_MAX;
        int li = bas[ish*BAS_SLOTS+ANG_OF];
        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
        if (thread_id == 0) {
            int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
            int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
            ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
            aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
            double aij = ai + aj;
            ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            fac = OVERLAP_FAC * env[ci] * env[cj] / (aij * sqrt(aij)) * factor;
            int ish_cell0 = ish;
            int bvk_cell_id = jsh / nbas;
            int jsh_cell0 = jsh - nbas * bvk_cell_id;
            if (ish_cell0 == jsh_cell0) {
                fac *= .5;
            }
            int i0 = envs.ao_loc[ish_cell0];
            int j0 = envs.ao_loc[jsh_cell0];
            uint32_t nao = envs.ao_loc[nbas];
            ij_offset = bvk_cell_id * nao * nao + i0 * nao + j0;
        }

        constexpr int stride_i = NGV_PER_BLOCK * 6;
        constexpr int stride_j = stride_i * (LMAX+2);
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
            int lij = li + lj + 2;
            int addrR = x_id * NGV_PER_BLOCK*2 + Gv_id;
            int addrI = addrR + NGV_PER_BLOCK;
            for (int img = nimgs_cum[x_id]; img < nimgs_cum[x_id+1]; ++img) {
                double Lx = L_bases[img];
                double xi = env[ri+x_id];
                double xjxi = env[rj+x_id] + Lx - xi;
                double theta_rr = theta_ij * xjxi * xjxi + .5*a2 * kx * kx;
                if (theta_rr > REMOTE_THRESHOLD) continue;
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
                double RpaR = xpa;
                double RpaI = -a2 * kx;
                s1xR = RpaR * s0xR - RpaI * s0xI;
                s1xI = RpaR * s0xI + RpaI * s0xR;
                swap[addrR+stride_i] = s1xR;
                swap[addrI+stride_i] = s1xI;
                gx[addrR+stride_i] += s1xR;
                gx[addrI+stride_i] += s1xI;
                for (int i = 2; i <= lij; i++) {
                    double ia2 = (i-1) * a2;
                    s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                    s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                    swap[addrR+i*stride_i] = s2xR;
                    swap[addrI+i*stride_i] = s2xI;
                    if (i <= li + 1) {
                        gx[addrR+i*stride_i] += s2xR;
                        gx[addrI+i*stride_i] += s2xI;
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
                        if (i <= li + 1) {
                            int ij = i * stride_i + j * stride_j;
                            gx[addrR+ij] += s2xR;
                            gx[addrI+ij] += s2xI;
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
        if (mesh_start[1] + y_in_tile < mesh_y && mesh_start[2] + z_in_tile < mesh_z) {
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int idx_i = lex_xyz_offset(li);
            int idx_j = lex_xyz_offset(lj);
            uint32_t nao = envs.ao_loc[nbas];
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
                double dm_fac = dm[ij_offset + i*nao+j] * fac;
                double *gxR = gx;
                double *gxI = gxR + NGV_PER_BLOCK;
                double yR = gxR[addry];
                double yI = gxI[addry];
                double zR = gxR[addrz];
                double zI = gxI[addrz];
                double yzR, yzI;
                multiply(yR, yI, zR, zI, yzR, yzI);
                yzR *= dm_fac;
                yzI *= dm_fac;

                double ai2 = ai * -2;
                double aj2 = aj * -2;
                double f3yR = ai2 * gxR[addry+stride_i+stride_j];
                double f3yI = ai2 * gxR[addry+stride_i+stride_j+NGV_PER_BLOCK];
                if (iy > 0) {
                    f3yR += iy * gxR[addry-stride_i+stride_j];
                    f3yI += iy * gxR[addry-stride_i+stride_j+NGV_PER_BLOCK];
                }
                f3yR *= aj2;
                f3yI *= aj2;
                if (jy > 0) {
                    double fyR = ai2 * gxR[addry+stride_i-stride_j];
                    double fyI = ai2 * gxR[addry+stride_i-stride_j+NGV_PER_BLOCK];
                    if (iy > 0) {
                        fyR += iy * gxR[addry-stride_i-stride_j];
                        fyI += iy * gxR[addry-stride_i-stride_j+NGV_PER_BLOCK];
                    }
                    f3yR += jy * fyR;
                    f3yI += jy * fyI;
                }
                double YZR, YZI;
                multiply(f3yR, f3yI, zR, zI, YZR, YZI);

                double f3zR = ai2 * gxR[addrz+stride_i+stride_j];
                double f3zI = ai2 * gxR[addrz+stride_i+stride_j+NGV_PER_BLOCK];
                if (iz > 0) {
                    f3zR += iz * gxR[addrz-stride_i+stride_j];
                    f3zI += iz * gxR[addrz-stride_i+stride_j+NGV_PER_BLOCK];
                }
                f3zR *= aj2;
                f3zI *= aj2;
                if (jz > 0) {
                    double fzR = ai2 * gxR[addrz+stride_i-stride_j];
                    double fzI = ai2 * gxR[addrz+stride_i-stride_j+NGV_PER_BLOCK];
                    if (iz > 0) {
                        fzR += iz * gxR[addrz-stride_i-stride_j];
                        fzI += iz * gxR[addrz-stride_i-stride_j+NGV_PER_BLOCK];
                    }
                    f3zR += jz * fzR;
                    f3zI += jz * fzI;
                }
                double tmpR, tmpI;
                multiply(yR, yI, f3zR, f3zI, tmpR, tmpI);
                YZR = (YZR + tmpR) * dm_fac;
                YZI = (YZI + tmpI) * dm_fac;

#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    if (mesh_start[0] + x >= mesh_x) break;
                    int addr = addrx + x;
                    double xR = gxR[addr];
                    double xI = gxI[addr];
                    double xyzR, xyzI;
                    multiply(xR, xI, yzR, yzI, xyzR, xyzI);
                    rho_R[n] += xyzR;
                    rho_I[n] += xyzI;

                    multiply(xR, xI, YZR, YZI, xyzR, xyzI);
                    tau_R[n] += xyzR;
                    tau_I[n] += xyzI;

                    double f3xR = ai2 * gxR[addr+stride_i+stride_j];
                    double f3xI = ai2 * gxR[addr+stride_i+stride_j+NGV_PER_BLOCK];
                    if (ix > 0) {
                        f3xR += ix * gxR[addr-stride_i+stride_j];
                        f3xI += ix * gxR[addr-stride_i+stride_j+NGV_PER_BLOCK];
                    }
                    f3xR *= aj2;
                    f3xI *= aj2;
                    if (jx > 0) {
                        double fxR = ai2 * gxR[addr+stride_i-stride_j];
                        double fxI = ai2 * gxR[addr+stride_i-stride_j+NGV_PER_BLOCK];
                        if (ix > 0) {
                            fxR += ix * gxR[addr-stride_i-stride_j];
                            fxI += ix * gxR[addr-stride_i-stride_j+NGV_PER_BLOCK];
                        }
                        f3xR += jx * fxR;
                        f3xI += jx * fxI;
                    }
                    multiply(f3xR, f3xI, yzR, yzI, xyzR, xyzI);
                    tau_R[n] += xyzR;
                    tau_I[n] += xyzI;
                }
            } }
        }
    }

    int Gx0 = mesh_start[0];
    int Gy0 = mesh_start[1];
    int Gz0 = mesh_start[2];
    int y_in_tile = thread_id / NGV_PER_BLOCK;
    int z_in_tile = Gv_id;
    int y = Gy0 + y_in_tile;
    int z = Gz0 + z_in_tile;
    if (y < mesh_y && z < mesh_z) {
#pragma unroll
        for (int n = 0; n < DENSITY_WIDTH; ++n) {
            int x = Gx0 + n;
            if (x >= mesh_x) break;
            int abc_idx = (x * mesh_y + y) * mesh_z + z;
            atomicAdd(densityR+abc_idx, rho_R[n]);
            atomicAdd(densityI+abc_idx, rho_I[n]);
            atomicAdd(tauR+abc_idx, tau_R[n]/2);
            atomicAdd(tauI+abc_idx, tau_I[n]/2);
        }
    }
}

extern "C" {
int orth_contract_ft_tau_dm(double *densityR, double *densityI,
                            double *tauR, double *tauI, double *dm,
                            PBCIntEnvVars *envs, int *shl_pair_offsets,
                            int64_t *bas_ij_idx, double *G_bases, double *L_bases,
                            int *mesh_cum, int *nimgs_cum, int *mesh,
                            int nbatches_shl_pair, double factor)
{
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int ntiles_x = (mesh_x + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_y = (mesh_y + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_z = (mesh_z + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles = ntiles_x * ntiles_y * ntiles_z;
    orth_ft_tau_dm_kernel<<<ntiles*nbatches_shl_pair, THREADS>>>(
        densityR, densityI, tauR, tauI, dm, *envs, shl_pair_offsets, bas_ij_idx, G_bases, L_bases,
        mesh_cum, nimgs_cum, ntiles, factor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in orth_ft_tau_dm_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
