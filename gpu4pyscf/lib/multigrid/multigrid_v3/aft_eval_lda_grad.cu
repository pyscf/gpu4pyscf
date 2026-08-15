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

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void orth_lda_grad_kernel(double *out, double *dm, cuDoubleComplex *vxcG,
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
    __shared__ double gx[NGV_PER_BLOCK*3*2*(LMAX+2)*LMAX1];
    __shared__ double swap[NGV_PER_BLOCK*3*2*(LMAX+LMAX+2)];
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
        constexpr int stride_j = stride_i * (LMAX+2);
        for (int n = thread_id; n < stride_j*(lj+1); n += THREADS) {
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
                vrr_hrr(gx, swap, addrR, li+1, lj, stride_j, a2, xjxi, aj_aij, xi,
                        kx, theta_rr);
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
                cuDoubleComplex val = vxcG[addr];
                vG_R[n] = val.x;
                vG_I[n] = val.y;
            }

            double ai2 = ai * -2;
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
                double zR = gxR[addrz] * dm_fac;
                double zI = gxI[addrz] * dm_fac;
                double f1yR, f1yI;
                double f1zR, f1zI;
                dI_gx(gxR, addry, stride_i, iy, ai2, f1yR, f1yI);
                dI_gx(gxR, addry, stride_i, iz, ai2, f1zR, f1zI);
                f1zR *= dm_fac;
                f1zI *= dm_fac;
                double yzR, yzI;
                double YzR, YzI;
                double yZR, yZI;
                multiply(yR, yI, zR, zI, yzR, yzI);
                multiply(f1yR, f1yI, zR, zI, YzR, YzI);
                multiply(yR, yI, f1zR, f1zI, yZR, yZI);
#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    int x_in_Gv_base = mesh_start[0] + x;
                    if (x_in_Gv_base >= mesh_x) break;
                    int addr = addrx + x;
                    double xR = gxR[addr];
                    double xI = gxI[addr];
                    double xyzR, xyzI;
                    multiply(xR, xI, yzR, yzI, xyzR, xyzI);
                    double gout0I = xyzR * vG_I[n] + xyzI * vG_R[n];

                    multiply(xR, xI, YzR, YzI, xyzR, xyzI);
                    double gouty = xyzR * vG_R[n] - xyzI * vG_I[n];
                    v_iy += gouty;
                    v_jy -= gout0I * ky + gouty;

                    multiply(xR, xI, yZR, yZI, xyzR, xyzI);
                    double goutz = xyzR * vG_R[n] - xyzI * vG_I[n];
                    v_iz += goutz;
                    v_jz -= gout0I * kz + goutz;

                    double f1xR, f1xI;
                    dI_gx(gxR, addr, stride_i, ix, ai2, f1xR, f1xI);
                    multiply(f1xR, f1xI, yzR, yzI, xyzR, xyzI);
                    double goutx = xyzR * vG_R[n] - xyzI * vG_I[n];
                    v_ix += goutx;
                    // (\nabla i|j) + (i|\nabla j) + -iG*(ij,G) = 0
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
int orth_aft_lda_grad(double *out, double *dm,
                      cuDoubleComplex *vxcG, cuDoubleComplex *placeholder,
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
    orth_lda_grad_kernel<<<ntile_batch*npair, THREADS>>>(
        out, dm, vxcG, *envs, bas_ij_idx, G_bases, L_bases,
        mesh_cum, nimgs_cum, npair, ntiles_x, ntiles_y, ntiles_z, factor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in orth_lda_grad_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
