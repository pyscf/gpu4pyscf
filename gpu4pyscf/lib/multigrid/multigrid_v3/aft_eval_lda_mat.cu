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
void orth_lda_mat_kernel(double *out, cuDoubleComplex *vxcG,
                         PBCIntEnvVars envs, int64_t *bas_ij_idx,
                         double *G_bases, double *L_bases,
                         int *mesh_cum, int *nimgs_cum,
                         int npair, int ntiles_x, int ntiles_y, int ntiles_z)
{
    int thread_id = threadIdx.x;
    int x_id = thread_id / NGV_PER_BLOCK;
    int Gv_id = thread_id % NGV_PER_BLOCK;
    __shared__ int tile_batch;
    int pair_id = blockIdx.x % npair;
    if (thread_id == 0) {
        tile_batch = blockIdx.x / npair;
    }
    __shared__ double gx[NGV_PER_BLOCK*3*2*LMAX1*LMAX1];
    __shared__ double swap[NGV_PER_BLOCK*3*2*(LMAX+LMAX+1)];
    __shared__ int mesh_start[3];
    __shared__ double vjR[NCART_MAX*NCART_MAX * WARPS];
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
    for (int n = thread_id; n < nfi * nfj * WARPS; n += THREADS) {
        vjR[n] = 0.;
    }

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
        constexpr int stride_j = stride_i * LMAX1;
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
                vrr_hrr(gx, swap, addrR, li, lj, stride_j, a2, xjxi, aj_aij, xi,
                        kx, theta_rr);
            }
        }
        __syncthreads();

        int y_in_tile = thread_id / NGV_PER_BLOCK;
        int z_in_tile = Gv_id;
        int y = mesh_start[1] + y_in_tile;
        int z = mesh_start[2] + z_in_tile;
        double vG_R[DENSITY_WIDTH];
        double vG_I[DENSITY_WIDTH];
        if (y < mesh_y && z < mesh_z) {
#pragma unroll
            for (int n = 0; n < DENSITY_WIDTH; ++n) {
                int x = mesh_start[0] + n;
                if (x >= mesh_x) break;
                size_t addr = (x * mesh_y + y) * (size_t)mesh_z + z;
                cuDoubleComplex val = vxcG[addr];
                vG_R[n] = val.x;
                vG_I[n] = val.y;
            }
        }

        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int idx_i = lex_xyz_offset(li);
        int idx_j = lex_xyz_offset(lj);
        for (int i = 0; i < nfi; ++i) {
        for (int j = 0; j < nfj; ++j) {
            double s = 0;
            if (y < mesh_y && z < mesh_z) {
                int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                int addrx = ix*stride_i + jx*stride_j;
                int addry = iy*stride_i + jy*stride_j + NGV_PER_BLOCK*2 + y_in_tile;
                int addrz = iz*stride_i + jz*stride_j + NGV_PER_BLOCK*4 + z_in_tile;
                double *gxR = gx;
                double *gxI = gxR + NGV_PER_BLOCK;
                double yR = gxR[addry];
                double yI = gxI[addry];
                double zR = gxR[addrz];
                double zI = gxI[addrz];
                double yzR, yzI;
                multiply(yR, yI, zR, zI, yzR, yzI);
#pragma unroll
                for (int n = 0; n < DENSITY_WIDTH; ++n) {
                    int x = n;
                    if (mesh_start[0] + x >= mesh_x) break;
                    double xR = gxR[addrx+x];
                    double xI = gxI[addrx+x];
                    double xyzR, xyzI;
                    multiply(xR, xI, yzR, yzI, xyzR, xyzI);
                    s += xyzR * vG_R[n] - xyzI * vG_I[n];
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                s += __shfl_down_sync(0xffffffff, s, offset);
            }
            int lane = thread_id % WARP_SIZE;
            int warp = thread_id / WARP_SIZE;
            if (lane == 0) {
                vjR[warp + WARPS*(i*nfj+j)] += s;
            }
        } }
    }

    __syncthreads();
    for (int n = thread_id; n < nfi * nfj; n += THREADS) {
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double aij = ai + aj;
        double fac = OVERLAP_FAC * env[ci] * env[cj] / (aij * sqrt(aij));
        int ish_cell0 = ish;
        int bvk_cell_id = jsh / nbas;
        int jsh_cell0 = jsh - nbas * bvk_cell_id;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        }
        size_t nao = envs.ao_loc[nbas];
        int i0 = envs.ao_loc[ish_cell0];
        int j0 = envs.ao_loc[jsh_cell0];
        int i = n * c_div_nf[lj];
        int j = n - nfj * i;
        double s = 0;
        for (int m = 0; m < WARPS; m++) {
            s += vjR[n*WARPS+m];
        }
        atomicAdd(out + bvk_cell_id*nao*nao + (i0+i)*nao + j0+j, s * fac);
    }
}

//__global__ static
//void monoclinic_aopair_coulG_kernel(double *out, double *coulG_R, double *coulG_I,
//                                    PBCIntEnvVars *envs,
//                                    int *shl_pair_offsets, int64_t *bas_ij_idx,
//                                    double *G_bases, int *mesh_cum, int *nimgs_cum,
//                                    int *mesh, int nbatches_shl_pair)
//{
//}

extern "C" {
int orth_aft_lda_mat(double *out, cuDoubleComplex *vxcG, cuDoubleComplex *placeholder,
                     PBCIntEnvVars *envs, int64_t *bas_ij_idx,
                     double *G_bases, double *L_bases, int *mesh_cum,
                     int *nimgs_cum, int *mesh, int npair)
{
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int ntiles_x = (mesh_x + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_y = (mesh_y + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles_z = (mesh_z + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    int ntiles = ntiles_x * ntiles_y * ntiles_z;
    int ntile_batch = (ntiles + TILES_PER_BATCH-1) / TILES_PER_BATCH;
    orth_lda_mat_kernel<<<ntile_batch*npair, THREADS>>>(
        out, vxcG, *envs, bas_ij_idx, G_bases, L_bases,
        mesh_cum, nimgs_cum, npair, ntiles_x, ntiles_y, ntiles_z);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in orth_lda_mat_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
