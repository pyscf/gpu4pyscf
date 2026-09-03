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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "pbc/recursion.cuh"

#define THREADS         256
#define GOUT_IP_WIDTH   20
#define BLOCK_SIZE      16
#define L_AUX           6
#define L_AUX1          (L_AUX+1)

__global__ static
void pbc_int2c2e_ip1_kernel(double *out, PBCIntEnvVars envs,
                            double omega, double lr_factor, double sr_factor,
                            int *shl_pair_offsets, uint32_t *bas_ij_idx,
                            int *gout_stride_lookup)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int nbas;
    __shared__ int li, lj, nroots, nao, iprim, jprim;
    __shared__ int gout_stride;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        nbas = envs.nbas * envs.bvk_ncells;
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        nroots = (li + lj + 1) / 2 + 1;
        if (omega < 0) {
            nroots *= 2; // omega < 0
        }
        nao = envs.ao_loc[envs.nbas];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int i_1 = nsp_per_block;
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 2;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    double goutx[GOUT_IP_WIDTH];
    double gouty[GOUT_IP_WIDTH];
    double goutz[GOUT_IP_WIDTH];

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
#pragma unroll
        for (int n = 0; n < GOUT_IP_WIDTH; ++n) {
            goutx[n] = 0.;
            gouty[n] = 0.;
            goutz[n] = 0.;
        }
        __syncthreads();
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        if (gout_id == 0) {
            double fac = PI_FAC;
            if (ish == jsh) {
                fac *= .5;
            } else if (ish < jsh) {
                fac = 0;
            }
            gx[gx_len] = fac;
        }
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xpq = ri[0] - (rj[0] + xjL);
                double ypq = ri[1] - (rj[1] + yjL);
                double zpq = ri[2] - (rj[2] + zjL);
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
                Rpq[3*nsp_per_block] = rr;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * -2;
                double aij = ai + aj;
                double theta = ai * aj / aij;
                if (gout_id == 0) {
                    double cicj = ci[ip] * cj[jp];
                    gx[0] = cicj / (ai*aj*sqrt(aij));
                }
                double rr = Rpq[3*nsp_per_block];
                rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                nsp_per_block, gout_stride, gout_id);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    vrr(gx, Rpq, ai, aj, rt, li+1, lj, gout_id, gout_stride, nsp_per_block);
                    if (pair_ij < shl_pair1) {
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int n = 0; n < GOUT_IP_WIDTH; ++n) {
                            uint32_t ij = gout_id + n * gout_stride;
                            if (ij >= nfij) break;
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = idx_i[i*3+0];
                            int iy = idx_i[i*3+1];
                            int iz = idx_i[i*3+2];
                            int jx = idx_j[j*3+0];
                            int jy = idx_j[j*3+1];
                            int jz = idx_j[j*3+2];
                            int addrx = (ix + jx*stride_j) * nsp_per_block;
                            int addry = (iy + jy*stride_j) * nsp_per_block + gx_len;
                            int addrz = (iz + jz*stride_j) * nsp_per_block + gx_len*2;
                            double fx0 = gx[addrx];
                            double fy0 = gx[addry];
                            double fz0 = gx[addrz];
                            double fx1 = ai2 * gx[addrx+i_1];
                            double fy1 = ai2 * gx[addry+i_1];
                            double fz1 = ai2 * gx[addrz+i_1];
                            if (ix > 0) fx1 += ix * gx[addrx-i_1];
                            if (iy > 0) fy1 += iy * gx[addry-i_1];
                            if (iz > 0) fz1 += iz * gx[addrz-i_1];
                            goutx[n] += fx1 * fy0 * fz0;
                            gouty[n] += fx0 * fy1 * fz0;
                            goutz[n] += fx0 * fy0 * fz1;
                        }
                    }
                }
            }
        }
        if (pair_ij < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            size_t nao2 = nao * nao;
            int cell_id = jsh / envs.nbas;
            int jsh_cell0 = jsh - cell_id * envs.nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jsh_cell0];
            double *outx = out + cell_id*nao2*3 + i0 * nao + j0;
            double *outy = outx + nao2;
            double *outz = outx + nao2 * 2;
#pragma unroll
            for (int n = 0; n < GOUT_IP_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij - j * nfi;
                int addr = i * nao + j;
                outx[addr] = goutx[n];
                outy[addr] = gouty[n];
                outz[addr] = goutz[n];
            }
        }
    }
}

extern "C" {
int fill_int2c2e_ip1(double *out, PBCIntEnvVars *envs,
                     double omega, double lr_factor, double sr_factor, int shm_size,
                     int nbatches_shl_pair, int *shl_pair_offsets,
                     uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(pbc_int2c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    pbc_int2c2e_ip1_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, omega, lr_factor, sr_factor,
            shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_ip1 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
