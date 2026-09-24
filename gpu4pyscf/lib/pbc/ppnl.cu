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

// Included by overlap.cu to share vrr_hrr and its constants.
// All three radial channels contract each image directly into nuclear and
// strain derivatives, without allocating derivative integral tensors.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"
#include "int3c2e.cuh"
#include "overlap.cuh"

#define GOUT_WIDTH      36
#define GOUT_WIDTH_IP1  18

// <i| |r-A|^2 |j>: raise projector-side powers by two, with A the i center.
__global__ static
void int1e_r2_origi_kernel(double *out, PBCIntEnvVars envs, int *bas_ij_idx,
                           int *shl_pair_offsets, int *gout_stride_lookup,
                           int naoi, int naoj, size_t ij_offset)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, iprim, jprim;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int stride_j = li + 3;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = g + gx_len + sp_id;
    double *gz = g + gx_len * 2 + sp_id;
    double *rjri = g + gx_len * 3 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    if (gout_id == 0) {
        gx[0] = PI_POW_1_5;
        gy[0] = 1.;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            gout[n] = 0.;
        }
        int bas_ij;
        if (pair_ij >= shl_pair1) {
            bas_ij = bas_ij_idx[shl_pair0];
        } else {
            bas_ij = bas_ij_idx[pair_ij];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xjxi = env[rj+0] + xjL - env[ri+0];
                double yjyi = env[rj+1] + yjL - env[ri+1];
                double zjzi = env[rj+2] + zjL - env[ri+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double cicj = env[ci+ip] * env[cj+jp];
                vrr_hrr(gx, rjri, ai, aj, cicj, li+2, lj, gout_id, gout_stride,
                        nsp_per_block);
                if (pair_ij >= shl_pair1) {
                    continue;
                }
                int nsp = nsp_per_block;
                int i_1 = nsp_per_block;
                int stride_j = li + 3;
                int nfi = c_nf[li];
                int nfj = c_nf[lj];
                int nfij = nfi * nfj;
                float div_nfi = c_div_nf[li];
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    uint32_t ij = gout_id + n * gout_stride;
                    if (ij >= nfij) break;
                    uint32_t j = ij * div_nfi;
                    uint32_t i = ij - j * nfi;
                    int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                    int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                    int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                    int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                    int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                    int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                    int addrx = (ix + jx*stride_j) * nsp;
                    int addry = (iy + jy*stride_j) * nsp;
                    int addrz = (iz + jz*stride_j) * nsp;
                    double sx = gx[addrx];
                    double sy = gy[addry];
                    double sz = gz[addrz];
                    double mx = gx[addrx+i_1*2];
                    double my = gy[addry+i_1*2];
                    double mz = gz[addrz+i_1*2];
                    gout[n] += mx*sy*sz + sx*my*sz + sx*sy*mz;
                }
            }
        }

        if (pair_ij < shl_pair1) {
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int nfij = nfi * nfj;
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *out_subblock = out + (cell_id*naoi+i0) * naoj + j0 - ij_offset;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                out_subblock[i*naoj+j] = gout[n];
            }
        }
    }
}

// <i| |r-A|^4 |j>: raise projector-side powers by four, with A the i center.
// r^4 = x^4+y^4+z^4 + 2(x^2*y^2+y^2*z^2+x^2*z^2).
__global__ static
void int1e_r4_origi_kernel(double *out, PBCIntEnvVars envs, int *bas_ij_idx,
                           int *shl_pair_offsets, int *gout_stride_lookup,
                           int naoi, int naoj, size_t ij_offset)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, iprim, jprim;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int stride_j = li + 5;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = g + gx_len + sp_id;
    double *gz = g + gx_len * 2 + sp_id;
    double *rjri = g + gx_len * 3 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    if (gout_id == 0) {
        gx[0] = PI_POW_1_5;
        gy[0] = 1.;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            gout[n] = 0.;
        }
        int bas_ij;
        if (pair_ij >= shl_pair1) {
            bas_ij = bas_ij_idx[shl_pair0];
        } else {
            bas_ij = bas_ij_idx[pair_ij];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xjxi = env[rj+0] + xjL - env[ri+0];
                double yjyi = env[rj+1] + yjL - env[ri+1];
                double zjzi = env[rj+2] + zjL - env[ri+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double cicj = env[ci+ip] * env[cj+jp];
                vrr_hrr(gx, rjri, ai, aj, cicj, li+4, lj, gout_id, gout_stride,
                        nsp_per_block);
                if (pair_ij >= shl_pair1) {
                    continue;
                }
                int nsp = nsp_per_block;
                int i_1 = nsp_per_block;
                int stride_j = li + 5;
                int nfi = c_nf[li];
                int nfj = c_nf[lj];
                int nfij = nfi * nfj;
                float div_nfi = c_div_nf[li];
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    uint32_t ij = gout_id + n * gout_stride;
                    if (ij >= nfij) break;
                    uint32_t j = ij * div_nfi;
                    uint32_t i = ij - j * nfi;
                    int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                    int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                    int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                    int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                    int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                    int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                    int addrx = (ix + jx*stride_j) * nsp;
                    int addry = (iy + jy*stride_j) * nsp;
                    int addrz = (iz + jz*stride_j) * nsp;
                    double Sx0 = gx[addrx+i_1*0];
                    double Sx2 = gx[addrx+i_1*2];
                    double Sx4 = gx[addrx+i_1*4];
                    double Sy0 = gy[addry+i_1*0];
                    double Sy2 = gy[addry+i_1*2];
                    double Sy4 = gy[addry+i_1*4];
                    double Sz0 = gz[addrz+i_1*0];
                    double Sz2 = gz[addrz+i_1*2];
                    double Sz4 = gz[addrz+i_1*4];
                    // r^4 = x^4+y^4+z^4 + 2(x^2*y^2 + y^2*z^2 + x^2*z^2)
                    gout[n] += Sx4*Sy0*Sz0 + Sx0*Sy4*Sz0 + Sx0*Sy0*Sz4
                            + 2.*(Sx2*Sy2*Sz0 + Sx0*Sy2*Sz2 + Sx2*Sy0*Sz2);
                }
            }
        }

        if (pair_ij < shl_pair1) {
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int nfij = nfi * nfj;
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *out_subblock = out + (cell_id*naoi+i0) * naoj + j0 - ij_offset;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                out_subblock[i*naoj+j] = gout[n];
            }
        }
    }
}

// <i| |r-A|^2 nabla_j |j>, with A the projector center.
// Raise i by 2 for the moment and j by one for its spatial derivative.
__global__ static
void int1e_r2_origi_ip2_kernel(double *out, PBCIntEnvVars envs, int *bas_ij_idx,
                               int *shl_pair_offsets, int *gout_stride_lookup,
                               int naoi, int naoj, size_t ij_offset)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, iprim, jprim;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int stride_j = li + 3;
    int g_size = stride_j * (lj + 2);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = g + gx_len + sp_id;
    double *gz = g + gx_len * 2 + sp_id;
    double *rjri = g + gx_len * 3 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    if (gout_id == 0) {
        gx[0] = PI_POW_1_5;
        gy[0] = 1.;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double goutx[GOUT_WIDTH_IP1];
        double gouty[GOUT_WIDTH_IP1];
        double goutz[GOUT_WIDTH_IP1];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
            goutx[n] = 0.;
            gouty[n] = 0.;
            goutz[n] = 0.;
        }
        __syncthreads();
        int bas_ij;
        if (pair_ij >= shl_pair1) {
            bas_ij = bas_ij_idx[shl_pair0];
        } else {
            bas_ij = bas_ij_idx[pair_ij];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xjxi = env[rj+0] + xjL - env[ri+0];
                double yjyi = env[rj+1] + yjL - env[ri+1];
                double zjzi = env[rj+2] + zjL - env[ri+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double cicj = env[ci+ip] * env[cj+jp];
                vrr_hrr(gx, rjri, ai, aj, cicj, li+2, lj+1, gout_id, gout_stride,
                        nsp_per_block);
                if (pair_ij >= shl_pair1) {
                    continue;
                }
                int nsp = nsp_per_block;
                int stride_j = li + 3;
                int j_1 = stride_j * nsp;
                double aj2 = aj * -2;
                float div_nfi = c_div_nf[li];
                int nfi = c_nf[li];
                int nfj = c_nf[lj];
                int nfij = nfi * nfj;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                    uint32_t ij = gout_id + n * gout_stride;
                    if (ij >= nfij) break;
                    uint32_t j = ij * div_nfi;
                    uint32_t i = ij - j * nfi;
                    int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                    int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                    int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                    int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                    int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                    int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                    int addrx = (ix + jx*stride_j) * nsp;
                    int addry = (iy + jy*stride_j) * nsp;
                    int addrz = (iz + jz*stride_j) * nsp;
                    double sx = gx[addrx];
                    double sy = gy[addry];
                    double sz = gz[addrz];
                    double Dsx = aj2*gx[addrx+j_1];
                    double Dsy = aj2*gy[addry+j_1];
                    double Dsz = aj2*gz[addrz+j_1];
                    if (jx > 0) Dsx += jx*gx[addrx-j_1];
                    if (jy > 0) Dsy += jy*gy[addry-j_1];
                    if (jz > 0) Dsz += jz*gz[addrz-j_1];
                    double mx = gx[addrx+2*nsp];
                    double my = gy[addry+2*nsp];
                    double mz = gz[addrz+2*nsp];
                    double Dmx = aj2*gx[addrx+2*nsp+j_1];
                    double Dmy = aj2*gy[addry+2*nsp+j_1];
                    double Dmz = aj2*gz[addrz+2*nsp+j_1];
                    if (jx > 0) Dmx += jx*gx[addrx+2*nsp-j_1];
                    if (jy > 0) Dmy += jy*gy[addry+2*nsp-j_1];
                    if (jz > 0) Dmz += jz*gz[addrz+2*nsp-j_1];
                    goutx[n] += Dmx*sy*sz + Dsx*(my*sz + sy*mz);
                    gouty[n] += Dmy*sx*sz + Dsy*(mx*sz + sx*mz);
                    goutz[n] += Dmz*sx*sy + Dsz*(mx*sy + sx*my);
                }
            }
        }

        if (pair_ij < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            size_t nao2 = naoi * naoj;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *outx = out + cell_id*nao2*3 + i0 * naoj + j0 - ij_offset;
            double *outy = outx + nao2;
            double *outz = outx + nao2 * 2;
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                outx[i*naoj+j] = goutx[n];
                outy[i*naoj+j] = gouty[n];
                outz[i*naoj+j] = goutz[n];
            }
        }
    }
}

// <i| |r-A|^4 nabla_j |j>, with A the projector center.
// Raise i by 4 for the moment and j by one for its spatial derivative.
__global__ static
void int1e_r4_origi_ip2_kernel(double *out, PBCIntEnvVars envs, int *bas_ij_idx,
                               int *shl_pair_offsets, int *gout_stride_lookup,
                               int naoi, int naoj, size_t ij_offset)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, iprim, jprim;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int stride_j = li + 5;
    int g_size = stride_j * (lj + 2);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = g + gx_len + sp_id;
    double *gz = g + gx_len * 2 + sp_id;
    double *rjri = g + gx_len * 3 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    if (gout_id == 0) {
        gx[0] = PI_POW_1_5;
        gy[0] = 1.;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double goutx[GOUT_WIDTH_IP1];
        double gouty[GOUT_WIDTH_IP1];
        double goutz[GOUT_WIDTH_IP1];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
            goutx[n] = 0.;
            gouty[n] = 0.;
            goutz[n] = 0.;
        }
        __syncthreads();
        int bas_ij;
        if (pair_ij >= shl_pair1) {
            bas_ij = bas_ij_idx[shl_pair0];
        } else {
            bas_ij = bas_ij_idx[pair_ij];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xjxi = env[rj+0] + xjL - env[ri+0];
                double yjyi = env[rj+1] + yjL - env[ri+1];
                double zjzi = env[rj+2] + zjL - env[ri+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double cicj = env[ci+ip] * env[cj+jp];
                vrr_hrr(gx, rjri, ai, aj, cicj, li+4, lj+1, gout_id, gout_stride,
                        nsp_per_block);
                if (pair_ij >= shl_pair1) {
                    continue;
                }
                int nsp = nsp_per_block;
                int stride_j = li + 5;
                int j_1 = stride_j * nsp;
                double aj2 = aj * -2;
                float div_nfi = c_div_nf[li];
                int nfi = c_nf[li];
                int nfj = c_nf[lj];
                int nfij = nfi * nfj;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                    uint32_t ij = gout_id + n * gout_stride;
                    if (ij >= nfij) break;
                    uint32_t j = ij * div_nfi;
                    uint32_t i = ij - j * nfi;
                    int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                    int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                    int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                    int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                    int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                    int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                    int addrx = (ix + jx*stride_j) * nsp;
                    int addry = (iy + jy*stride_j) * nsp;
                    int addrz = (iz + jz*stride_j) * nsp;
                    double sx = gx[addrx];
                    double sy = gy[addry];
                    double sz = gz[addrz];
                    double Dsx = aj2*gx[addrx+j_1];
                    double Dsy = aj2*gy[addry+j_1];
                    double Dsz = aj2*gz[addrz+j_1];
                    if (jx > 0) Dsx += jx*gx[addrx-j_1];
                    if (jy > 0) Dsy += jy*gy[addry-j_1];
                    if (jz > 0) Dsz += jz*gz[addrz-j_1];
                    double mx2 = gx[addrx+2*nsp];
                    double my2 = gy[addry+2*nsp];
                    double mz2 = gz[addrz+2*nsp];
                    double Dmx2 = aj2*gx[addrx+2*nsp+j_1];
                    double Dmy2 = aj2*gy[addry+2*nsp+j_1];
                    double Dmz2 = aj2*gz[addrz+2*nsp+j_1];
                    if (jx > 0) Dmx2 += jx*gx[addrx+2*nsp-j_1];
                    if (jy > 0) Dmy2 += jy*gy[addry+2*nsp-j_1];
                    if (jz > 0) Dmz2 += jz*gz[addrz+2*nsp-j_1];
                    double mx4 = gx[addrx+4*nsp];
                    double my4 = gy[addry+4*nsp];
                    double mz4 = gz[addrz+4*nsp];
                    double Dmx4 = aj2*gx[addrx+4*nsp+j_1];
                    double Dmy4 = aj2*gy[addry+4*nsp+j_1];
                    double Dmz4 = aj2*gz[addrz+4*nsp+j_1];
                    if (jx > 0) Dmx4 += jx*gx[addrx+4*nsp-j_1];
                    if (jy > 0) Dmy4 += jy*gy[addry+4*nsp-j_1];
                    if (jz > 0) Dmz4 += jz*gz[addrz+4*nsp-j_1];
                    // r^4 = x^4+y^4+z^4 + 2(x^2*y^2+y^2*z^2+x^2*z^2)
                    goutx[n] += Dmx4*sy*sz + Dsx*my4*sz + Dsx*sy*mz4
                             + 2.*(Dmx2*my2*sz + Dsx*my2*mz2 + Dmx2*sy*mz2);
                    gouty[n] += mx4*Dsy*sz + sx*Dmy4*sz + sx*Dsy*mz4
                             + 2.*(mx2*Dmy2*sz + sx*Dmy2*mz2 + mx2*Dsy*mz2);
                    goutz[n] += mx4*sy*Dsz + sx*my4*Dsz + sx*sy*Dmz4
                             + 2.*(mx2*my2*Dsz + sx*my2*Dmz2 + mx2*sy*Dmz2);
                }
            }
        }

        if (pair_ij < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            size_t nao2 = naoi * naoj;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *outx = out + cell_id*nao2*3 + i0 * naoj + j0 - ij_offset;
            double *outy = outx + nao2;
            double *outz = outx + nao2 * 2;
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                outx[i*naoj+j] = goutx[n];
                outy[i*naoj+j] = gouty[n];
                outz[i*naoj+j] = goutz[n];
            }
        }
    }
}

template<int RADIAL>
static __global__
void ppnl_derivatives_kernel(double *grad, double *sigma, double *dm, PBCIntEnvVars envs,
                             int *shl_pair_offsets, int *bas_ij_idx,
                             int *gout_stride_lookup, int naoi, int naoj)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    int cell0_nbas = envs.cell0_nbas;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, iprim, jprim;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + RADIAL + 1) * (lj + 2);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = g + gx_len + sp_id;
    double *gz = g + gx_len * 2 + sp_id;
    double *rjri = g + gx_len * 3 + sp_id;
    if (gout_id == 0) {
        gx[0] = PI_POW_1_5;
        gy[0] = 1.;
    }
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);

    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_xz = 0;
    double sigma_yx = 0;
    double sigma_yy = 0;
    double sigma_yz = 0;
    double sigma_zx = 0;
    double sigma_zy = 0;
    double sigma_zz = 0;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        __syncthreads();
        int bas_ij = bas_ij_idx[shl_pair0];
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int cell_j = jsh / cell0_nbas;
        int jsh_cell0 = jsh - cell0_nbas * cell_j;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh_cell0] - naoi;
        // Rectangular weights are stored as (image, projector AO, cell AO).
        const double *dm_ij = dm + (size_t)cell_j*naoi*naoj + (size_t)i0*naoj + j0;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double grad_ix = 0;
        double grad_iy = 0;
        double grad_iz = 0;
        for (int img = 0; img < envs.nimgs; img++) {
            double xi = env[ri+0];
            double yi = env[ri+1];
            double zi = env[ri+2];
            double xj = env[rj+0] + img_coords[img*3+0];
            double yj = env[rj+1] + img_coords[img*3+1];
            double zj = env[rj+2] + img_coords[img*3+2];
            __syncthreads();
            if (gout_id == 0) {
                double xjxi = xj - xi;
                double yjyi = yj - yi;
                double zjzi = zj - zi;
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
            }
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double cicj = env[ci+ip] * env[cj+jp];
                vrr_hrr(gx, rjri, ai, aj, cicj, li+RADIAL, lj+1, gout_id, gout_stride,
                        nsp_per_block);
                if (pair_ij >= shl_pair1) {
                    continue;
                }
                int stride_j = li + RADIAL + 1;
                float div_nfi = c_div_nf[li];
                int nfi = c_nf[li];
                int nfj = c_nf[lj];
                int nfij = nfi * nfj;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    uint32_t ij = gout_id + n * gout_stride;
                    if (ij >= nfij) break;
                    uint32_t j = ij * div_nfi;
                    uint32_t i = ij - j * nfi;
                    int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                    int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                    int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                    int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                    int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                    int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                    double v[3] = {0., 0., 0.};
                    // Expand |r-A|^RADIAL on the projector (i) center.
                    // Translational invariance gives d/dA = <i*r^RADIAL|nabla_j>.
                    // This includes motion of the radial moment's origin.
#pragma unroll
                    for (int rx = 0; rx <= RADIAL/2; ++rx) {
#pragma unroll
                        for (int ry = 0; ry <= RADIAL/2-rx; ++ry) {
                            int rz = RADIAL/2-rx-ry;
                            double weight = 1.;
                            if (RADIAL == 4 && (rx == 1 || ry == 1 || rz == 1)) {
                                weight = 2.;
                            }
                            int addrx = (ix + 2*rx + jx*stride_j) * nsp_per_block;
                            int addry = (iy + 2*ry + jy*stride_j) * nsp_per_block;
                            int addrz = (iz + 2*rz + jz*stride_j) * nsp_per_block;
                            int dj = stride_j * nsp_per_block;
                            double sx = gx[addrx];
                            double sy = gy[addry];
                            double sz = gz[addrz];
                            double dx = -2*aj*gx[addrx+dj];
                            double dy = -2*aj*gy[addry+dj];
                            double dz = -2*aj*gz[addrz+dj];
                            if (jx > 0) dx += jx*gx[addrx-dj];
                            if (jy > 0) dy += jy*gy[addry-dj];
                            if (jz > 0) dz += jz*gz[addrz-dj];
                            v[0] += weight * dx*sy*sz;
                            v[1] += weight * sx*dy*sz;
                            v[2] += weight * sx*sy*dz;
                        }
                    }
                    double dm_val = dm_ij[(size_t)i*naoj+j];
                    v_ix += v[0] * dm_val;
                    v_iy += v[1] * dm_val;
                    v_iz += v[2] * dm_val;
                }
            }
            double xjxi = rjri[0*nsp_per_block];
            double yjyi = rjri[1*nsp_per_block];
            double zjzi = rjri[2*nsp_per_block];
            sigma_xx -= v_ix * xjxi;
            sigma_xy -= v_ix * yjyi;
            sigma_xz -= v_ix * zjzi;
            sigma_yx -= v_iy * xjxi;
            sigma_yy -= v_iy * yjyi;
            sigma_yz -= v_iy * zjzi;
            sigma_zx -= v_iz * xjxi;
            sigma_zy -= v_iz * yjyi;
            sigma_zz -= v_iz * zjzi;
            grad_ix += v_ix;
            grad_iy += v_iy;
            grad_iz += v_iz;
        }
        int ish_cell0 = ish;
        int ia = bas[ish_cell0*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh_cell0*BAS_SLOTS+ATOM_OF];
        atomicAdd(grad+ia*3+0, grad_ix);
        atomicAdd(grad+ia*3+1, grad_iy);
        atomicAdd(grad+ia*3+2, grad_iz);
        atomicAdd(grad+ja*3+0, -grad_ix);
        atomicAdd(grad+ja*3+1, -grad_iy);
        atomicAdd(grad+ja*3+2, -grad_iz);
    }
    atomicAdd(sigma+0, sigma_xx);
    atomicAdd(sigma+1, sigma_xy);
    atomicAdd(sigma+2, sigma_xz);
    atomicAdd(sigma+3, sigma_yx);
    atomicAdd(sigma+4, sigma_yy);
    atomicAdd(sigma+5, sigma_yz);
    atomicAdd(sigma+6, sigma_zx);
    atomicAdd(sigma+7, sigma_zy);
    atomicAdd(sigma+8, sigma_zz);
}

template<int RADIAL>
static int ppnl_derivatives(double *grad, double *sigma, double *dm,
                            PBCIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                            int *shl_pair_offsets, int *bas_ij_idx,
                            int *gout_stride_lookup, int naoi, int naoj)
{
    if (nbatches_shl_pair == 0) return 0;
    cudaError_t err = cudaFuncSetAttribute(
        ppnl_derivatives_kernel<RADIAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    if (err == cudaSuccess) {
        ppnl_derivatives_kernel<RADIAL><<<nbatches_shl_pair, THREADS, shm_size>>>(
            grad, sigma, dm, *envs, shl_pair_offsets, bas_ij_idx,
            gout_stride_lookup, naoi, naoj);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ppnl derivatives (r^%d): %s\n",
                RADIAL, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int PBCint1e_r2_origi(double *out, PBCIntEnvVars *envs, int shm_size,
                      int nbatches_shl_pair, int *bas_ij_idx,
                      int *shl_pair_offsets, int *gout_stride_lookup,
                      int naoi, int naoj, size_t ij_offset)
{
    cudaFuncSetAttribute(int1e_r2_origi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int1e_r2_origi_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
            naoi, naoj, ij_offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_r2_origi kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCint1e_r4_origi(double *out, PBCIntEnvVars *envs, int shm_size,
                      int nbatches_shl_pair, int *bas_ij_idx,
                      int *shl_pair_offsets, int *gout_stride_lookup,
                      int naoi, int naoj, size_t ij_offset)
{
    cudaFuncSetAttribute(int1e_r4_origi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int1e_r4_origi_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
            naoi, naoj, ij_offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_r4_origi kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCint1e_r2_origi_ip2(double *out, PBCIntEnvVars *envs, int shm_size,
                          int nbatches_shl_pair, int *bas_ij_idx,
                          int *shl_pair_offsets, int *gout_stride_lookup,
                          int naoi, int naoj, size_t ij_offset)
{
    cudaFuncSetAttribute(int1e_r2_origi_ip2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int1e_r2_origi_ip2_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
            naoi, naoj, ij_offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_r2_origi_ip2 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCint1e_r4_origi_ip2(double *out, PBCIntEnvVars *envs, int shm_size,
                          int nbatches_shl_pair, int *bas_ij_idx,
                          int *shl_pair_offsets, int *gout_stride_lookup,
                          int naoi, int naoj, size_t ij_offset)
{
    cudaFuncSetAttribute(int1e_r4_origi_ip2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int1e_r4_origi_ip2_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
            naoi, naoj, ij_offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_r4_origi_ip2 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCovlp_cross_derivatives(double *grad, double *sigma, double *dm,
                            PBCIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                            int *shl_pair_offsets, int *bas_ij_idx,
                            int *gout_stride_lookup, int naoi, int naoj)
{
    return ppnl_derivatives<0>(grad, sigma, dm, envs, shm_size,
                                 nbatches_shl_pair, shl_pair_offsets, bas_ij_idx,
                                 gout_stride_lookup, naoi, naoj);
}

int PBCint1e_r2_origi_derivatives(double *grad, double *sigma, double *dm,
                            PBCIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                            int *shl_pair_offsets, int *bas_ij_idx,
                            int *gout_stride_lookup, int naoi, int naoj)
{
    return ppnl_derivatives<2>(grad, sigma, dm, envs, shm_size,
                                 nbatches_shl_pair, shl_pair_offsets, bas_ij_idx,
                                 gout_stride_lookup, naoi, naoj);
}

int PBCint1e_r4_origi_derivatives(double *grad, double *sigma, double *dm,
                            PBCIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                            int *shl_pair_offsets, int *bas_ij_idx,
                            int *gout_stride_lookup, int naoi, int naoj)
{
    return ppnl_derivatives<4>(grad, sigma, dm, envs, shm_size,
                                 nbatches_shl_pair, shl_pair_offsets, bas_ij_idx,
                                 gout_stride_lookup, naoi, naoj);
}

}
