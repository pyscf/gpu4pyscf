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
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"

#define THREADS         256
#define GOUT_IP_WIDTH   20
#define BLOCK_SIZE      16
#define L_AUX           6
#define L_AUX1          (L_AUX+1)

__global__ static
void pbc_int2c2e_ip1_kernel(double *out, PBCIntEnvVars envs, int *shl_pair_offsets,
                            uint32_t *bas_ij_idx, int *gout_stride_lookup
                            #ifdef USE_SYCL
                            , sycl::nd_item<1> &item, char *shm_mem
                            #endif
                            )
{
    #ifdef USE_SYCL
    int sp_block_id = item.get_group(0);
    int thread_id = item.get_local_id(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nbas = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double &omega = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int nbas;
    __shared__ int li, lj, nroots, nao, iprim, jprim;
    __shared__ int gout_stride;
    __shared__ double omega;

    extern __shared__ double shared_memory[];
    #endif

    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
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
        omega = env[PTR_RANGE_OMEGA];
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
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    const int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    const int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    double goutx[GOUT_IP_WIDTH];
    double gouty[GOUT_IP_WIDTH];
    double goutz[GOUT_IP_WIDTH];
    if (gout_id == 0) {
        gx[gx_len] = 1.;
    }

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
            bas_ij = bas_ij_idx[shl_pair0];;
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
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
                    gx[0] = PI_FAC * cicj / (ai*aj*sqrt(aij));
                }
                double rr = Rpq[3*nsp_per_block];
                rys_roots_rs(nroots, theta, rr, omega, rw, nsp_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    double rt_aa = rt / aij;
                    __syncthreads();
                    double rt_ai = rt_aa * aj;
                    double b10 = .5/ai * (1 - rt_ai);
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = gx + n * gx_len;
                        double c0x = -rt_ai * Rpq[n*nsp_per_block];
                        s0x = _gx[0];
                        s1x = c0x * s0x;
                        _gx[nsp_per_block] = s1x;
                        for (int i = 1; i <= li; ++i) {
                            s2x = c0x * s1x + i * b10 * s0x;
                            _gx[(i+1)*nsp_per_block] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                    if (lj > 0) {
                        int li3 = (li+2)*3;
                        double rt_ak  = rt_aa * ai;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/aj  * (1 - rt_ak);
                        for (int n = gout_id; n < li3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(li+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                            double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                            //for i in range(li+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < li3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp_per_block];
                                }
                                _gx[stride_j*nsp_per_block] = s1x;
                            }
                            for (int j = 1; j < lj; ++j) {
                                __syncthreads();
                                if (n < li3) {
                                    s2x = cpx*s1x + j*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(j*stride_j-1)*nsp_per_block];
                                    }
                                    _gx[(j*stride_j+stride_j)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }
                    __syncthreads();
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

__global__ static
void e_int2c2e_ip1_kernel(double *out, double *dm, PBCIntEnvVars envs,
                          int *shl_pair_offsets, uint32_t *bas_ij_idx,
                          int *gout_stride_lookup
                          #ifdef USE_SYCL
                          , sycl::nd_item<1> &item, char *shm_mem
                          #endif
                          )
{
    #ifdef USE_SYCL
    int sp_block_id = item.get_group(0);
    int thread_id = item.get_local_id(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nbas = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double &omega = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int nbas;
    __shared__ int li, lj, nroots, nao, iprim, jprim;
    __shared__ int gout_stride;
    __shared__ double omega;

    extern __shared__ double shared_memory[];
    #endif
    
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
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
        omega = env[PTR_RANGE_OMEGA];
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
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 2;
    int i_1 =          nsp_per_block;
    int j_1 = stride_j*nsp_per_block;
    int g_size = stride_j * (lj + 2);
    int gx_len = g_size * nsp_per_block;
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    const int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    const int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    if (gout_id == 0) {
        gx[gx_len] = 1.;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        __syncthreads();
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];;
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        double *dm_local = dm + j0 * nao + i0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double theta = ai * aj / aij;
                if (gout_id == 0) {
                    double cicj = PI_FAC * ci[ip] * cj[jp];
                    if (ish == jsh) {
                        cicj *= .5;
                    } else if (ish < jsh) {
                        cicj = 0;
                    }
                    gx[0] = cicj / (ai*aj*sqrt(aij));
                }
                double rr = Rpq[3*nsp_per_block];
                rys_roots_rs(nroots, theta, rr, omega, rw, nsp_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    double rt_aa = rt / aij;
                    __syncthreads();
                    double rt_ai = rt_aa * aj;
                    double b10 = .5/ai * (1 - rt_ai);
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = gx + n * gx_len;
                        double c0x = -rt_ai * Rpq[n*nsp_per_block];
                        s0x = _gx[0];
                        s1x = c0x * s0x;
                        _gx[nsp_per_block] = s1x;
                        for (int i = 1; i <= li; ++i) {
                            s2x = c0x * s1x + i * b10 * s0x;
                            _gx[(i+1)*nsp_per_block] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                    int li3 = (li+2)*3;
                    double rt_ak  = rt_aa * ai;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/aj  * (1 - rt_ak);
                    for (int n = gout_id; n < li3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(li+1):
                        int _ix = n % 3; // TODO: remove _ix for nroots > 2
                        double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                        double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                        //for i in range(li+1):
                        //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                        if (n < li3) {
                            s0x = _gx[0];
                            s1x = cpx * s0x;
                            if (i > 0) {
                                s1x += i * b00 * _gx[-nsp_per_block];
                            }
                            _gx[stride_j*nsp_per_block] = s1x;
                        }
                        for (int j = 1; j <= lj; ++j) {
                            __syncthreads();
                            if (n < li3) {
                                s2x = cpx*s1x + j*b01*s0x;
                                if (i > 0) {
                                    s2x += i * b00 * _gx[(j*stride_j-1)*nsp_per_block];
                                }
                                _gx[(j*stride_j+stride_j)*nsp_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1) {
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = idx_i[i*3+0];
                            int iy = idx_i[i*3+1];
                            int iz = idx_i[i*3+2];
                            int jx = idx_j[j*3+0];
                            int jy = idx_j[j*3+1];
                            int jz = idx_j[j*3+2];
                            int addrx = (ix + jx*stride_j) * nsp_per_block;
                            int addry = (iy + jy*stride_j + g_size) * nsp_per_block;
                            int addrz = (iz + jz*stride_j + g_size*2) * nsp_per_block;
                            double Ix = gx[addrx];
                            double Iy = gx[addry];
                            double Iz = gx[addrz];
                            double dm_ij = dm_local[j*nao+i];
                            double prod_xy = Ix * Iy * dm_ij;
                            double prod_xz = Ix * Iz * dm_ij;
                            double prod_yz = Iy * Iz * dm_ij;
                            double fix = ai2 * gx[addrx+i_1]; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                            double fiy = ai2 * gx[addry+i_1]; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                            double fiz = ai2 * gx[addrz+i_1]; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                            double fjx = aj2 * gx[addrx+j_1]; if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                            double fjy = aj2 * gx[addry+j_1]; if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                            double fjz = aj2 * gx[addrz+j_1]; if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                        }
                    }
                }
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            atomicAdd(out+ia*3+0, v_ix * 2);
            atomicAdd(out+ia*3+1, v_iy * 2);
            atomicAdd(out+ia*3+2, v_iz * 2);
            atomicAdd(out+ja*3+0, v_jx * 2);
            atomicAdd(out+ja*3+1, v_jy * 2);
            atomicAdd(out+ja*3+2, v_jz * 2);
        }
    }
}

extern "C" {
int fill_int2c2e_ip1(double *out, PBCIntEnvVars *envs, int shm_size,
                     int nbatches_shl_pair, int *shl_pair_offsets,
                     uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    #ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class pbc_int2c2e_ip1_sycl>(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
        pbc_int2c2e_ip1_kernel(out, dev_envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup,
                               item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(pbc_int2c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    pbc_int2c2e_ip1_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, *envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_ip1 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int e_int2c2e_ip1(double *out, double *dm, PBCIntEnvVars *envs, int shm_size,
                     int nbatches_shl_pair, int *shl_pair_offsets,
                     uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    #ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class e_int2c2e_ip1_sycl>(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
        e_int2c2e_ip1_kernel(out, dm, dev_envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup,
                             item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(e_int2c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    e_int2c2e_ip1_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, dm, *envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_ip1 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
