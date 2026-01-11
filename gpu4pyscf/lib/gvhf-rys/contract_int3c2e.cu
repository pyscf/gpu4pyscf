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
#include "gint/cuda_alloc.cuh"
#include "vhf.cuh"
#include "rys_roots.cu"
#include "rys_contract_k.cuh"
#include "create_tasks.cu"

#define THREADS         256
#define BLOCK_SIZE      16
#define NF_AUX_MAX      28
#define IJ_WIDTH        30

__global__ static
void contract_int3c2e_dm_kernel(double *out, double *dm, RysIntEnvVars envs,
                                int *shl_pair_offsets, uint32_t *bas_ij_idx,
                                int *gout_stride_lookup)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int ksh = blockIdx.x + nbas;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nfi, nfk, nfij;
    __shared__ int nao;
    __shared__ int gout_stride;
    if (thread_id == 0) {
        int sp_block_id = gridDim.y - blockIdx.y - 1;
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj;
        nroots = (lij + lk) / 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = ao_loc[nbas];
        nfi = (li + 1) * (li + 2) / 2;
        int nfj = (lj + 1) * (lj + 2) / 2;
        nfk = (lk + 1) * (lk + 2) / 2;
        nfij = nfi * nfj;
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfj = (lj + 1) * (lj + 2) / 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 4 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    int *idx_i = (int*)(shared_memory + nsp_per_block*(g_size*3+nroots*2+7));
    int *idx_j = idx_i + nfi * 3;
    int *idx_k = idx_j + nfj * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nsp_per_block;
        idx_i[thread_id] += (thread_id % 3) * gx_len;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nsp_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nsp_per_block;
    }
    double vj_xyz[NF_AUX_MAX];
    for (int n = 0; n < NF_AUX_MAX; ++n) {
        vj_xyz[n] = 0;
    }

    __shared__ double xk, yk, zk;
    __shared__ double *expk, *ck;
    if (thread_id == 0) {
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        xk = rk[0];
        yk = rk[1];
        zk = rk[2];
        expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        __syncthreads();
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        double *dm_local = dm + j0 * nao + i0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xjxi = rj[0] - xi;
        double yjyi = rj[1] - yi;
        double zjzi = rj[2] - zi;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        double fac_ij = PI_FAC;
        if (ish == jsh) {
            fac_ij *= .5;
        } else if (ish < jsh) {
            fac_ij = 0;
        }
        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xij = xi + rjri[0*nsp_per_block] * aj_aij;
            double yij = yi + rjri[1*nsp_per_block] * aj_aij;
            double zij = zi + rjri[2*nsp_per_block] * aj_aij;
            double xpq = xij - xk;
            double ypq = yij - yk;
            double zpq = zij - zk;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            if (gout_id == 0) {
                double theta_ij = ai * aj_aij;
                double rr_ij = rjri[3*nsp_per_block];
                double Kab = theta_ij * rr_ij;
                double cicj = fac_ij * ci[ip] * cj[jp];
                gx[0] = cicj * exp(-Kab);
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
            }
            for (int kp = 0; kp < kprim; ++kp) {
                __syncthreads();
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                if (gout_id == 0) {
                    gx[gx_len] = ck[kp] / (aij*ak*sqrt(aij+ak));
                }
                rys_roots_rs(nroots, theta, rr, omega, rw, nsp_per_block,
                             gout_id, gout_stride);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    double rt_aa = rt / (aij + ak);
                    double s0x, s1x, s2x;
                    int lij = li + lj;
                    if (lij > 0) {
                        __syncthreads();
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double xpa = rjri[n*nsp_per_block] * aj_aij;
                            double c0x = xpa - rt_aij * Rpq[n*nsp_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsp_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsp_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lk > 0) {
                        double rt_ak  = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/ak  * (1 - rt_ak);
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3;
                            int _ix = n % 3;
                            double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                            double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp_per_block];
                                }
                                _gx[stride_k*nsp_per_block] = s1x;
                            }
                            for (int k = 1; k < lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }

                    if (lj > 0) {
                        __syncthreads();
                        if (pair_ij < shl_pair1) {
                            for (int m = gout_id; m < (lk+1)*3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xjxi = rjri[_ix*nsp_per_block];
                                double *_gx = gx + (_ix*g_size + k*stride_k) * nsp_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = lij + j*li; // = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nsp_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nsp_per_block];
                                        _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (pair_ij < shl_pair1) {
                        for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                            int i = ij % nfi;
                            int j = ij / nfi;
                            double dm_ij = dm_local[j*nao+i];
                            int ij_addrx = idx_i[i*3+0] + idx_j[j*3+0];
                            int ij_addry = idx_i[i*3+1] + idx_j[j*3+1];
                            int ij_addrz = idx_i[i*3+2] + idx_j[j*3+2];
#pragma unroll
                            for (int k = 0; k < NF_AUX_MAX; ++k) {
                                if (k >= nfk) break;
                                int addrx = ij_addrx + idx_k[k*3+0];
                                int addry = ij_addry + idx_k[k*3+1];
                                int addrz = ij_addrz + idx_k[k*3+2];
                                vj_xyz[k] += gx[addrx] * gx[addry] * gx[addrz] * dm_ij;
                            }
                        }
                    }
                }
            }
        }
    }
    int k0 = ao_loc[ksh] - ao_loc[nbas];
    double *vj = out + k0;
#pragma unroll
    for (int k = 0; k < NF_AUX_MAX; ++k) {
        if (k >= nfk) break;
        atomicAdd(vj + k, vj_xyz[k]);
    }
}

__global__ static
void contract_int3c2e_auxvec_kernel(double *out, double *auxvec, RysIntEnvVars envs,
                                    int *shl_pair_offsets, uint32_t *bas_ij_idx,
                                    int *ksh_offsets, int *gout_stride_lookup)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nfi, nfk, nfij;
    __shared__ int gout_stride;
    double omega = env[PTR_RANGE_OMEGA];
    if (thread_id == 0) {
        int sp_block_id = gridDim.x - blockIdx.x - 1;
        int ksh_block_id = gridDim.y - blockIdx.y - 1;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        int lij = li + lj;
        nroots = (lij + lk) / 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nfi = (li + 1) * (li + 2) / 2;
        int nfj = (lj + 1) * (lj + 2) / 2;
        nfk = (lk + 1) * (lk + 2) / 2;
        nfij = nfi * nfj;
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfj = (lj + 1) * (lj + 2) / 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 4 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    int *idx_i = (int*)(shared_memory + nsp_per_block*(g_size*3+nroots*2+7));
    int *idx_j = idx_i + nfi * 3;
    int *idx_k = idx_j + nfj * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nsp_per_block;
        idx_i[thread_id] += (thread_id % 3) * gx_len;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nsp_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nsp_per_block;
    }
    __shared__ double vj_aux[NF_AUX_MAX];

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xjxi = rj[0] - xi;
        double yjyi = rj[1] - yi;
        double zjzi = rj[2] - zi;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        double gout[IJ_WIDTH];
#pragma unroll
        for (int n = 0; n < IJ_WIDTH; ++n) { gout[n] = 0; }

        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            if (gout_id == 0) {
                double theta_ij = ai * aj_aij;
                double rr_ij = rjri[3*nsp_per_block];
                double Kab = exp(-theta_ij * rr_ij);
                double cicj = ci[ip] * cj[jp] * Kab;
                if (pair_ij >= shl_pair1) {
                    cicj = 0;
                }
                gx[0] = PI_FAC * cicj;
            }

            for (int ksh = ksh0; ksh < ksh1; ++ksh) {
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh] - ao_loc[nbas];
                __syncthreads();
                if (thread_id < nfk) {
                    vj_aux[thread_id] = auxvec[k0+thread_id];
                }
                double xij = xi + rjri[0*nsp_per_block] * aj_aij;
                double yij = yi + rjri[1*nsp_per_block] * aj_aij;
                double zij = zi + rjri[2*nsp_per_block] * aj_aij;
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                if (gout_id == 0) {
                    Rpq[0*nsp_per_block] = xpq;
                    Rpq[1*nsp_per_block] = ypq;
                    Rpq[2*nsp_per_block] = zpq;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    __syncthreads();
                    double ak = expk[kp];
                    double theta = aij * ak / (aij + ak);
                    if (gout_id == 0) {
                        gx[gx_len] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, rr, omega, rw, nsp_per_block,
                                 gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                        }
                        double rt = rw[irys*2*nsp_per_block];
                        double rt_aa = rt / (aij + ak);
                        double s0x, s1x, s2x;
                        int lij = li + lj;
                        if (lij > 0) {
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            __syncthreads();
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * g_size * nsp_per_block;
                                double Rpa = (rjri[n*nsp_per_block]) * aj_aij;
                                double c0x = Rpa - rt_aij * Rpq[n*nsp_per_block];
                                s0x = _gx[0];
                                s1x = c0x * s0x;
                                _gx[nsp_per_block] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    s2x = c0x * s1x + i * b10 * s0x;
                                    _gx[(i+1)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        if (lk > 0) {
                            double rt_ak = rt_aa * aij;
                            double b00 = .5 * rt_aa;
                            double b01 = .5/ak * (1 - rt_ak);
                            int lij3 = (lij+1)*3;
                            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                                __syncthreads();
                                int i = n / 3;
                                int _ix = n % 3;
                                double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                                double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                                if (n < lij3) {
                                    s0x = _gx[0];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[-nsp_per_block];
                                    }
                                    _gx[stride_k*nsp_per_block] = s1x;
                                }
                                for (int k = 1; k < lk; ++k) {
                                    __syncthreads();
                                    if (n < lij3) {
                                        s2x = cpx*s1x + k*b01*s0x;
                                        if (i > 0) {
                                            s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                        }
                                        _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                        s0x = s1x;
                                        s1x = s2x;
                                    }
                                }
                            }
                        }

                        if (lj > 0) {
                            __syncthreads();
                            if (pair_ij < shl_pair1) {
                                for (int m = gout_id; m < (lk+1)*3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nsp_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nsp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = lij + j*li; // = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nsp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nsp_per_block];
                                            _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }

                        __syncthreads();
                        if (pair_ij < shl_pair1) {
#pragma unroll
                            for (int n = 0; n < IJ_WIDTH; ++n) {
                                int ij = gout_id + n * gout_stride;
                                if (ij >= nfij) break;
                                int i = ij % nfi;
                                int j = ij / nfi;
                                int addrx = idx_i[i*3+0] + idx_j[j*3+0];
                                int addry = idx_i[i*3+1] + idx_j[j*3+1];
                                int addrz = idx_i[i*3+2] + idx_j[j*3+2];
                                for (int k = 0; k < nfk; ++k) {
                                    double Ix = gx[addrx + idx_k[k*3+0]];
                                    double Iy = gx[addry + idx_k[k*3+1]];
                                    double Iz = gx[addrz + idx_k[k*3+2]];
                                    gout[n] += Ix * Iy * Iz * vj_aux[k];
                                }
                            }
                        }
                    }
                }
            }
        }

        if (pair_ij < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            int nao = ao_loc[nbas];
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jsh];
            double *vj = out + i0 * nao + j0;
#pragma unroll
            for (int n = 0; n < IJ_WIDTH; ++n) {
                int ij = gout_id + n * gout_stride;
                if (ij >= nfij) break;
                int i = ij % nfi;
                int j = ij / nfi;
                atomicAdd(vj+i*nao+j, gout[n]);
            }
        }
    }
}

extern "C" {
int contract_int3c2e_dm(double *out, double *dm, RysIntEnvVars *envs, int shm_size,
                        int nbas_aux, int nbatches_shl_pair, int *shl_pair_offsets,
                        uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(contract_int3c2e_dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbas_aux, nbatches_shl_pair);
    contract_int3c2e_dm_kernel<<<blocks, THREADS, shm_size>>>(
            out, dm, *envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// contract('ijP,P->ij', int3c2e, auxvec)
int contract_int3c2e_auxvec(double *vj, double *auxvec, RysIntEnvVars *envs, int shm_size,
                            int nbatches_shl_pair, int nbatches_ksh,
                            int *shl_pair_offsets, int *ksh_offsets,
                            uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(contract_int3c2e_auxvec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    contract_int3c2e_auxvec_kernel<<<blocks, THREADS, shm_size>>>(
            vj, auxvec, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_auxvec, error message = %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
