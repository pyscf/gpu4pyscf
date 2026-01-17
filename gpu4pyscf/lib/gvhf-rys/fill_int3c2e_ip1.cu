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
#include "vhf.cuh"
#include "rys_roots.cu"
#include "rys_contract_k.cuh"

#define THREADS         256
#define GOUT_IP1_WIDTH  27

__global__ static
void int3c2e_ip1_kernel(double *out, RysIntEnvVars envs, int *shl_pair_offsets,
                    uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int ao_pair_offset, int aux_offset,
                    int nao_pairs, int naux)
{
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1, nshl_pair;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lij, lk, nroots, nf, nao;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int gout_stride, nst_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        nshl_pair = shl_pair1 - shl_pair0;
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        lij = li + lj + 1;
        nroots = (lij + lk) / 2 + 1;
        double omega = env[PTR_RANGE_OMEGA];
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        nao = envs.ao_loc[nbas];
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 1);
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;
    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory;
    double *Rpq = shared_memory + nst_per_block * 4 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    double goutx[GOUT_IP1_WIDTH];
    double gouty[GOUT_IP1_WIDTH];
    double goutz[GOUT_IP1_WIDTH];
    if (gout_id == 0) {
        gx[gx_len] = PI_FAC;
    }

    int nksp = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nksp+st_id; ijk_idx += nst_per_block) {
        // convert task_id to ish, jsh, ksh
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nksp) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[gx_len] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = env[rj+0] - env[ri+0];
            double yjyi = env[rj+1] - env[ri+1];
            double zjzi = env[rj+2] - env[ri+2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[st_id+0*nst_per_block] = xjxi;
            rjri[st_id+1*nst_per_block] = yjyi;
            rjri[st_id+2*nst_per_block] = zjzi;
            rjri[st_id+3*nst_per_block] = rr_ij;
        }
#pragma unroll
        for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
            goutx[n] = 0;
            gouty[n] = 0;
            goutz[n] = 0;
        }

        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            __syncthreads();
            int expi = bas[ish*BAS_SLOTS+PTR_EXP];
            int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
            int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = env[expi+ip];
            double aj = env[expj+jp];
            double ak = env[expk+kp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xij = rjri[st_id+0*nst_per_block] * aj_aij + env[ri+0];
            double yij = rjri[st_id+1*nst_per_block] * aj_aij + env[ri+1];
            double zij = rjri[st_id+2*nst_per_block] * aj_aij + env[ri+2];
            double xpq = xij - env[rk+0];
            double ypq = yij - env[rk+1];
            double zpq = zij - env[rk+2];
            if (gout_id == 0) {
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double rr_ij = rjri[st_id+3*nst_per_block];
                double Kab = theta_ij * rr_ij;
                gx[0] = fac * exp(-Kab);
                Rpq[0*nst_per_block] = xpq;
                Rpq[1*nst_per_block] = ypq;
                Rpq[2*nst_per_block] = zpq;
            }
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * ak / (aij + ak);
            double omega = env[PTR_RANGE_OMEGA];
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, gout_stride);
            double s0x, s1x, s2x;
            for (int irys = 0; irys < nroots; ++irys) {
                int stride_j = li + 2;
                int stride_k = stride_j * (lj + 1);
                int gsize = g_size;
                int i_1 = nst_per_block;
                int nst = nst_per_block;
                __syncthreads();
                if (gout_id == 0) {
                    gx[gsize*nst*2] = rw[(irys*2+1)*nst];
                }
                double rt = rw[ irys*2   *nst];
                double rt_aa = rt / (aij + ak);

                __syncthreads();
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                for (int n = gout_id; n < 3; n += gout_stride) {
                    double *_gx = gx + n * gsize * nst;
                    double xjxi = rjri[st_id+n*nst];
                    double xpa = xjxi * aj_aij;
                    //double c0x = Rpa[ir] - rt_aij * Rpq[n*nst];
                    double c0x = xpa - rt_aij * Rpq[n*nst];
                    s0x = _gx[0];
                    s1x = c0x * s0x;
                    _gx[nst] = s1x;
                    for (int i = 1; i < lij; ++i) {
                        s2x = c0x * s1x + i * b10 * s0x;
                        _gx[(i+1)*nst] = s2x;
                        s0x = s1x;
                        s1x = s2x;
                    }
                }

                if (lk > 0) {
                    int lij3 = (lij+1)*3;
                    double rt_ak  = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/ak  * (1 - rt_ak );
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3; // TODO: remove _ix for nroots > 2
                        double *_gx = gx + (i + _ix * gsize) * nst;
                        double cpx = rt_ak * Rpq[_ix*nst];
                        //for i in range(lij+1):
                        //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                        if (n < lij3) {
                            s0x = _gx[0];
                            s1x = cpx * s0x;
                            if (i > 0) {
                                s1x += i * b00 * _gx[-nst];
                            }
                            _gx[stride_k*nst] = s1x;
                        }
                        //for k in range(1, lk):
                        //    for i in range(lij+1):
                        //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                        for (int k = 1; k < lk; ++k) {
                            __syncthreads();
                            if (n < lij3) {
                                s2x = cpx*s1x + k*b01*s0x;
                                if (i > 0) {
                                    s2x += i * b00 * _gx[(k*stride_k-1)*nst];
                                }
                                _gx[(k*stride_k+stride_k)*nst] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }
                }

                if (lj > 0) {
                    __syncthreads();
                    int lk3 = (lk+1)*3;
                    for (int m = gout_id; m < lk3; m += gout_stride) {
                        int k = m / 3;
                        int _ix = m % 3;
                        double xjxi = rjri[st_id+_ix*nst];
                        double *_gx = gx + (_ix*gsize + k*stride_k) * nst;
                        for (int j = 0; j < lj; ++j) {
                            int ij = (lij-j) + j*stride_j;
                            s1x = _gx[ij*nst];
                            for (--ij; ij >= j*stride_j; --ij) {
                                s0x = _gx[ij*nst];
                                _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x;
                                s1x = s0x;
                            }
                        }
                    }
                }

                __syncthreads();
                if (ijk_idx < nksp) {
                    int nfi = c_nf[li];
                    int nfj = c_nf[lj];
                    float div_nfi = c_div_nf[li];
                    float div_nfj = c_div_nf[lj];
                    double ai2 = ai * 2;
#pragma unroll
                    for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t jk = ijk * div_nfi;
                        uint32_t i = ijk - jk * nfi;
                        uint32_t k = jk * div_nfj;
                        uint32_t j = jk - k * nfj;
                        int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                        int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                        int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                        int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                        int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                        int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                        int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                        int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                        int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                        int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                        int addry = (iy + jy*stride_j + ky*stride_k + gsize) * nst;
                        int addrz = (iz + jz*stride_j + kz*stride_k + gsize*2) * nst;
                        double gix = gx[addrx+i_1];
                        double giy = gx[addry+i_1];
                        double giz = gx[addrz+i_1];
                        double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                        double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                        double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }
                        goutx[n] += fix * gx[addry] * gx[addrz];
                        gouty[n] += gx[addrx] * fiy * gx[addrz];
                        goutz[n] += gx[addrx] * gx[addry] * fiz;
                    }
                }
            }
        }

        if (ijk_idx < nksp) {
            int nfk = c_nf[lk];
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            size_t offset = nao_pairs * naux;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
                int ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                int ij = ijk / nfk;
                int k  = ijk - nfk * ij;
                j3c_tensor[            ij*naux + k*nksh] = goutx[n];
                j3c_tensor[offset    + ij*naux + k*nksh] = gouty[n];
                j3c_tensor[offset *2 + ij*naux + k*nksh] = goutz[n];
            }
        }
    }
}

__global__ static
void int3c2e_ipaux_kernel(double *out, RysIntEnvVars envs, int *shl_pair_offsets,
                    uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int ao_pair_offset, int aux_offset,
                    int nao_pairs, int naux)
{
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1, nshl_pair;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lij, lk, nroots, nf, nao;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int gout_stride, nst_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        nshl_pair = shl_pair1 - shl_pair0;
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        lij = li + lj;
        nroots = (lij + lk + 1) / 2 + 1;
        double omega = env[PTR_RANGE_OMEGA];
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        nao = envs.ao_loc[nbas];
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        int stride_j = li + 1;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 2);
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;
    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory;
    double *Rpq = shared_memory + nst_per_block * 4 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    double goutx[GOUT_IP1_WIDTH];
    double gouty[GOUT_IP1_WIDTH];
    double goutz[GOUT_IP1_WIDTH];
    if (gout_id == 0) {
        gx[gx_len] = PI_FAC;
    }

    int nksp = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nksp+st_id; ijk_idx += nst_per_block) {
        // convert task_id to ish, jsh, ksh
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nksp) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[gx_len] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = env[rj+0] - env[ri+0];
            double yjyi = env[rj+1] - env[ri+1];
            double zjzi = env[rj+2] - env[ri+2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[st_id+0*nst_per_block] = xjxi;
            rjri[st_id+1*nst_per_block] = yjyi;
            rjri[st_id+2*nst_per_block] = zjzi;
            rjri[st_id+3*nst_per_block] = rr_ij;
        }
#pragma unroll
        for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
            goutx[n] = 0;
            gouty[n] = 0;
            goutz[n] = 0;
        }

        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            __syncthreads();
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = env[expi+ip];
            double aj = env[expj+jp];
            double ak = env[expk+kp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xij = rjri[st_id+0*nst_per_block] * aj_aij + env[ri+0];
            double yij = rjri[st_id+1*nst_per_block] * aj_aij + env[ri+1];
            double zij = rjri[st_id+2*nst_per_block] * aj_aij + env[ri+2];
            double xpq = xij - env[rk+0];
            double ypq = yij - env[rk+1];
            double zpq = zij - env[rk+2];
            if (gout_id == 0) {
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double rr_ij = rjri[st_id+3*nst_per_block];
                double Kab = theta_ij * rr_ij;
                gx[0] = fac * exp(-Kab);
                Rpq[0*nst_per_block] = xpq;
                Rpq[1*nst_per_block] = ypq;
                Rpq[2*nst_per_block] = zpq;
            }
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * ak / (aij + ak);
            double omega = env[PTR_RANGE_OMEGA];
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, gout_stride);
            double s0x, s1x, s2x;
            for (int irys = 0; irys < nroots; ++irys) {
                int stride_j = li + 1;
                int stride_k = stride_j * (lj + 1);
                int gsize = g_size;
                int k_1 = stride_k*nst_per_block;
                int nst = nst_per_block;
                __syncthreads();
                if (gout_id == 0) {
                    gx[gsize*nst*2] = rw[(irys*2+1)*nst];
                }
                double rt = rw[ irys*2   *nst];
                double rt_aa = rt / (aij + ak);

                __syncthreads();
                if (lij > 0) {
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = gx + n * gsize * nst;
                        double xjxi = rjri[st_id+n*nst];
                        double xpa = xjxi * aj_aij;
                        //double c0x = Rpa[ir] - rt_aij * Rpq[n*nst];
                        double c0x = xpa - rt_aij * Rpq[n*nst];
                        s0x = _gx[0];
                        s1x = c0x * s0x;
                        _gx[nst] = s1x;
                        for (int i = 1; i < lij; ++i) {
                            s2x = c0x * s1x + i * b10 * s0x;
                            _gx[(i+1)*nst] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                }

                int lij3 = (lij+1)*3;
                double rt_ak  = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak  * (1 - rt_ak );
                for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                    __syncthreads();
                    int i = n / 3; //for i in range(lij+1):
                    int _ix = n % 3; // TODO: remove _ix for nroots > 2
                    double *_gx = gx + (i + _ix * gsize) * nst;
                    double cpx = rt_ak * Rpq[_ix*nst];
                    //for i in range(lij+1):
                    //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                    if (n < lij3) {
                        s0x = _gx[0];
                        s1x = cpx * s0x;
                        if (i > 0) {
                            s1x += i * b00 * _gx[-nst];
                        }
                        _gx[stride_k*nst] = s1x;
                    }
                    //for k in range(1, lk):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    for (int k = 1; k <= lk; ++k) {
                        __syncthreads();
                        if (n < lij3) {
                            s2x = cpx*s1x + k*b01*s0x;
                            if (i > 0) {
                                s2x += i * b00 * _gx[(k*stride_k-1)*nst];
                            }
                            _gx[(k*stride_k+stride_k)*nst] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                }

                if (lj > 0) {
                    __syncthreads();
                    int lk3 = (lk+2)*3;
                    for (int m = gout_id; m < lk3; m += gout_stride) {
                        int k = m / 3;
                        int _ix = m % 3;
                        double xjxi = rjri[st_id+_ix*nst];
                        double *_gx = gx + (_ix*gsize + k*stride_k) * nst;
                        for (int j = 0; j < lj; ++j) {
                            int ij = (lij-j) + j*stride_j;
                            s1x = _gx[ij*nst];
                            for (--ij; ij >= j*stride_j; --ij) {
                                s0x = _gx[ij*nst];
                                _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x;
                                s1x = s0x;
                            }
                        }
                    }
                }

                __syncthreads();
                if (ijk_idx < nksp) {
                    int nfi = c_nf[li];
                    int nfj = c_nf[lj];
                    float div_nfi = c_div_nf[li];
                    float div_nfj = c_div_nf[lj];
                    double ak2 = ak * 2;
#pragma unroll
                    for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t jk = ijk * div_nfi;
                        uint32_t i = ijk - jk * nfi;
                        uint32_t k = jk * div_nfj;
                        uint32_t j = jk - k * nfj;
                        int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                        int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                        int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                        int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                        int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                        int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                        int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                        int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                        int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                        int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                        int addry = (iy + jy*stride_j + ky*stride_k + gsize) * nst;
                        int addrz = (iz + jz*stride_j + kz*stride_k + gsize*2) * nst;
                        double gkx = gx[addrx+k_1];
                        double gky = gx[addry+k_1];
                        double gkz = gx[addrz+k_1];
                        double fkx = ak2 * gkx; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; }
                        double fky = ak2 * gky; if (ky > 0) { fky -= ky * gx[addry-k_1]; }
                        double fkz = ak2 * gkz; if (kz > 0) { fkz -= kz * gx[addrz-k_1]; }
                        goutx[n] += fkx * gx[addry] * gx[addrz];
                        gouty[n] += gx[addrx] * fky * gx[addrz];
                        goutz[n] += gx[addrx] * gx[addry] * fkz;
                    }
                }
            }
        }

        if (ijk_idx < nksp) {
            int nfk = c_nf[lk];
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            size_t offset = nao_pairs * naux;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < GOUT_IP1_WIDTH; ++n) {
                int ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                int ij = ijk / nfk;
                int k  = ijk - nfk * ij;
                j3c_tensor[            ij*naux + k*nksh] = goutx[n];
                j3c_tensor[offset    + ij*naux + k*nksh] = gouty[n];
                j3c_tensor[offset *2 + ij*naux + k*nksh] = goutz[n];
            }
        }
    }
}

extern "C" {
int fill_int3c2e_ip1(double *out, RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                 int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                 int *ksh_offsets, int *gout_stride_lookup, int *ao_pair_loc,
                 int ao_pair_offset, int aux_offset, int nao_pairs, int naux)
{
    cudaFuncSetAttribute(int3c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    int3c2e_ip1_kernel<<<blocks, THREADS, shm_size>>>(
            out, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, ao_pair_offset, aux_offset,
            nao_pairs, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int fill_int3c2e_ipaux(double *out, RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                 int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                 int *ksh_offsets, int *gout_stride_lookup, int *ao_pair_loc,
                 int ao_pair_offset, int aux_offset, int nao_pairs, int naux)
{
    cudaFuncSetAttribute(int3c2e_ipaux_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    int3c2e_ipaux_kernel<<<blocks, THREADS, shm_size>>>(
            out, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, ao_pair_offset, aux_offset,
            nao_pairs, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
