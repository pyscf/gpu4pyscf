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
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"
//#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "int3c2e.cuh"

#define GOUT_WIDTH      54

//__device__ int int3c2e_bdiv_unrolled(double *out, Int3c2eEnvVars& envs, BDiv3c2eBounds& bounds);
#include "unrolled_int3c2e_bdiv.cu"

__global__
void int3c2e_bdiv_kernel(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    if (int3c2e_bdiv_unrolled(out, envs, bounds)) {
        return;
    }
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    __shared__ int li, lj, lk, nroots;
    __shared__ int nfi, nfj, nfk, nfij;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nshl_pair, nksh;
    __shared__ int stride_j, stride_k, g_size;
    if (thread_id == 0) {
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        nroots = (li + lj + lk) / 2 + 1;
        nfi = (li + 1) * (li + 2) / 2;
        nfj = (lj + 1) * (lj + 2) / 2;
        nfk = (lk + 1) * (lk + 2) / 2;
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nksh = ksh1 - ksh0;
        nshl_pair = shl_pair1 - shl_pair0;
        stride_j = li + 1;
        stride_k = stride_j * (lj + 1);
        nfij = nfi * nfj;
        g_size = stride_k * (lk + 1);
    }
    __syncthreads();
    int lij = li + lj;
    int nst_per_block = blockDim.x;
    if (lij + lk > 2) {
        nst_per_block = bounds.nst_lookup[(lk*LMAX1+lj)*LMAX1+li];
    }
    int gout_stride = blockDim.x / nst_per_block;
    int st_id = thread_id % nst_per_block;
    int gout_id = thread_id / nst_per_block;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 4 + st_id;
    double *rw = shared_memory + nst_per_block * 7 + st_id;
    double *gx = shared_memory + nst_per_block * (nroots*2+7) + st_id;
    int *idx_i = (int*)(shared_memory + nst_per_block*(g_size*3+nroots*2+7));
    int *idx_j = idx_i + nfi * 3;
    int *idx_k = idx_j + nfj * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nst_per_block;
        idx_i[thread_id] += (thread_id % 3) * nst_per_block * g_size;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nst_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nst_per_block;
    }

    double gout[GOUT_WIDTH];
    size_t naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    if (gout_id == 0) {
        gx[0] = 1.;
    }

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += nst_per_block) {
        // convert task_id to ish, jsh, ksh
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nst_per_block] = xjxi;
            rjri[1*nst_per_block] = yjyi;
            rjri[2*nst_per_block] = zjzi;
            rjri[3*nst_per_block] = rr_ij;
        }

        for (int gout_start = 0; gout_start < nfij*nfk; gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            int ijprim = iprim * jprim;
            int ijkprim = ijprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp % kprim;
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                __syncthreads();
                double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = ci[ip] * cj[jp] * ck[kp];
                    double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nst_per_block];
                    gx[g_size*nst_per_block] = fac * exp(-Kab);
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
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[g_size*nst_per_block*2] = rw[(irys*2+1)*nst_per_block];
                    }
                    double rt = rw[ irys*2   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    int lij = li + lj;

                    if (lij > 0) {
                        __syncthreads();
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * g_size * nst_per_block;
                            double xjxi = rjri[n*nst_per_block];
                            double xpa = xjxi * aj_aij;
                            //double c0x = Rpa[ir] - rt_aij * Rpq[n*nst_per_block];
                            double c0x = xpa - rt_aij * Rpq[n*nst_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nst_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nst_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
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
                            double *_gx = gx + (i + _ix * g_size) * nst_per_block;
                            double cpx = rt_ak * Rpq[_ix*nst_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nst_per_block];
                                }
                                _gx[stride_k*nst_per_block] = s1x;
                            }
                            //for k in range(1, lk):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nst_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nst_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    if (lj > 0) {
                        __syncthreads();
                        int lk3 = (lk+1)*3;
                        for (int m = gout_id; m < lk3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xjxi = rjri[_ix*nst_per_block];
                            double *_gx = gx + (_ix*g_size + k*stride_k) * nst_per_block;
                            for (int j = 0; j < lj; ++j) {
                                int ij = (lij-j) + j*stride_j;
                                s1x = _gx[ij*nst_per_block];
                                for (--ij; ij >= j*stride_j; --ij) {
                                    s0x = _gx[ij*nst_per_block];
                                    _gx[(ij+stride_j)*nst_per_block] = s1x - xjxi * s0x;
                                    s1x = s0x;
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (ijk_idx < nst) {
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            int ijk = gout_start + n*gout_stride+gout_id;
                            int ij = ijk / nfk;
                            if (ij >= nfij) break;
                            int k = ijk % nfk;
                            int i = ij % nfi;
                            int j = ij / nfi;
                            int addrx = idx_i[i*3+0] + idx_j[j*3+0] + idx_k[k*3+0];
                            int addry = idx_i[i*3+1] + idx_j[j*3+1] + idx_k[k*3+1];
                            int addrz = idx_i[i*3+2] + idx_j[j*3+2] + idx_k[k*3+2];
                            gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                        }
                    }
                }
            }

            if (ijk_idx < nst) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh0] - ao_loc[bounds.aux_sh_offset];
                double *eri_tensor = out_local + shl_pair_in_block * nfij * naux
                        + k0 + ksh_in_block * nfk;
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijk = gout_start + n*gout_stride+gout_id;
                    int k  = ijk % nfk;
                    int ij = ijk / nfk;
                    if (ij >= nfij) break;
                    eri_tensor[ij * naux + k] = gout[n];
                }
            }
        }
    }
}
