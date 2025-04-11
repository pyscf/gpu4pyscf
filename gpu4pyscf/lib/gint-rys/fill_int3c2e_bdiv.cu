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
#include "gvhf-rys/rys_roots.cu"
#include "int3c2e.cuh"

// TODO: benchmark performance for 32, 38, 40, 45, 54
#define GOUT_WIDTH      45

__device__ int int3c2e_bdiv_unrolled(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds);

__global__
void int3c2e_bdiv_kernel(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    if (int3c2e_bdiv_unrolled(out, envs, bounds)) {
        return;
    }
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int lk = bas[ksh0*BAS_SLOTS+ANG_OF];
    int lij = li + lj;
    int nroots = (lij + lk) / 2 + 1;
    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfk = (lk + 1) * (lk + 2) / 2;
    int nfij = nfi * nfj;
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int lk_offset = lk * (lk + 1) * (lk + 2) / 2;
    int *idx_k = c_g_cart_idx + lk_offset;
    int *idy_k = idx_k + nfk;
    int *idz_k = idy_k + nfk;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;

    int nst_per_block = blockDim.x;
    if (lij + lk > 2) {
        nst_per_block = bounds.nst_lookup[(lk*LMAX1+lj)*LMAX1+li];
    }
    int gout_stride = blockDim.x / nst_per_block;
    int thread_id = threadIdx.x;
    int st_id = thread_id % nst_per_block;
    int gout_id = thread_id / nst_per_block;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    int gx_len = g_size * nst_per_block;
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *g = rw + nst_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    double *Rpq = gz + gx_len;
    double *rjri = Rpq + nst_per_block * 3;
    double gout[GOUT_WIDTH];
    int naux = bounds.naux;
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
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
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

        for (int gout_start = 0; gout_start < nfij*nfk;
             gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
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
                    gy[0] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nst_per_block];
                    }
                    double rt = rw[ irys*2   *nst_per_block];
                    double rt_aa = rt / (aij + ak);

                    if (lij > 0) {
                        __syncthreads();
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
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
                            double *_gx = g + (_ix*g_size + k*stride_k) *
                                nst_per_block;
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
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        int ijk = gout_start + n*gout_stride+gout_id;
                        int k  = ijk % nfk;
                        int ij = ijk / nfk;
                        if (ij >= nfij) break;
                        int addrx = (idx_ij[ij] + idx_k[k] * stride_k) * nst_per_block;
                        int addry = (idy_ij[ij] + idy_k[k] * stride_k) * nst_per_block;
                        int addrz = (idz_ij[ij] + idz_k[k] * stride_k) * nst_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }

            if (ijk_idx < nst) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh0] - ao_loc[nbas];
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
