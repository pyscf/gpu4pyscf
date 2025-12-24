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
#define GOUT_WIDTH      54

    __global__ static
void int3c2e_kernel(double *out, RysIntEnvVars envs, int *shl_pair_offsets,
                    uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int ao_pair_offset, int aux_offset, int naux)
{
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1, nshl_pair;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lij, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nfi, nfj, nfk, nfij, nf, nao;
    __shared__ int gout_stride;
    __shared__ double omega;
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
        nroots = (lij + lk) / 2 + 1;
        omega = env[PTR_RANGE_OMEGA];
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nfi = (li + 1) * (li + 2) / 2;
        nfj = (lj + 1) * (lj + 2) / 2;
        nfk = (lk + 1) * (lk + 2) / 2;
        nfij = nfi * nfj;
        nf = nfij * nfk;
        nao = envs.ao_loc[nbas];
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    }
    __syncthreads();
    int nst_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;

    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 4 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
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
    if (gout_id == 0) {
        gx[gx_len] = PI_FAC;
    }

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += nst_per_block) {
        // convert task_id to ish, jsh, ksh
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
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
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            __syncthreads();
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double rr_ij = rjri[3*nst_per_block];
                double Kab = theta_ij * rr_ij;
                gx[0] = fac * exp(-Kab);
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
                    gx[g_size*nst_per_block*2] = rw[(irys*2+1)*nst_per_block];
                }
                double rt = rw[ irys*2   *nst_per_block];
                double rt_aa = rt / (aij + ak);

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
                        int ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        int ij = ijk / nfk;
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
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                int ij = ijk / nfk;
                int k  = ijk - nfk * ij;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        }
    }
}

extern "C" {
int fill_int3c2e(double *out, RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                 int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                 int *ksh_offsets, int *gout_stride_lookup, int *ao_pair_loc,
                 int ao_pair_offset, int aux_offset, int naux)
{
    cudaFuncSetAttribute(int3c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    int3c2e_kernel<<<blocks, THREADS, shm_size>>>(
            out, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, ao_pair_offset, aux_offset, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
