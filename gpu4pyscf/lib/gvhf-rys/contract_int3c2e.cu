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
#include "gint/cuda_alloc.cuh"
#include "vhf.cuh"
#include "rys_roots.cu"
#include "rys_contract_k.cuh"
#include "create_tasks.cu"

#define THREADS         256
#define BLOCK_SIZE      16
#define NF_AUX_MAX      28
#define AUX_CACHE_SIZE  30
#define IJ_WIDTH        30

template <int NF, int N_DM> __device__ __forceinline__
void _dot_dm(double *vj_xyz, double *gout, double *dm, uint32_t n_dm, uint32_t nao2)
{
#pragma unroll
    for (uint32_t m = 0; m < N_DM; ++m) {
        if (m >= n_dm) break;
        double dm_ij = dm[m*nao2];
#pragma unroll
        for (uint32_t k = 0; k < NF; k++) {
            vj_xyz[m*NF+k] += gout[k] * dm_ij;
        }
    }
}

template <int NF, int N_DM> __device__ __forceinline__
void _store_vj(double *out, double *vj_xyz, uint32_t n_dm, uint32_t naux)
{
#pragma unroll
    for (uint32_t m = 0; m < N_DM; ++m) {
        if (m >= n_dm) break;
#pragma unroll
        for (uint32_t k = 0; k < NF; k++) {
            atomicAdd(out+m*naux+k, vj_xyz[m*NF+k]);
        }
    }
}

__global__ static
void contract_int3c2e_dm_kernel(double *out, double *dm, int n_dm, int naux,
                                RysIntEnvVars envs, int *shl_pair_offsets,
                                uint32_t *bas_ij_idx, int *gout_stride_lookup
                                #ifdef USE_SYCL
                                , sycl::nd_item<2> &item, char *shm_mem
                                #endif
                                )
{
    #ifdef USE_SYCL
    int threadIdx_x = item.get_local_id(1);
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int gridDim_y = item.get_group_range(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &g_size = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    uint32_t &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nsp_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &dm_id0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double &xk = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yk = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zk = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    int &expk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ck = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int threadIdx_x = threadIdx.x;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int gridDim_y = gridDim.y;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ uint32_t nao;
    __shared__ int gout_stride, nsp_per_block;
    __shared__ int dm_id0;
    extern __shared__ double shared_memory[];
    __shared__ double xk, yk, zk;
    __shared__ int expk, ck;
    #endif

    int thread_id = threadIdx_x;
    int nbas = envs.nbas;
    int ksh = blockIdx_x + nbas;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (thread_id == 0) {
        int sp_block_id = gridDim_y - blockIdx_y - 1;
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
        int stride_j = li + 1;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 1);
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nsp_per_block = THREADS / gout_stride;
        dm_id0 = 0;
    }
    __syncthreads();
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;

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



    if (thread_id == 0) {
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        xk = rk[0];
        yk = rk[1];
        zk = rk[2];
        expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
    }

while (dm_id0 < n_dm) {
    double gout[NF_AUX_MAX];
    double vj_xyz[AUX_CACHE_SIZE];
    for (int n = 0; n < AUX_CACHE_SIZE; ++n) {
        vj_xyz[n] = 0;
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
        uint32_t i0 = ao_loc[ish];
        uint32_t j0 = ao_loc[jsh];
        uint32_t dm_off = (dm_id0 * nao + j0) * nao + i0;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = env[ri+0];
        double yi = env[ri+1];
        double zi = env[ri+2];
        double xjxi = env[rj+0] - xi;
        double yjyi = env[rj+1] - yi;
        double zjzi = env[rj+2] - zi;
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
            double ai = env[expi+ip];
            double aj = env[expj+jp];
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
                double cicj = fac_ij * env[ci+ip] * env[cj+jp];
                gx[0] = cicj * exp(-Kab);
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
            }
            for (int kp = 0; kp < kprim; ++kp) {
                __syncthreads();
                double ak = env[expk+kp];
                double theta = aij * ak / (aij + ak);
                if (gout_id == 0) {
                    gx[gx_len] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                }
                rys_roots_rs(nroots, theta, rr, omega, rw, nsp_per_block,
                             gout_id, gout_stride);
                for (int irys = 0; irys < nroots; ++irys) {
                    int stride_j = li + 1;
                    int stride_k = stride_j * (lj + 1);
                    int nsp = nsp_per_block;
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp];
                    }
                    double rt = rw[ irys*2   *nsp];
                    double rt_aa = rt / (aij + ak);
                    double s0x, s1x, s2x;
                    int lij = li + lj;
                    if (lij > 0) {
                        __syncthreads();
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double xpa = rjri[n*nsp] * aj_aij;
                            double c0x = xpa - rt_aij * Rpq[n*nsp];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsp] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsp] = s2x;
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
                            double *_gx = gx + (i + _ix * g_size) * nsp;
                            double cpx = rt_ak * Rpq[_ix*nsp];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp];
                                }
                                _gx[stride_k*nsp] = s1x;
                            }
                            for (int k = 1; k < lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsp];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsp] = s2x;
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
                                double xjxi = rjri[_ix*nsp];
                                double *_gx = gx + (_ix*g_size + k*stride_k) * nsp;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = lij + j*li; // = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nsp];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nsp];
                                        _gx[(ij+stride_j)*nsp] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (pair_ij < shl_pair1) {
                        int nfi = c_nf[li];
                        int nfj = c_nf[lj];
                        int nfij = nfi * nfj;
                        float div_nfi = c_div_nf[li];
                        for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ij_addrx = idx_i[i*3+0] + idx_j[j*3+0];
                            int ij_addry = idx_i[i*3+1] + idx_j[j*3+1];
                            int ij_addrz = idx_i[i*3+2] + idx_j[j*3+2];
                            double dm_ij = dm[dm_off+j*nao+i];
#pragma unroll
                            for (int k = 0; k < NF_AUX_MAX; ++k) {
                                if (k >= nfk) break;
                                int addrx = ij_addrx + idx_k[k*3+0];
                                int addry = ij_addry + idx_k[k*3+1];
                                int addrz = ij_addrz + idx_k[k*3+2];
                                gout[k] = gx[addrx] * gx[addry] * gx[addrz];
                                vj_xyz[k] += gout[k] * dm_ij;
                            }
                            int dm_remaining = n_dm - dm_id0 - 1;
                            if (dm_remaining > 0) {
                                uint32_t nao2 = nao * nao;
                                double *dm_local = dm + dm_off+nao2 + j*nao+i;
                                switch (lk) {
                                case 0: _dot_dm<1, 9>(vj_xyz+1 , gout, dm_local, dm_remaining, nao2); break;
                                case 1: _dot_dm<3, 9>(vj_xyz+3 , gout, dm_local, dm_remaining, nao2); break;
                                case 2: _dot_dm<6, 4>(vj_xyz+6 , gout, dm_local, dm_remaining, nao2); break;
                                case 3: _dot_dm<10,2>(vj_xyz+10, gout, dm_local, dm_remaining, nao2); break;
                                case 4: _dot_dm<15,1>(vj_xyz+15, gout, dm_local, dm_remaining, nao2); break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int k0 = ao_loc[ksh] - ao_loc[nbas];
    double *vj = out + k0 + dm_id0 * naux;
    int dm_remaining = n_dm - dm_id0;
    switch (lk) {
    case 0: _store_vj<1,10>(vj, vj_xyz, dm_remaining, naux); break;
    case 1: _store_vj<3,10>(vj, vj_xyz, dm_remaining, naux); break;
    case 2: _store_vj<6, 5>(vj, vj_xyz, dm_remaining, naux); break;
    case 3: _store_vj<10,3>(vj, vj_xyz, dm_remaining, naux); break;
    case 4: _store_vj<15,2>(vj, vj_xyz, dm_remaining, naux); break;
    case 5: _store_vj<21,1>(vj, vj_xyz, dm_remaining, naux); break;
    case 6: _store_vj<28,1>(vj, vj_xyz, dm_remaining, naux); break;
    }
    __syncthreads();
    if (thread_id == 0) {
        switch (lk) {
        case 0: dm_id0 +=10; break;
        case 1: dm_id0 +=10; break;
        case 2: dm_id0 += 5; break;
        case 3: dm_id0 += 3; break;
        case 4: dm_id0 += 2; break;
        default: dm_id0 += 1;
        }
    }
    __syncthreads();
}
}

__global__ static
void contract_int3c2e_auxvec_kernel(double *out, double *auxvec, RysIntEnvVars envs,
                                    int *shl_pair_offsets, uint32_t *bas_ij_idx,
                                    int *ksh_offsets, int *gout_stride_lookup
                                    #ifdef USE_SYCL
                                    , sycl::nd_item<2> &item, char *shm_mem
                                    #endif
                                    )
{
    // For better load balance, consume blocks in the reversed order
    #ifdef USE_SYCL
    int threadIdx_x = item.get_local_id(1);
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int gridDim_x = item.get_group_range(1);
    int gridDim_y = item.get_group_range(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ksh0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ksh1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nfij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    double (&vj_aux)[NF_AUX_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[NF_AUX_MAX]>(thread_block);
    #else
    int threadIdx_x = threadIdx.x;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int gridDim_x = gridDim.x;
    int gridDim_y = gridDim.y;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nfij;
    __shared__ int gout_stride;
    extern __shared__ double shared_memory[];
    __shared__ double vj_aux[NF_AUX_MAX];
    #endif

    int thread_id = threadIdx_x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (thread_id == 0) {
        int sp_block_id = gridDim_x - blockIdx_x - 1;
        int ksh_block_id = gridDim_y - blockIdx_y - 1;
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
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        nfij = nfi * nfj;
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nsp_per_block;

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
                            float div_nfi = c_div_nf[li];
#pragma unroll
                            for (int n = 0; n < IJ_WIDTH; ++n) {
                                uint32_t ij = gout_id + n * gout_stride;
                                if (ij >= nfij) break;
                                uint32_t j = ij * div_nfi;
                                uint32_t i = ij - nfi * j;
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
int contract_int3c2e_dm(double *out, double *dm, int n_dm, int naux,
                        RysIntEnvVars *envs, int shm_size,
                        int nbas_aux, int nbatches_shl_pair, int *shl_pair_offsets,
                        uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
#ifdef USE_SYCL
    sycl::range<2> threads(1, THREADS);
    sycl::range<2> blocks(nbatches_shl_pair, nbas_aux);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class contract_int3c2e_dm_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        contract_int3c2e_dm_kernel(
            out, dm, n_dm, naux, dev_envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
#else
    cudaFuncSetAttribute(contract_int3c2e_dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbas_aux, nbatches_shl_pair);
    contract_int3c2e_dm_kernel<<<blocks, THREADS, shm_size>>>(
            out, dm, n_dm, naux, *envs, shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
    return 0;
}

// contract('ijP,P->ij', int3c2e, auxvec)
int contract_int3c2e_auxvec(double *vj, double *auxvec, RysIntEnvVars *envs, int shm_size,
                            int nbatches_shl_pair, int nbatches_ksh,
                            int *shl_pair_offsets, int *ksh_offsets,
                            uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
#ifdef USE_SYCL
    sycl::range<2> threads(1, THREADS);
    sycl::range<2> blocks(nbatches_ksh, nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class contract_int3c2e_auxvec_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        contract_int3c2e_auxvec_kernel(
            vj, auxvec, dev_envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
#else
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
#endif
    return 0;
}

}
