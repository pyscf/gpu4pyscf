/*
 * Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gint/cuda_alloc.cuh"
#include "vhf.cuh"
#include "rys_roots_for_k.cu"
#include "create_tasks.cu"
#include "rys_contract_k.cuh"
#include "build_rys_gxyz.cuh"

#define GOUT_WIDTH      81

__device__ static
void _fill_tasks(int& ntasks, int& pair_kl0, uint32_t *pair_kl_idx,
                 int pair_ij, int ish, int jsh,
                 float *q_cond_ij, float *q_cond_kl, int *swap,
                 RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float kl_cutoff = cutoff - q_ij;
    if (q_cond_kl[pair_kl0] + Q_COND_MARGIN < kl_cutoff) {
        return;
    }

    int pair_kl1 = min(pair_kl0 + (QUEUE_DEPTH - 512), bounds.npairs_kl);

    while (pair_kl0 < pair_kl1 && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        int keep = 0;
        if (pair_kl < pair_kl1) {
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = pair_kl1;
            }
            keep = q_kl >= kl_cutoff;
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            pair_kl_idx[ntasks + offset] = pair_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    // pad data to avoid overflow
    if (threadIdx.y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        pair_kl_idx[ntasks+t_id] = pair_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__global__ static
void int4c2e_kernel(double *out, RysIntEnvVars envs, BoundsInfo bounds,
                    float *q_cond_ij, float *q_cond_kl,
                    int *ij_pair_loc, int *kl_pair_loc,
                    uint32_t *pool, int *head, int nf, int nao_pairs,
                    int reserved_shm_size, double omega)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    uint32_t *pair_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    extern __shared__ double shared_memory[];
    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish, jsh, ij0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;

    int t_id = gout_id * nsq_per_block + sq_id;
    int threads = nsq_per_block * gout_stride;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = bounds.g_size;
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *gx = shared_memory + nsq_per_block * 6 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+6) + sq_id;
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    double *cicj_cache = shared_memory + reserved_shm_size - iprim*jprim;
    int *idx_i = (int*)(shared_memory + reserved_shm_size);
    int *idx_j = idx_i + ntiles_i * 9;
    int *idx_k = idx_j + ntiles_j * 9;
    int *idx_l = idx_k + ntiles_k * 9;
    if (t_id < ntiles_i * 9) {
        idx_i[t_id] = lex_xyz_address(li, t_id) * nsq_per_block;
        idx_i[t_id] += (t_id % 3) * nsq_per_block * g_size;
    }
    if (t_id < ntiles_j * 9) {
        idx_j[t_id] = lex_xyz_address(lj, t_id) * stride_j * nsq_per_block;
    }
    if (t_id < ntiles_k * 9) {
        idx_k[t_id] = lex_xyz_address(lk, t_id) * stride_k * nsq_per_block;
    }
    if (t_id < ntiles_l * 9) {
        idx_l[t_id] = lex_xyz_address(ll, t_id) * stride_l * nsq_per_block;
    }
while (1) {
    __syncthreads();
    if (t_id == 0) {
        int task_id = atomicAdd(head, 1);
        int batch_kl = task_id / bounds.npairs_ij;
        pair_ij = task_id - bounds.npairs_ij * batch_kl;
        pair_kl0 = batch_kl * (QUEUE_DEPTH - 512);
        uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        ij0 = ij_pair_loc[pair_ij];
    }
    __syncthreads();
    if (pair_kl0 >= bounds.npairs_kl) {
        break;
    }
    _fill_tasks(ntasks, pair_kl0, pair_kl_idx, pair_ij, ish, jsh,
                q_cond_ij, q_cond_kl, (int *)shared_memory, envs, bounds);
    if (ntasks == 0) {
        continue;
    }

    if (t_id == 0) {
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }
    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int li = bounds.li;
        int lj = bounds.lj;
        int lk = bounds.lk;
        int ll = bounds.ll;
        int iprim = bounds.iprim;
        int jprim = bounds.jprim;
        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        int stride_j = bounds.stride_j;
        int stride_k = bounds.stride_k;
        int stride_l = bounds.stride_l;
        int g_size = bounds.g_size;

        int pair_kl = pair_kl_idx[task_id];
        uint32_t bas_kl = bounds.pair_kl_mapping[pair_kl];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }

        for (int gout_start = 0; gout_start < nf; gout_start+=gout_stride*GOUT_WIDTH) {
            double gout[GOUT_WIDTH];
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                if (gout_id == 0) {
                    double xlxk = rlrk[0*nsq_per_block];
                    double ylyk = rlrk[1*nsq_per_block];
                    double zlzk = rlrk[2*nsq_per_block];
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    gx[0] = PI_FAC * ckcl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + (rjri[0]) * aj_aij;
                    double yij = ri[1] + (rjri[1]) * aj_aij;
                    double zij = ri[2] + (rjri[2]) * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double theta = aij * akl / (aij + akl);
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, omega, 1., 1.);
                    for (int irys = 0; irys < nroots; ++irys) {
                        BUILD_4C_GXYZ(lj, ll, task_id < ntasks);
                        if (task_id >= ntasks) {
                            continue;
                        }
                        int nfi = bounds.nfi;
                        int nfj = bounds.nfj;
                        int nfk = bounds.nfk;
                        float div_nfi = c_div_nf[li];
                        float div_nfj = c_div_nf[lj];
                        float div_nfk = c_div_nf[lk];
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            uint32_t ijkl = gout_start + n*gout_stride+gout_id;
                            if (ijkl >= nf) break;
                            uint32_t jkl = ijkl * div_nfi;
                            uint32_t i = ijkl - jkl * nfi;
                            uint32_t kl = jkl * div_nfj;
                            uint32_t j = jkl - kl * nfj;
                            uint32_t l = kl * div_nfk;
                            uint32_t k = kl - l * nfk;
                            int addrx = idx_i[i*3+0] + idx_j[j*3+0] + idx_k[k*3+0] + idx_l[l*3+0];
                            int addry = idx_i[i*3+1] + idx_j[j*3+1] + idx_k[k*3+1] + idx_l[l*3+1];
                            int addrz = idx_i[i*3+2] + idx_j[j*3+2] + idx_k[k*3+2] + idx_l[l*3+2];
                            gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                        }
                    }
                }
            }
            __syncthreads();

            if (task_id < ntasks) {
                int kl0 = kl_pair_loc[pair_kl];
                int nfij = bounds.nfi * bounds.nfj;
                float div_nfij = c_div_nf[li] * c_div_nf[lj];
                size_t N = nao_pairs;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijkl = gout_start + n*gout_stride+gout_id;
                    if (ijkl >= nf) break;
                    int kl = ijkl * div_nfij;
                    int ij = ijkl - kl * nfij;
                    out[(ij0+ij) * N + (kl0+kl)] = gout[n];
                }
            }
        }
    }
}
}

static void threads_scheme(int *scheme, BoundsInfo &bounds, int shm_size, int nf)
{
    int ijprim = bounds.iprim * bounds.jprim;
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
    int g_size = bounds.g_size;
    int nroots = bounds.nroots;
    int root_g_cache_size = nroots*2 + g_size*3 + 6;
    int unit = root_g_cache_size;
    int counts = (shm_size - cart_idx_size*4 - ijprim*8) / (unit*8);
    int nbatches = (nf + GOUT_WIDTH - 1) / GOUT_WIDTH;
    int THREADS = 256;
    int gout_stride = 1;
    while (nbatches > 1 && gout_stride < 32) {
        gout_stride *= 2;
        nbatches = (nbatches + 1) / 2;
    }
    int nsq_per_block = min(counts, THREADS / gout_stride);
    int reserved_shm_size = nsq_per_block * unit + ijprim;
    scheme[0] = nsq_per_block;
    scheme[1] = gout_stride;
    scheme[2] = reserved_shm_size*8 + cart_idx_size*4;
    scheme[3] = reserved_shm_size;
}

extern "C" {
int RYS_fill_int4c2e(double *out, int nao_pairs, double omega,
                     RysIntEnvVars *envs, int *shls_slice, int shm_size,
                     int npairs_ij, int npairs_kl,
                     uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
                     float *q_cond_ij, float *q_cond_kl,
                     int *ij_pair_loc, int *kl_pair_loc,
                     float cutoff, uint32_t *pool, int *bas)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    int jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int ntiles_i = (nfi + 2) / 3;
    int ntiles_j = (nfj + 2) / 3;
    int ntiles_k = (nfk + 2) / 3;
    int ntiles_l = (nfl + 2) / 3;
    int order = li + lj + lk + ll;
    int nroots = order / 2 + 1;
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size = stride_l * (ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        NULL, NULL, NULL, cutoff,
        ntiles_i, ntiles_j, ntiles_k, ntiles_l};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    int *head = (int *)(pool + workers * QUEUE_DEPTH);
    cudaMemset(head, 0, sizeof(int));

    int nf = nfi * nfj * nfk * nfl;
    int scheme[4];
    threads_scheme(scheme, bounds, shm_size, nf);
    dim3 threads(scheme[0], scheme[1]);
    int buflen = scheme[2];
    int reserved_shm_size = scheme[3];
    cudaFuncSetAttribute(int4c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", buflen,
                cudaGetErrorString(err));
        return 1;
    }

    int4c2e_kernel<<<workers, threads, buflen>>>(
        out, *envs, bounds, q_cond_ij, q_cond_kl, ij_pair_loc, kl_pair_loc,
        pool, head, nf, nao_pairs, reserved_shm_size, omega);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int4c2e_kernel, li,lj,lk,ll = %d,%d,%d,%d, error message = %s\n", li,lj,lk,ll, cudaGetErrorString(err));
        fflush(stderr);
        return 1;
    }
    return 0;
}
}
