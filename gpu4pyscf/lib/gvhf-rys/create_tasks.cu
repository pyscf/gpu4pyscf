/*
 * Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
#include <string.h>

#include "vhf.cuh"

#define Q_COND_MARGIN   4.f

// np.where(threads_mask)[0]
__device__ inline
int mask_to_index(int keep, int *tmp_storage, int threads, int t_id)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    #endif
    tmp_storage[t_id] = keep;
    __syncthreads();
    for (int offset = 1; offset < threads; offset <<= 1) {
        int val = 0;
        if (t_id >= offset) {
            val = tmp_storage[t_id - offset];
        }
        __syncthreads();
        tmp_storage[t_id] += val;
        __syncthreads();
    }
    int offset = tmp_storage[t_id] - keep;
    return offset;
}

__device__ static
void _fill_vk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                    int pair_ij, int ish, int jsh,
                    float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                    RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float kl_cutoff = cutoff - q_ij;
    uint32_t bas_ij = ish * nbas + jsh;

    int *swap = (int *)shared_memory;

    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                float d_cutoff = kl_cutoff - q_kl;
                keep = (dm_cond[ish*nbas+ksh] > d_cutoff ||
                        dm_cond[jsh*nbas+ksh] > d_cutoff ||
                        dm_cond[ish*nbas+lsh] > d_cutoff ||
                        dm_cond[jsh*nbas+lsh] > d_cutoff);
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    // pad data to avoid overflow
    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_vjk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                     int pair_ij, int ish, int jsh,
                     float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                     RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif
    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                float d_cutoff = kl_cutoff - q_kl;
                keep = (d_ij                  > d_cutoff ||
                        dm_cond[bas_kl]       > d_cutoff ||
                        dm_cond[ish*nbas+ksh] > d_cutoff ||
                        dm_cond[jsh*nbas+ksh] > d_cutoff ||
                        dm_cond[ish*nbas+lsh] > d_cutoff ||
                        dm_cond[jsh*nbas+lsh] > d_cutoff);
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_vj_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                    int pair_ij, int ish, int jsh,
                    float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                    RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif
    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                float d_cutoff = kl_cutoff - q_kl;
                keep = (d_ij            > d_cutoff ||
                        dm_cond[bas_kl] > d_cutoff);
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif
    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_vk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                       int pair_ij, int ish, int jsh,
                       float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                       float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                       RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    double *env = envs.env;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float xpa = xjxi * aj_aij;
    float ypa = yjyi * aj_aij;
    float zpa = zjzi * aj_aij;
    float xij = xi + xpa;
    float yij = yi + ypa;
    float zij = zi + zpa;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float s_ij = s_cond_ij[pair_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);
    uint32_t bas_ij = ish * nbas + jsh;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                float ak = diffuse_exps[ksh];
                float al = diffuse_exps[lsh];
                float akl = ak + al;
                float al_akl = al / akl;
                float xk = rk[0];
                float yk = rk[1];
                float zk = rk[2];
                float xl = rl[0];
                float yl = rl[1];
                float zl = rl[2];
                float xlxk = xl - xk;
                float ylyk = yl - yk;
                float zlzk = zl - zk;
                float xqc = xlxk * al_akl;
                float yqc = ylyk * al_akl;
                float zqc = zlzk * al_akl;
                float xkl = xk + xqc;
                float ykl = yk + yqc;
                float zkl = zk + zqc;
                float theta = theta_ij * akl / (theta_ij + akl);
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                float theta_rr = logf(rr + 1.f) + theta * rr;
                float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                keep &= dm_penalty > d_cutoff;
                if (keep) {
                    d_cutoff = max(kl_cutoff - q_kl, d_cutoff);
                    keep = (dm_cond[jsh*nbas+ksh] > d_cutoff ||
                            dm_cond[jsh*nbas+lsh] > d_cutoff ||
                            dm_cond[ish*nbas+ksh] > d_cutoff ||
                            dm_cond[ish*nbas+lsh] > d_cutoff);
                }
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif
    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_vjk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                        int pair_ij, int ish, int jsh,
                        float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                        float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                        RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    double *env = envs.env;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float xpa = xjxi * aj_aij;
    float ypa = yjyi * aj_aij;
    float zpa = zjzi * aj_aij;
    float xij = xi + xpa;
    float yij = yi + ypa;
    float zij = zi + zpa;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float s_ij = s_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
      if (threadIdx_y == active_y) {
    #endif
    int *swap = (int *)shared_memory;

    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                float ak = diffuse_exps[ksh];
                float al = diffuse_exps[lsh];
                float akl = ak + al;
                float al_akl = al / akl;
                float xk = rk[0];
                float yk = rk[1];
                float zk = rk[2];
                float xl = rl[0];
                float yl = rl[1];
                float zl = rl[2];
                float xlxk = xl - xk;
                float ylyk = yl - yk;
                float zlzk = zl - zk;
                float xqc = xlxk * al_akl;
                float yqc = ylyk * al_akl;
                float zqc = zlzk * al_akl;
                float xkl = xk + xqc;
                float ykl = yk + yqc;
                float zkl = zk + zqc;
                float theta = theta_ij * akl / (theta_ij + akl);
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                float theta_rr = logf(rr + 1.f) + theta * rr;
                float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                keep &= dm_penalty > d_cutoff;
                if (keep) {
                    d_cutoff = max(kl_cutoff - q_kl, d_cutoff);
                    keep = (d_ij > d_cutoff ||
                            dm_cond[bas_kl] > d_cutoff ||
                            dm_cond[ish*nbas+ksh] > d_cutoff ||
                            dm_cond[jsh*nbas+ksh] > d_cutoff ||
                            dm_cond[ish*nbas+lsh] > d_cutoff ||
                            dm_cond[jsh*nbas+lsh] > d_cutoff);
                }
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_vj_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                       int pair_ij, int ish, int jsh,
                       float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                       float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                       RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    double *env = envs.env;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float xpa = xjxi * aj_aij;
    float ypa = yjyi * aj_aij;
    float zpa = zjzi * aj_aij;
    float xij = xi + xpa;
    float yij = yi + ypa;
    float zij = zi + zpa;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float s_ij = s_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                float ak = diffuse_exps[ksh];
                float al = diffuse_exps[lsh];
                float akl = ak + al;
                float al_akl = al / akl;
                float xk = rk[0];
                float yk = rk[1];
                float zk = rk[2];
                float xl = rl[0];
                float yl = rl[1];
                float zl = rl[2];
                float xlxk = xl - xk;
                float ylyk = yl - yk;
                float zlzk = zl - zk;
                float xqc = xlxk * al_akl;
                float yqc = ylyk * al_akl;
                float zqc = zlzk * al_akl;
                float xkl = xk + xqc;
                float ykl = yk + yqc;
                float zkl = zk + zqc;
                float theta = theta_ij * akl / (theta_ij + akl);
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                float theta_rr = logf(rr + 1.f) + theta * rr;
                float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                keep &= dm_penalty > d_cutoff;
                if (keep) {
                    d_cutoff = max(kl_cutoff - q_kl, d_cutoff);
                    keep = d_ij > d_cutoff || dm_cond[bas_kl] > d_cutoff;
                }
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_vjk_tasks_nosym(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                           int pair_ij, int ish, int jsh,
                           float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                           RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float d_ij = dm_cond[ish * nbas + jsh];
    float kl_cutoff = cutoff - q_ij;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                float d_cutoff = kl_cutoff - q_kl;
                keep = (d_ij                  > d_cutoff ||
                        dm_cond[bas_kl]       > d_cutoff ||
                        dm_cond[ish*nbas+ksh] > d_cutoff ||
                        dm_cond[jsh*nbas+ksh] > d_cutoff ||
                        dm_cond[ish*nbas+lsh] > d_cutoff ||
                        dm_cond[jsh*nbas+lsh] > d_cutoff);
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_vjk_tasks_nosym(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                              int pair_ij, int ish, int jsh,
                              float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                              float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                              RysIntEnvVars &envs, BoundsInfo &bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    double *env = envs.env;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    float ai = expi[iprim-1];
    float aj = expj[jprim-1];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float xpa = xjxi * aj_aij;
    float ypa = yjyi * aj_aij;
    float zpa = zjzi * aj_aij;
    float xij = xi + xpa;
    float yij = yi + ypa;
    float zij = zi + zpa;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float s_ij = s_cond_ij[pair_ij];
    float d_ij = dm_cond[ish * nbas + jsh];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl + dm_penalty >= kl_cutoff;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                float ak = expk[kprim-1];
                float al = expl[lprim-1];
                float akl = ak + al;
                float al_akl = al / akl;
                float xk = rk[0];
                float yk = rk[1];
                float zk = rk[2];
                float xl = rl[0];
                float yl = rl[1];
                float zl = rl[2];
                float xlxk = xl - xk;
                float ylyk = yl - yk;
                float zlzk = zl - zk;
                float xqc = xlxk * al_akl;
                float yqc = ylyk * al_akl;
                float zqc = zlzk * al_akl;
                float xkl = xk + xqc;
                float ykl = yk + yqc;
                float zkl = zk + zqc;
                float theta = theta_ij * akl / (theta_ij + akl);
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                float theta_rr = logf(rr + 1.f) + theta * rr;
                float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                keep &= dm_penalty > d_cutoff;
                if (keep) {
                    d_cutoff = max(kl_cutoff - q_kl, d_cutoff);
                    keep = (d_ij > d_cutoff ||
                            dm_cond[bas_kl] > d_cutoff ||
                            dm_cond[ish*nbas+ksh] > d_cutoff ||
                            dm_cond[jsh*nbas+ksh] > d_cutoff ||
                            dm_cond[ish*nbas+lsh] > d_cutoff ||
                            dm_cond[jsh*nbas+lsh] > d_cutoff);
                }
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif
    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__
static void _fill_ejk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                            int pair_ij, int ish, int jsh,
                            float *q_cond_ij, float *q_cond_kl,
                            JKEnergy &jk, RysIntEnvVars envs, BoundsInfo bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    int do_j = jk.j_factor != 0;
    int do_k = jk.k_factor != 0;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
      if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                float d_cutoff = kl_cutoff - q_kl;
                keep = ((do_k && (dm_cond[ish*nbas+lsh]+dm_cond[jsh*nbas+ksh] > d_cutoff ||
                                  dm_cond[ish*nbas+ksh]+dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                        (do_j && d_ij+dm_cond[bas_kl] > d_cutoff));
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}

__device__
static void _fill_sr_ejk_tasks(int& ntasks, int& pair_kl0, uint32_t *bas_kl_idx,
                               int pair_ij, int ish, int jsh,
                               float *q_cond_ij, float *q_cond_kl,
                               float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                               JKEnergy &jk, RysIntEnvVars envs, BoundsInfo bounds, double *shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    #endif

    int t_id = threadIdx_y * blockDim_x + threadIdx_x;
    int threads = blockDim_x * blockDim_y;
    __syncthreads();
    if (t_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *dm_cond = bounds.dm_cond;
    double *env = envs.env;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float xpa = xjxi * aj_aij;
    float ypa = yjyi * aj_aij;
    float zpa = zjzi * aj_aij;
    float xij = xi + xpa;
    float yij = yi + ypa;
    float zij = zi + zpa;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond_ij[pair_ij];
    float s_ij = s_cond_ij[pair_ij];
    uint32_t bas_ij = ish * nbas + jsh;
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = jk.omega;
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);
    int do_j = jk.j_factor != 0;
    int do_k = jk.k_factor != 0;

    int *swap = (int *)shared_memory;

    #ifdef USE_SYCL
    for (int active_y = 0; active_y < blockDim_y; ++active_y) {
        if (threadIdx_y == active_y) {
    #endif
    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + t_id;
        __syncthreads();
        uint32_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            if (q_kl + Q_COND_MARGIN < kl_cutoff) {
                pair_kl0 = bounds.npairs_kl;
            }
            keep = q_kl >= kl_cutoff && bas_ij >= bas_kl;
            if (keep) {
                uint32_t ksh = bas_kl / nbas;
                uint32_t lsh = bas_kl - nbas * ksh;
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                float ak = diffuse_exps[ksh];
                float al = diffuse_exps[lsh];
                float akl = ak + al;
                float al_akl = al / akl;
                float xk = rk[0];
                float yk = rk[1];
                float zk = rk[2];
                float xl = rl[0];
                float yl = rl[1];
                float zl = rl[2];
                float xlxk = xl - xk;
                float ylyk = yl - yk;
                float zlzk = zl - zk;
                float xqc = xlxk * al_akl;
                float yqc = ylyk * al_akl;
                float zqc = zlzk * al_akl;
                float xkl = xk + xqc;
                float ykl = yk + yqc;
                float zkl = zk + zqc;
                float theta = theta_ij * akl / (theta_ij + akl);
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                float theta_rr = logf(rr + 1.f) + theta * rr;
                float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                keep &= d_cutoff < 0;
                if (keep) {
                    d_cutoff = max(kl_cutoff - q_kl, d_cutoff);
                    float dm_jk = dm_cond[jsh*nbas+ksh];
                    float dm_jl = dm_cond[jsh*nbas+lsh];
                    float dm_ik = dm_cond[ish*nbas+ksh];
                    float dm_il = dm_cond[ish*nbas+lsh];
                    keep = ((do_k && (dm_il+dm_jk > d_cutoff || dm_ik+dm_jl > d_cutoff)) ||
                            (do_j && d_ij+dm_cond[bas_kl] > d_cutoff));
                }
            }
        }
        int offset = mask_to_index(keep, swap, threads, t_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (t_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    #ifdef USE_SYCL
        } // for: if threadIdx_y
        __syncthreads();
    } // for: active_y
    #endif

    if (threadIdx_y == 0 && ntasks + t_id < QUEUE_DEPTH && ntasks > 0) {
        bas_kl_idx[ntasks+t_id] = bas_kl_idx[ntasks-1];
    }
    __syncthreads();
}
