/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
#include <cuda_runtime.h>

#include "vhf.cuh"

__device__ static
void _fill_vk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                    RysIntEnvVars &envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        uint32_t bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if ((dm_cond[ish*nbas+ksh] > d_cutoff ||
             dm_cond[jsh*nbas+ksh] > d_cutoff ||
             dm_cond[ish*nbas+lsh] > d_cutoff ||
             dm_cond[jsh*nbas+lsh] > d_cutoff)) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // pad data to avoid overflow
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_vjk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                     RysIntEnvVars &envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // pad data to avoid overflow
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_vj_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                    RysIntEnvVars &envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    float *q_cond = bounds.q_cond;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // pad data to avoid overflow
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_sr_vk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                       RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *s_estimator = bounds.s_estimator;
    float *dm_cond = bounds.dm_cond;
    float *diffuse_exps = s_estimator + nbas*nbas;
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
    float q_ij = q_cond[bas_ij];
    float s_ij = s_estimator[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        float dm_jk = dm_cond[jsh*nbas+ksh];
        float dm_jl = dm_cond[jsh*nbas+lsh];
        float dm_ik = dm_cond[ish*nbas+ksh];
        float dm_il = dm_cond[ish*nbas+lsh];
        if ((dm_ik > d_cutoff || dm_jk > d_cutoff ||
             dm_il > d_cutoff || dm_jl > d_cutoff)) {
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
            float d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
            if ((dm_ik > d_cutoff || dm_jk > d_cutoff ||
                 dm_il > d_cutoff || dm_jl > d_cutoff)) {
                int off = atomicAdd(ntasks, 1);
                bas_kl_idx[off] = bas_kl;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_sr_vjk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                        RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *s_estimator = bounds.s_estimator;
    float *dm_cond = bounds.dm_cond;
    float *diffuse_exps = s_estimator + nbas*nbas;
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
    float q_ij = q_cond[bas_ij];
    float s_ij = s_estimator[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
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
            float d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
            if (d_ij                  > d_cutoff ||
                dm_cond[bas_kl]       > d_cutoff ||
                dm_cond[ish*nbas+ksh] > d_cutoff ||
                dm_cond[jsh*nbas+ksh] > d_cutoff ||
                dm_cond[ish*nbas+lsh] > d_cutoff ||
                dm_cond[jsh*nbas+lsh] > d_cutoff) {
                int off = atomicAdd(ntasks, 1);
                bas_kl_idx[off] = bas_kl;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_sr_vj_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                       RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *s_estimator = bounds.s_estimator;
    float *dm_cond = bounds.dm_cond;
    float *diffuse_exps = s_estimator + nbas*nbas;
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
    float q_ij = q_cond[bas_ij];
    float s_ij = s_estimator[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff) {
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
            float d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
            if (d_cutoff > 0) {
                continue;
            }
            if (d_ij                  > d_cutoff ||
                dm_cond[bas_kl]       > d_cutoff) {
                int off = atomicAdd(ntasks, 1);
                bas_kl_idx[off] = bas_kl;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_vjk_tasks_nosym(int *ntasks, int *bas_kl_idx, int bas_ij,
                           RysIntEnvVars &envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // pad data to avoid overflow
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__ static
void _fill_sr_vjk_tasks_nosym(int *ntasks, int *bas_kl_idx, int bas_ij,
                              RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *s_estimator = bounds.s_estimator;
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
    float q_ij = q_cond[bas_ij];
    float s_ij = s_estimator[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
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
            float d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
            if (d_cutoff > 0) {
                continue;
            }
            if (d_ij                  > d_cutoff ||
                dm_cond[bas_kl]       > d_cutoff ||
                dm_cond[ish*nbas+ksh] > d_cutoff ||
                dm_cond[jsh*nbas+ksh] > d_cutoff ||
                dm_cond[ish*nbas+lsh] > d_cutoff ||
                dm_cond[jsh*nbas+lsh] > d_cutoff) {
                int off = atomicAdd(ntasks, 1);
                bas_kl_idx[off] = bas_kl;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__
static void _fill_ejk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                            JKEnergy &jk, RysIntEnvVars envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    int do_j = jk.j_factor != 0;
    int do_k = jk.k_factor != 0;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if ((do_k && (dm_cond[ish*nbas+lsh]+dm_cond[jsh*nbas+ksh] > d_cutoff ||
                      dm_cond[ish*nbas+ksh]+dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
            (do_j && d_ij+dm_cond[bas_kl] > d_cutoff)) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // pad data to avoid overflow
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}

__device__
static void _fill_sr_ejk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij,
                               JKEnergy &jk, RysIntEnvVars envs, BoundsInfo bounds)
{
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float *q_cond = bounds.q_cond;
    float *s_estimator = bounds.s_estimator;
    float *dm_cond = bounds.dm_cond;
    float *diffuse_exps = s_estimator + nbas*nbas;
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
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float s_ij = s_estimator[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = jk.omega;
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);
    int do_j = jk.j_factor != 0;
    int do_k = jk.k_factor != 0;

    for (int pair_kl = t_id; pair_kl < bounds.npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) {
            continue;
        }
        if (bas_ij < bas_kl) {
            continue;
        }
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        float dm_jk = dm_cond[jsh*nbas+ksh];
        float dm_jl = dm_cond[jsh*nbas+lsh];
        float dm_ik = dm_cond[ish*nbas+ksh];
        float dm_il = dm_cond[ish*nbas+lsh];
        if ((do_k && (dm_il+dm_jk > d_cutoff || dm_ik+dm_jl > d_cutoff)) ||
            (do_j && d_ij+dm_cond[bas_kl] > d_cutoff)) {
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
            float d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
            if ((do_k && (dm_il+dm_jk > d_cutoff || dm_ik+dm_jl > d_cutoff)) ||
                (do_j && d_ij+dm_cond[bas_kl] > d_cutoff)) {
                int off = atomicAdd(ntasks, 1);
                bas_kl_idx[off] = bas_kl;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks+t_id] = pair_kl_mapping[0];
    }
}
