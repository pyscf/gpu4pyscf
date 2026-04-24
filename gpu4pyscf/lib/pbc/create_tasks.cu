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
#include "gvhf-rys/vhf.cuh"

#define THREADS         256

__device__ static
void _fill_sr_vk_tasks(int &ntasks, int &pair_kl0, uint32_t *bas_kl_idx, uint32_t bas_ij,
                       int *bas_mask_idx, int *Ts_ij_lookup, int nimgs, int nbas_cell0,
                       RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    if (thread_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int _jsh = bas_mask_idx[jsh];
    int ish_cell0 = ish;
    int jsh_cell0 = _jsh % nbas_cell0;
    int cell_j = _jsh / nbas_cell0;
    int bas_ij_cell0 = ish_cell0 * nbas_cell0 + jsh_cell0;
    int nbas2 = nbas_cell0 * nbas_cell0;
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

    extern __shared__ double shared_memory[];
    int *swap = (int *)shared_memory;

    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + thread_id;
        int bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond[bas_kl];
            keep = q_kl >= kl_cutoff;
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int _ksh = bas_mask_idx[ksh];
            int _lsh = bas_mask_idx[lsh];
            int ksh_cell0 = _ksh % nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            keep &= bas_ij_cell0 >= ksh_cell0*nbas_cell0+lsh_cell0;

            if (keep) {
                int cell_k = _ksh / nbas_cell0;
                int cell_l = _lsh / nbas_cell0;
                float d_cutoff = kl_cutoff - q_kl;
                int _jk = Ts_ij_lookup[cell_j+cell_k*nimgs] * nbas2;
                int _jl = Ts_ij_lookup[cell_j+cell_l*nimgs] * nbas2;
                int _ik = Ts_ij_lookup[cell_k             ] * nbas2;
                int _il = Ts_ij_lookup[cell_l             ] * nbas2;
                float dm_jk = dm_cond[_jk + jsh_cell0*nbas_cell0+ksh_cell0];
                float dm_jl = dm_cond[_jl + jsh_cell0*nbas_cell0+lsh_cell0];
                float dm_ik = dm_cond[_ik + ish_cell0*nbas_cell0+ksh_cell0];
                float dm_il = dm_cond[_il + ish_cell0*nbas_cell0+lsh_cell0];
                keep &= (dm_jk > d_cutoff || dm_jl > d_cutoff ||
                         dm_ik > d_cutoff || dm_il > d_cutoff);
                if (keep) {
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
                    keep &= (dm_jk > d_cutoff || dm_jl > d_cutoff ||
                             dm_ik > d_cutoff || dm_il > d_cutoff);
                }
            }
        }

        swap[thread_id] = keep;
        __syncthreads();
        for (int offset = 1; offset < threads; offset <<= 1) {
            int val = 0;
            if (thread_id >= offset) {
                val = swap[thread_id - offset];
            }
            __syncthreads();
            swap[thread_id] += val;
            __syncthreads();
        }
        if (keep) {
            int offset = swap[thread_id] - keep;
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (thread_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        bas_kl_idx[ntasks+thread_id] = pair_kl_mapping[0];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_ejk_tasks(int &ntasks, int &pair_kl0, uint32_t *bas_kl_idx, uint32_t bas_ij,
                        int *bas_mask_idx, int *Ts_ij_lookup, int nimgs, int nbas_cell0,
                        JKEnergy &jk, RysIntEnvVars &envs, BoundsInfo &bounds)
{
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    if (thread_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    uint32_t nbas = envs.nbas;
    uint32_t *pair_kl_mapping = bounds.pair_kl_mapping;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int _jsh = bas_mask_idx[jsh];
    int ish_cell0 = ish;
    int jsh_cell0 = _jsh % nbas_cell0;
    int cell_j = _jsh / nbas_cell0;
    int bas_ij_cell0 = ish_cell0 * nbas_cell0 + jsh_cell0;
    int nbas2 = nbas_cell0 * nbas_cell0;
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
    float dm_ji = dm_cond[Ts_ij_lookup[cell_j]*nbas2 + jsh_cell0*nbas_cell0+ish_cell0];
    dm_ji += 1.5f;
    float s_ij = s_estimator[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    float skl_cutoff = cutoff - s_ij;
    float omega = jk.omega;
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);
    int do_j = jk.j_factor != 0;
    int do_k = jk.k_factor != 0;

    extern __shared__ double shared_memory[];
    int *swap = (int *)shared_memory;

    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + thread_id;
        int bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond[bas_kl];
            keep = q_kl >= kl_cutoff;
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int _ksh = bas_mask_idx[ksh];
            int _lsh = bas_mask_idx[lsh];
            int ksh_cell0 = _ksh % nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            keep &= bas_ij_cell0 >= ksh_cell0*nbas_cell0+lsh_cell0;

            if (keep) {
                int cell_k = _ksh / nbas_cell0;
                int cell_l = _lsh / nbas_cell0;
                //float d_cutoff = kl_cutoff - q_kl;
                float dm_jk = dm_cond[Ts_ij_lookup[cell_j+cell_k*nimgs]*nbas2 + jsh_cell0*nbas_cell0+ksh_cell0];
                float dm_jl = dm_cond[Ts_ij_lookup[cell_j+cell_l*nimgs]*nbas2 + jsh_cell0*nbas_cell0+lsh_cell0];
                float dm_ik = dm_cond[Ts_ij_lookup[cell_k             ]*nbas2 + ish_cell0*nbas_cell0+ksh_cell0];
                float dm_il = dm_cond[Ts_ij_lookup[cell_l             ]*nbas2 + ish_cell0*nbas_cell0+lsh_cell0];
                float dm_lk = dm_cond[Ts_ij_lookup[cell_l+cell_k*nimgs]*nbas2 + lsh_cell0*nbas_cell0+ksh_cell0];
                float dm_jk_il = dm_jk + dm_il;
                float dm_ik_jl = dm_ik + dm_jl;
                float dm_ij_kl = dm_ji + dm_lk;
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
                keep &= ((do_k && (dm_jk_il > d_cutoff || dm_ik_jl > d_cutoff)) ||
                         (do_j && dm_ij_kl > d_cutoff));
            }
        }

        swap[thread_id] = keep;
        __syncthreads();
        for (int offset = 1; offset < threads; offset <<= 1) {
            int val = 0;
            if (thread_id >= offset) {
                val = swap[thread_id - offset];
            }
            __syncthreads();
            swap[thread_id] += val;
            __syncthreads();
        }
        if (keep) {
            int offset = swap[thread_id] - keep;
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (thread_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
        bas_kl_idx[ntasks+thread_id] = pair_kl_mapping[0];
    }
    __syncthreads();
}
