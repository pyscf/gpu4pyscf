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
#include <string.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

#define THREADS         256
#define NBAS_MAX        1048576

// np.where(threads_mask)[0]
__device__ inline
int mask_to_index(int keep, int *tmp_storage, int threads, int t_id)
{
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
void _fill_sr_vk_tasks(int &ntasks, int &pair_kl0, int64_t *bas_kl_idx,
                       int pair_ij, int ish, int jsh,
                       int64_t *pair_kl_mapping, int *bas_mask_idx,
                       int *Ts_ij_lookup, int nimgs, int nbas_cell0,
                       float *q_cond_ij, float *q_cond_kl,
                       float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                       JKMatrix& kmat, RysIntEnvVars& envs, BoundsInfo& bounds)
{
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    __syncthreads();
    if (thread_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    int _jsh = bas_mask_idx[jsh];
    int ish_cell0 = ish;
    int jsh_cell0 = _jsh % nbas_cell0;
    int cell_j = _jsh / nbas_cell0;
    int bas_ij_cell0 = ish_cell0 * nbas_cell0 + jsh_cell0;
    uint32_t nbas2 = nbas_cell0 * nbas_cell0;
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
    float omega = kmat.omega;
    float omega2 = omega * omega;
    float theta_ij = omega2 * aij / (aij + omega2);

    extern __shared__ double shared_memory[];
    int *swap = (int *)shared_memory;

    while (pair_kl0 < bounds.npairs_kl && ntasks < QUEUE_DEPTH - 512) {
        int pair_kl = pair_kl0 + thread_id;
        int64_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            keep = q_kl >= kl_cutoff;
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
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
                keep = (dm_jk > d_cutoff || dm_jl > d_cutoff ||
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
                    float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                    keep = (dm_jk > d_cutoff || dm_jl > d_cutoff ||
                            dm_ik > d_cutoff || dm_il > d_cutoff);
                }
            }
        }

        int offset = mask_to_index(keep, swap, threads, thread_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (thread_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && ntasks + thread_id < QUEUE_DEPTH) {
        bas_kl_idx[ntasks+thread_id] = pair_kl_mapping[0];
    }
    __syncthreads();
}

__device__ static
void _fill_sr_ejk_tasks(int &ntasks, int &pair_kl0, int64_t *bas_kl_idx,
                        int pair_ij, int ish, int jsh,
                        int64_t *pair_kl_mapping, int *bas_mask_idx,
                        int *Ts_ij_lookup, int nimgs, int nbas_cell0,
                        float *q_cond_ij, float *q_cond_kl,
                        float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                        JKEnergy& jk, RysIntEnvVars& envs, BoundsInfo& bounds)
{
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    __syncthreads();
    if (thread_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int *bas = envs.bas;
    int _jsh = bas_mask_idx[jsh];
    int ish_cell0 = ish;
    int jsh_cell0 = _jsh % nbas_cell0;
    int cell_j = _jsh / nbas_cell0;
    int bas_ij_cell0 = ish_cell0 * nbas_cell0 + jsh_cell0;
    uint32_t nbas2 = nbas_cell0 * nbas_cell0;
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
    float dm_ji = dm_cond[Ts_ij_lookup[cell_j]*nbas2 + jsh_cell0*nbas_cell0+ish_cell0];
    dm_ji += 1.5f;
    float s_ij = s_cond_ij[pair_ij];
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
        int64_t bas_kl = 0;
        int keep = 0;
        if (pair_kl < bounds.npairs_kl) {
            bas_kl = pair_kl_mapping[pair_kl];
            float q_kl = q_cond_kl[pair_kl];
            keep = q_kl >= kl_cutoff;
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int _lsh = bas_mask_idx[lsh];
            int ksh_cell0 = _ksh % nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            keep &= bas_ij_cell0 >= ksh_cell0*nbas_cell0+lsh_cell0;

            if (keep) {
                int cell_k = _ksh / nbas_cell0;
                int cell_l = _lsh / nbas_cell0;
                float d_cutoff = kl_cutoff - q_kl;
                float dm_jk = dm_cond[Ts_ij_lookup[cell_j+cell_k*nimgs]*nbas2 + jsh_cell0*nbas_cell0+ksh_cell0];
                float dm_jl = dm_cond[Ts_ij_lookup[cell_j+cell_l*nimgs]*nbas2 + jsh_cell0*nbas_cell0+lsh_cell0];
                float dm_ik = dm_cond[Ts_ij_lookup[cell_k             ]*nbas2 + ish_cell0*nbas_cell0+ksh_cell0];
                float dm_il = dm_cond[Ts_ij_lookup[cell_l             ]*nbas2 + ish_cell0*nbas_cell0+lsh_cell0];
                float dm_lk = dm_cond[Ts_ij_lookup[cell_l+cell_k*nimgs]*nbas2 + lsh_cell0*nbas_cell0+ksh_cell0];
                float dm_jk_il = dm_jk + dm_il;
                float dm_ik_jl = dm_ik + dm_jl;
                float dm_ij_kl = dm_ji + dm_lk;
                keep = ((do_k && (dm_jk_il > d_cutoff || dm_ik_jl > d_cutoff)) ||
                        (do_j && dm_ij_kl > d_cutoff));
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
                    float d_cutoff = skl_cutoff - s_cond_kl[pair_kl] + theta_rr;
                    keep = ((do_k && (dm_jk_il > d_cutoff || dm_ik_jl > d_cutoff)) ||
                            (do_j && dm_ij_kl > d_cutoff));
                }
            }
        }

        int offset = mask_to_index(keep, swap, threads, thread_id);
        if (keep) {
            bas_kl_idx[ntasks + offset] = bas_kl;
        }
        __syncthreads();
        if (thread_id == 0) {
            ntasks += swap[threads - 1];
            pair_kl0 += threads;
        }
        __syncthreads();
    }
    if (threadIdx.y == 0 && ntasks + thread_id < QUEUE_DEPTH) {
        bas_kl_idx[ntasks+thread_id] = pair_kl_mapping[0];
    }
    __syncthreads();
}
