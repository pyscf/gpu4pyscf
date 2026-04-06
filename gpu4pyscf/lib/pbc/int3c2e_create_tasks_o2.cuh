/*
 * Copyright 2024-2026 The PySCF Developers. All Rights Reserved.
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
#include <cub/cub.cuh>

#define THREADS         256
#define WARP_SIZE       32
#define WARPS           8
#define LMAX            4
#define LMAX1           (LMAX+1)
#define MAX_IMGS_PER_TASK  30
#define POOL_SIZE       16384

typedef struct {
    int ksh;
    int pair_ij;
    int img_count;
    int remaining_imgs;
    uint32_t img_j_offset;
    int nimgs_j;
    float theta_rr_threshold;
    float theta;
    float theta_ij;
    float aj_ij;
    float xjxi;
    float yjyi;
    float zjzi;
    float xixk;
    float yiyk;
    float zizk;
} ShellTripletTaskInfo;

__device__ inline
void initialize_ijk_tasks(uint32_t *img_pool, uint32_t *rem_task_idx,
                          ShellTripletTaskInfo *ijk_tasks_info,
                          PBCIntEnvVars &envs, int shl_pair0, int shl_pair1,
                          int ksh0_cell0, int ksh1_cell0, int li, int lj, int nauxbas,
                          uint32_t *bas_ij_idx, int *img_idx, uint32_t *sp_img_offsets,
                          float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int thread_id = threadIdx.x;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    int nimgs = envs.nimgs;
    int nksh = ksh1_cell0 - ksh0_cell0;
    int bvk_nksh = nksh * ncells;
    int nshl_pairs = shl_pair1 - shl_pair0;
    for (int ijk_id = thread_id; ijk_id < bvk_nksh * nshl_pairs; ijk_id += THREADS) {
        int pair_ij = ijk_id / bvk_nksh;
        int kidx = ijk_id - pair_ij * bvk_nksh;
        pair_ij += shl_pair0;

        uint32_t img0 = sp_img_offsets[pair_ij];
        int nimgs_j = sp_img_offsets[pair_ij+1] - img0;
        int nimgs2 = nimgs_j * nimgs;

        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int cell_id = kidx / nksh;
        int ksh_cell0 = ksh0_cell0 + kidx - nksh * cell_id;
        int ksh = cell_id * nauxbas + ksh_cell0 + bvk_nbas;
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;

        float ai = diffuse_exps[ish];
        float aj = diffuse_exps[jsh];
        float ak = diffuse_exps[ksh];
        float aij = ai + aj;
        float aj_aij = aj / aij;
        float ai_aij = ai / aij;
        float theta_ij = ai * aj_aij;
        float aij_ak = aij * ak;
        float theta = aij_ak * omega2 / (aij_ak + (aij + ak) * omega2);
        float ci = diffuse_coefs[ish];
        float cj = diffuse_coefs[jsh];
        float omega_aij = omega2 / (omega2 + aij);
        // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
        //           ~ between [0, 2]
        float fac_guess = .5f - logf(omega2)/4;
        // log(ci*cj * (pi/aij)**1.5)
        float log_fac = logf(fabsf(ci*cj)) + 1.717f - 1.5f*logf(aij) + fac_guess;
        // An addiitonal factor for Coulomb integrals
        // log_fac += .25 * logf(2./pi * aij)
        log_fac += .25f * logf(0.6366f * aij);
        float log_cutoff_w_fac = log_cutoff - log_fac;

        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        float xi = env[ri+0];
        float yi = env[ri+1];
        float zi = env[ri+2];
        float xj = env[rj+0];
        float yj = env[rj+1];
        float zj = env[rj+2];
        float xk = env[rk+0];
        float yk = env[rk+1];
        float zk = env[rk+2];
        float xjxi = xj - xi;
        float yjyi = yj - yi;
        float zjzi = zj - zi;
        float xixk = xi - xk;
        float yiyk = yi - yk;
        float zizk = zi - zk;

        // The estimated rr for dri_fac and drj_fac below satisfies the condition
        // (dri_fac + drj_fac - theta*rr > log_cutoff). In this case rr_ij = 0,
        // corresponding to the maximum overlap between the two orbital basis.
        // The resulting dri_fac and drj_fac can be used as an estimation of the upper limits.
        // These factors are then combined with log_cutoff to provide an overall threshold.
        // float rt_aij = omega_aij * sqrtf(rr);
        // float dr = sqrtf(rr_ij);
        // float dri = aj_aij * dr + rt_aij;
        // float drj = ai_aij * dr + rt_aij;
        // float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
        // float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
        // theta_rr_threshold ~ dri_fac + drj_fac - log_cutoff_w_fac
        float penalty = logf(5e-1f);
        float rr_estimate = fabsf(log_cutoff_w_fac + penalty) / theta;
        float rt_aij = omega_aij * sqrtf(rr_estimate);
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float dr = sqrtf(rr_ij);
        float dri = dr/2 + rt_aij;
        float u = .25f / aij;
        float log_rt_aij = max(0.f, logf(dri*dri + (li+lj)*u));
        float theta_rr_threshold = .5f*(li+lj)*log_rt_aij - log_cutoff_w_fac;

        ShellTripletTaskInfo cur_task = {
            ksh, pair_ij, 0, nimgs2, img0, nimgs_j,
            theta_rr_threshold, theta, theta_ij, aj_aij,
            xjxi, yjyi, zjzi, xixk, yiyk, zizk};
        ijk_tasks_info[ijk_id] = cur_task;
        rem_task_idx[ijk_id] = ijk_id;
    }
    __syncthreads();
}

__device__ inline
void _filter_ijk_tasks(uint32_t *rem_task_idx, int& num_ijk_tasks,
                       ShellTripletTaskInfo *ijk_tasks_info)
{
    int thread_id = threadIdx.x;
    int tot_tasks = num_ijk_tasks;
    __syncthreads();
    if (thread_id == 0) {
        num_ijk_tasks = 0;
    }
    using BlockScan = cub::BlockScan<int, THREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    for (int base = 0; base < tot_tasks; base += THREADS) {
        int task_id = base + thread_id;
        register int ijk_id = 0;
        int keep = 0;
        if (task_id < tot_tasks) {
            ijk_id = rem_task_idx[task_id];
            keep = ijk_tasks_info[ijk_id].remaining_imgs > 0;
        }

        int prefix, block_total;
        BlockScan(temp_storage).ExclusiveSum(keep, prefix, block_total);
        __syncthreads();  // required before reusing temp_storage

        if (keep) {
            rem_task_idx[num_ijk_tasks + prefix] = ijk_id;
        }
        __syncthreads();
        if (thread_id == 0) {
            num_ijk_tasks += block_total;
        }
    }
    __syncthreads();
}

__device__ inline
void _select_sub_tasks(uint32_t *sub_task_idx, int &num_sub_tasks,
                       int& img_count_lower, uint32_t *rem_task_idx,
                       int num_ijk_tasks, int nst_per_block,
                       ShellTripletTaskInfo *ijk_tasks_info)
{
    int thread_id = threadIdx.x;
    __syncthreads();
    if (thread_id == 0) {
        num_sub_tasks = 0;
    }
    int img_count_upper = img_count_lower * 2;
    if (num_ijk_tasks < 2 * nst_per_block) {
        img_count_upper *= 2;
    }

    using BlockScan = cub::BlockScan<int, THREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    for (int base = 0; base < num_ijk_tasks; base += THREADS) {
        int task_id = base + thread_id;
        register int ijk_id = 0;
        int keep = 0;
        if (task_id < num_ijk_tasks) {
            ijk_id = rem_task_idx[task_id];
            int img_count = ijk_tasks_info[ijk_id].img_count;
            keep = img_count_lower <= img_count && img_count < img_count_upper;
        }

        int prefix, block_total;
        BlockScan(temp_storage).ExclusiveSum(keep, prefix, block_total);
        __syncthreads();  // required before reusing temp_storage

        if (keep) {
            sub_task_idx[num_sub_tasks + prefix] = ijk_id;
        }
        __syncthreads();
        if (thread_id == 0) {
            num_sub_tasks += block_total;
        }
    }
    __syncthreads();
    if (thread_id == 0) {
        img_count_lower = img_count_upper;
    }
    __syncthreads();
}

__device__ inline
void _filter_jk_images(uint32_t *img_pool, uint32_t *rem_task_idx,
                       int num_ijk_tasks, ShellTripletTaskInfo *ijk_tasks_info,
                       PBCIntEnvVars &envs, int *sp_img_idx)
{
    int thread_id = threadIdx.x;
    __shared__ int task_head;
    if (thread_id == 0) {
        task_head = THREADS;
    }
    __syncthreads();

    double *img_coords = envs.img_coords;
    uint32_t nimgs = envs.nimgs;
    register int remaining_imgs = 0;
    register int task_id = -1;
    register int ijk_id = 0;
    register int img_count = MAX_IMGS_PER_TASK;
    float theta, theta_ij, aj_aij, theta_rr_threshold;
    float xjxi, yjyi, zjzi;
    float xixk, yiyk, zizk;
    uint32_t img0;
    while (1) {
        if (img_count == MAX_IMGS_PER_TASK || remaining_imgs <= 0) {
            if (task_id < 0) { // the first iteration, nothing needs to be updated
                task_id = thread_id;
            } else {
                ijk_tasks_info[ijk_id].img_count = img_count;
                ijk_tasks_info[ijk_id].remaining_imgs = remaining_imgs;
                // next available ijk_task
                task_id = atomicAdd(&task_head, 1);
            }
            if (task_id >= num_ijk_tasks) break;

            img_count = 0;
            ijk_id = rem_task_idx[task_id];
            ShellTripletTaskInfo cur_task = ijk_tasks_info[ijk_id];
            remaining_imgs = cur_task.remaining_imgs;
            if (remaining_imgs <= 0) continue;
            img0 = cur_task.img_j_offset;
            theta = cur_task.theta;
            theta_ij = cur_task.theta_ij;
            aj_aij = cur_task.aj_ij;
            theta_rr_threshold = cur_task.theta_rr_threshold;
            xjxi = cur_task.xjxi;
            yjyi = cur_task.yjyi;
            zjzi = cur_task.zjzi;
            xixk = cur_task.xixk;
            yiyk = cur_task.yiyk;
            zizk = cur_task.zizk;
        }

        remaining_imgs--;
        int jL = remaining_imgs / nimgs;
        int kL = remaining_imgs - nimgs * jL;
        jL = sp_img_idx[img0 + (uint32_t)jL];
        float xixkL = xixk - img_coords[kL*3+0];
        float yiykL = yiyk - img_coords[kL*3+1];
        float zizkL = zizk - img_coords[kL*3+2];
        float xjLxi = xjxi + img_coords[jL*3+0];
        float yjLyi = yjyi + img_coords[jL*3+1];
        float zjLzi = zjzi + img_coords[jL*3+2];
        float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        float theta_ij_rr = theta_ij * rr_ij;
        float xijk = xjLxi * aj_aij + xixkL;
        float yijk = yjLyi * aj_aij + yiykL;
        float zijk = zjLzi * aj_aij + zizkL;
        float rr = xijk * xijk + yijk * yijk + zijk * zijk;
        float theta_rr = theta * rr + theta_ij_rr;
        //if (theta_rr < REMOTE_THRESHOLD) {
        //    // theta*rr ~ log_cutoff would be the largest rr we need to consider
        //    // dri_fac and drj_fac at this rr would be the upper limit of the
        //    // contribution from these factors.
        //    // The approx value of dri_fac and drj_fac are estimated and
        //    // cached in the ijk_tasks_info
        //    float rt_aij = omega_aij * sqrtf(rr);
        //    float dr = sqrtf(rr_ij);
        //    float dri = aj_aij * dr + rt_aij;
        //    float drj = ai_aij * dr + rt_aij;
        //    float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
        //    float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
        //    float estimator = dri_fac + drj_fac - theta_rr;
        //    if (estimator > log_cutoff) {
        //        img_pool[img_count*POOL_SIZE+task_id] = jL * nimgs + kL;
        //        img_count++;
        //    }
        //}
        if (theta_rr < theta_rr_threshold) {
            img_pool[ijk_id+POOL_SIZE*img_count] = jL * nimgs + kL;
            img_count++;
        }
    }
    __syncthreads();
}

__device__ inline
int warp_max(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline
void block_max(int val, int& out)
{
    int thread_id = threadIdx.x;
    val = warp_max(val);
    __shared__ int buf[WARPS];
    int lane = thread_id % warpSize;
    int warp_id = thread_id / warpSize;
    if (lane == 0) {
        buf[warp_id] = val;
    }
    __syncthreads();
    if (thread_id < WARPS) {
        val = buf[thread_id];
    }
    for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xff, val, offset));
    }
    if (thread_id == 0) {
        out = val;
    }
    __syncthreads();
}

