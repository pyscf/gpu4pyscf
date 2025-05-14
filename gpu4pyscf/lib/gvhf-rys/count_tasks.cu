/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else
#include <cuda_runtime.h>
#endif

#include "vhf.cuh"
#include "create_tasks.cu"

__global__
static void count_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                            ShellQuartet *pool, uint32_t *batch_head
                            #ifdef USE_SYCL
                            , sycl::nd_item<2> &item, char *shm_mem
                            #endif
                            )
{
#ifdef USE_SYCL
    int b_id = item.get_group(1);
    int t_id = item.get_local_id(0) * item.get_local_range(1) + item.get_local_id(1);
    int& batch_id = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(item.get_group());
#else
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ int batch_id;
#endif
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    if (t_id == 0) {
      batch_id = atomicAdd(batch_head, (uint32_t)1);
    }
    __syncthreads();
    double omega = envs.env[PTR_RANGE_OMEGA];
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        uint32_t ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl, shm_mem);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl, shm_mem);
        }
        if (t_id == 0) {
          batch_id = atomicAdd(batch_head, (uint32_t)1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

extern "C" {
int RYS_count_jk_tasks(double *vj, double *vk, double *dm, int n_dm, int nao,
                 RysIntEnvVars envs, int *scheme, int *shls_slice,
                 int ntile_ij_pairs, int ntile_kl_pairs,
                 int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                 float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                 ShellQuartet *pool, uint32_t *batch_head, int workers,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = order / 2 + 1;
    uint8_t stride_j = li + 1;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};
    JKMatrix jk = {vj, vk, dm, (uint16_t)n_dm};

    int threads = scheme[0]*scheme[1];
    int buflen = threads;

    #ifdef USE_SYCL
    sycl::queue& stream = *sycl_get_queue();
    stream.memset(batch_head, 0, 2*sizeof(uint32_t)).wait();
    sycl::range<2> blocks(1, workers);
    sycl::range<2> thread(1, threads);
    stream.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(buflen*sizeof(double), cgh);
      cgh.parallel_for(sycl::nd_range<2>(blocks * thread, thread), [=](auto item) {
        count_jk_kernel(envs, jk, bounds, pool, batch_head,
                        item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));
    count_jk_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
    #endif
    return 0;
}
}
