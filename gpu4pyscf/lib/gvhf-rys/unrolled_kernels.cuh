/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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

#include "gvhf-rys/vhf.cuh"

#define JKMATRIX_KERNEL_ARGS \
    RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, int *head

#define JKMATRIX_KERNEL_SETUP() \
    int sq_id = threadIdx.x; \
    int gout_id = threadIdx.y; \
    int _nsq_per_block = blockDim.x; \
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH; \
    extern __shared__ double shared_memory[]; \
    __shared__ int ntasks, pair_ij, pair_kl0; \
    __shared__ int ish, jsh; \
    __shared__ double ri[3]; \
    __shared__ double rjri[3]; \
    __shared__ int expi; \
    __shared__ int expj;

#define LAUNCH_JKMATRIX_KERNEL(KERNEL) \
    KERNEL<<<workers, threads, buflen*sizeof(double)>>>( \
    *envs, *jk, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head)

#define JKENERGY_KERNEL_ARGS \
    RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, double *dd_pool, int *head

#define JKENERGY_KERNEL_SETUP() \
    int sq_id = threadIdx.x; \
    int gout_id = threadIdx.y; \
    int worker_id = blockIdx.x; \
    extern __shared__ double shared_memory[]; \
    __shared__ int ntasks, pair_ij, pair_kl0; \
    __shared__ int ish, jsh; \
    __shared__ double ri[3]; \
    __shared__ double rjri[3]; \
    __shared__ int expi, expj;

#define LAUNCH_JKENERGY_KERNEL(KERNEL) \
    KERNEL<<<workers, threads, buflen*sizeof(double)>>>( \
    *envs, *jk, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, dd_pool, head)
