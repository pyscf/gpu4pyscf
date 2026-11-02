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

// The kernel prologue and launch below are written against the gpu_compat.h
// macros, which expand to the native CUDA constructs on the CUDA backend.
// The 19 rys_k_* kernels defined in both unrolled_rys_jk.cu and
// unrolled_rys_k.cu are renamed per translation unit by -D options in
// gvhf-rys/CMakeLists.txt (SYCL only), so these generated files stay
// byte-identical to upstream.

#define JKMATRIX_KERNEL_ARGS \
    RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, int *head

#define JKMATRIX_KERNEL_SETUP() \
    setup_context(); \
    int sq_id = threadIdx_x; \
    int gout_id = threadIdx_y; \
    int _nsq_per_block = blockDim_x; \
    uint32_t *bas_kl_idx = pool + blockIdx_x * QUEUE_DEPTH; \
    SHARED_ARRAY(int, ntasks); \
    SHARED_ARRAY(int, pair_ij); \
    SHARED_ARRAY(int, pair_kl0); \
    SHARED_ARRAY(int, ish); \
    SHARED_ARRAY(int, jsh); \
    SHARED_ARRAY(double, ri, [3]); \
    SHARED_ARRAY(double, rjri, [3]); \
    SHARED_ARRAY(int, expi); \
    SHARED_ARRAY(int, expj); \
    DYNAMIC_SHARED_PTR(double, shared_memory);

#define LAUNCH_JKMATRIX_KERNEL(KERNEL) { \
    auto _rys_envs = *envs; auto _rys_jk = *jk; auto _rys_bounds = *bounds; \
    auto _rys_blocks = make_grid(workers, 1); \
    auto _rys_threads = make_block(nsq_per_block, gout_stride); \
    LAUNCH_KERNEL( KERNEL, _rys_blocks, _rys_threads, \
        (buflen)*sizeof(double), \
        _rys_envs, _rys_jk, _rys_bounds, q_cond_ij, q_cond_kl, \
        dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); \
  }

#define JKENERGY_KERNEL_ARGS \
    RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, double *dd_pool, int *head

#define JKENERGY_KERNEL_SETUP() \
    setup_context(); \
    int sq_id = threadIdx_x; \
    int gout_id = threadIdx_y; \
    int worker_id = blockIdx_x; \
    SHARED_ARRAY(int, ntasks); \
    SHARED_ARRAY(int, pair_ij); \
    SHARED_ARRAY(int, pair_kl0); \
    SHARED_ARRAY(int, ish); \
    SHARED_ARRAY(int, jsh); \
    SHARED_ARRAY(double, ri, [3]); \
    SHARED_ARRAY(double, rjri, [3]); \
    SHARED_ARRAY(int, expi); \
    SHARED_ARRAY(int, expj); \
    DYNAMIC_SHARED_PTR(double, shared_memory);

#define LAUNCH_JKENERGY_KERNEL(KERNEL) { \
    auto _rys_envs = *envs; auto _rys_jk = *jk; auto _rys_bounds = *bounds; \
    auto _rys_blocks = make_grid(workers, 1); \
    auto _rys_threads = make_block(nsq_per_block, gout_stride); \
    LAUNCH_KERNEL( KERNEL, _rys_blocks, _rys_threads, \
        (buflen)*sizeof(double), \
        _rys_envs, _rys_jk, _rys_bounds, q_cond_ij, q_cond_kl, \
        dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, \
        dd_pool, head); \
  }
