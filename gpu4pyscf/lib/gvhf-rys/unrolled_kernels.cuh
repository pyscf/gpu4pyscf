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

#ifdef USE_SYCL

// ---------------------------------------------------------------------
// Per-translation-unit kernel-name disambiguation.
//
// unrolled_rys_jk.cu and unrolled_rys_k.cu BOTH define 19 kernels named
// rys_k_0000 .. rys_k_3200 with DIFFERENT bodies. 
//
// RYS_UNROLLED_KERNEL_TAG is injected per source file by
// gvhf-rys/CMakeLists.txt. Do NOT define it inside the .cu files --
// they are auto-generated upstream and must stay byte-identical.
// ---------------------------------------------------------------------
#ifndef RYS_UNROLLED_KERNEL_TAG
#error "RYS_UNROLLED_KERNEL_TAG is not defined. Every unrolled_*.cu that includes unrolled_kernels.cuh must get a unique tag via set_source_files_properties(... COMPILE_DEFINITIONS RYS_UNROLLED_KERNEL_TAG=<tag>) in gvhf-rys/CMakeLists.txt. Without it, identically-named kernels in different translation units silently alias."
#endif

#define RYS_KERNEL_TAG_CAT_(KERNEL, TAG) KERNEL##_##TAG##_sycl
#define RYS_KERNEL_TAG_CAT(KERNEL, TAG)  RYS_KERNEL_TAG_CAT_(KERNEL, TAG)
#define RYS_KERNEL_TAG(KERNEL)           RYS_KERNEL_TAG_CAT(KERNEL, RYS_UNROLLED_KERNEL_TAG)

#define dim3 sycl::range<2>

#define JKMATRIX_KERNEL_ARGS \
    RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, int *head, void *shm_mem

#define JKMATRIX_KERNEL_SETUP() \
    setup_context(); \
    int sq_id = threadIdx_x; \
    int gout_id = threadIdx_y; \
    int _nsq_per_block = blockDim_x; \
    uint32_t *bas_kl_idx = pool + blockIdx_x * QUEUE_DEPTH; \
    SHARED_SCALAR(int, ntasks); \
    SHARED_SCALAR(int, pair_ij); \
    SHARED_SCALAR(int, pair_kl0); \
    SHARED_SCALAR(int, ish); \
    SHARED_SCALAR(int, jsh); \
    SHARED_ARRAY(double, ri, [3]); \
    SHARED_ARRAY(double, rjri, [3]); \
    SHARED_SCALAR(int, expi); \
    SHARED_SCALAR(int, expj); \
    DYNAMIC_SHARED_PTR(double, shared_memory, shm_mem);

#define LAUNCH_JKMATRIX_KERNEL(KERNEL) { \
    auto _rys_envs = *envs; auto _rys_jk = *jk; auto _rys_bounds = *bounds; \
    auto _rys_blocks = make_grid(workers, 1); \
    auto _rys_threads = make_block(nsq_per_block, gout_stride); \
    LAUNCH_KERNEL_DYN_Q(sycl_get_queue(), KERNEL, _rys_blocks, _rys_threads, \
        (buflen)*sizeof(double), \
        _rys_envs, _rys_jk, _rys_bounds, q_cond_ij, q_cond_kl, \
        dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); \
  }

#define JKENERGY_KERNEL_ARGS \
    RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds, \
    float *q_cond_ij, float *q_cond_kl, float dm_penalty, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    uint32_t *pool, double *dd_pool, int *head, void *shm_mem

#define JKENERGY_KERNEL_SETUP() \
    setup_context(); \
    int sq_id = threadIdx_x; \
    int gout_id = threadIdx_y; \
    int worker_id = blockIdx_x; \
    SHARED_SCALAR(int, ntasks); \
    SHARED_SCALAR(int, pair_ij); \
    SHARED_SCALAR(int, pair_kl0); \
    SHARED_SCALAR(int, ish); \
    SHARED_SCALAR(int, jsh); \
    SHARED_ARRAY(double, ri, [3]); \
    SHARED_ARRAY(double, rjri, [3]); \
    SHARED_SCALAR(int, expi); \
    SHARED_SCALAR(int, expj); \
    DYNAMIC_SHARED_PTR(double, shared_memory, shm_mem);

#define LAUNCH_JKENERGY_KERNEL(KERNEL) { \
    auto _rys_envs = *envs; auto _rys_jk = *jk; auto _rys_bounds = *bounds; \
    auto _rys_blocks = make_grid(workers, 1); \
    auto _rys_threads = make_block(nsq_per_block, gout_stride); \
    LAUNCH_KERNEL_DYN_Q(sycl_get_queue(), KERNEL, _rys_blocks, _rys_threads, \
        (buflen)*sizeof(double), \
        _rys_envs, _rys_jk, _rys_bounds, q_cond_ij, q_cond_kl, \
        dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, \
        dd_pool, head); \
  }

#else  // !USE_SYCL  -- byte-identical to upstream/master

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

#endif // USE_SYCL
