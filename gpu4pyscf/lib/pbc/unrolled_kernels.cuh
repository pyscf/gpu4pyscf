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
    RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds, \
    int64_t *pair_ij_mapping, int64_t *pair_kl_mapping, \
    int *supcell_shl, int *Ts_ij_lookup, \
    int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao, \
    float *q_cond_ij, float *q_cond_kl, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    float dm_penalty, int64_t *pool, int *head, void *shm_mem

#define JKMATRIX_KERNEL_SETUP() \
    setup_context(); \
    int sq_id = threadIdx_x; \
    int gout_id = threadIdx_y; \
    int _nsq_per_block = blockDim_x; \
    int64_t *bas_kl_idx = pool + blockIdx_x * QUEUE_DEPTH; \
    SHARED_SCALAR(int, ntasks); \
    SHARED_SCALAR(int, pair_ij); \
    SHARED_SCALAR(int, pair_kl0); \
    SHARED_SCALAR(int, cell_j); \
    SHARED_SCALAR(int, ish_cell0); \
    SHARED_SCALAR(int, jsh_cell0); \
    SHARED_SCALAR(int, i0); \
    SHARED_SCALAR(int, j0); \
    SHARED_ARRAY(double, ri, [3]); \
    SHARED_ARRAY(double, rjri, [3]); \
    SHARED_SCALAR(int, expi); \
    SHARED_SCALAR(int, expj); \
    DYNAMIC_SHARED_PTR(double, shared_memory, shm_mem);

#define LAUNCH_JKMATRIX_KERNEL(KERNEL) { \
    auto _rys_envs = *envs; auto _rys_kmat = *kmat; auto _rys_bounds = *bounds; \
    auto _rys_blocks = make_grid(workers, 1); \
    auto _rys_threads = make_block(nsq_per_block, gout_stride); \
    LAUNCH_KERNEL_DYN( KERNEL, _rys_blocks, _rys_threads, \
        (buflen)*sizeof(double), \
        _rys_envs, _rys_kmat, _rys_bounds, \
        pair_ij_mapping, pair_kl_mapping, supcell_shl, Ts_ij_lookup, \
        nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl, \
        s_cond_ij, s_cond_kl, diffuse_exps, dm_penalty, pool, head); \
  }
