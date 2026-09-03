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

#define dim3 sycl::range<2>

#define JKMATRIX_KERNEL_ARGS                                             \
RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,           \
  int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,           \
  int *supcell_shl, int *Ts_ij_lookup,                          \
  int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,      \
  float *q_cond_ij, float *q_cond_kl,                           \
  float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,      \
  float dm_penalty, int64_t *pool, int *head                    \
  , sycl::nd_item<2> &item, double *shared_memory

#define JKMATRIX_KERNEL_SETUP()                                                  \
  int sq_id = item.get_local_id(1);                                     \
  int gout_id = item.get_local_id(0);                                   \
  int _nsq_per_block = item.get_local_range(1);                         \
  int blockIdx_x = item.get_group(1);                                   \
  int64_t *bas_kl_idx = pool + blockIdx_x * QUEUE_DEPTH;                 \
  auto thread_block = item.get_group();                                 \
  int &ntasks   = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &pair_ij  = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &cell_j   = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &ish_cell0= *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &jsh_cell0= *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &i0       = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &j0       = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  double (&ri)[3]   = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block); \
  double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block); \
  int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block); \
  int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

#define LAUNCH_JKMATRIX_KERNEL(KERNEL)                                           \
    {                                                                   \
      auto dev_envs = *envs; auto dev_kmat = *kmat; auto dev_bounds = *bounds; \
      sycl::range<2> blocks(1, workers);                                \
      sycl::range<2> cuda_threads(gout_stride, nsq_per_block);          \
      sycl_get_queue()->submit([&](sycl::handler &cgh) {                \
        sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); \
        cgh.parallel_for<class KERNEL##_pbc_k_sycl>(sycl::nd_range<2>(blocks * cuda_threads, cuda_threads), [=](auto item) { \
          KERNEL(dev_envs, dev_kmat, dev_bounds,                        \
                 pair_ij_mapping, pair_kl_mapping, supcell_shl, Ts_ij_lookup, \
                 nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl, \
                 s_cond_ij, s_cond_kl, diffuse_exps, dm_penalty, pool, head, \
                 item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));   \
        });                                                             \
      });                                                               \
    }

#else // USE_SYCL

#define JKMATRIX_KERNEL_ARGS \
    RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds, \
    int64_t *pair_ij_mapping, int64_t *pair_kl_mapping, \
    int *supcell_shl, int *Ts_ij_lookup, \
    int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao, \
    float *q_cond_ij, float *q_cond_kl, \
    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps, \
    float dm_penalty, int64_t *pool, int *head

#define JKMATRIX_KERNEL_SETUP() \
    int sq_id = threadIdx.x; \
    int gout_id = threadIdx.y; \
    int _nsq_per_block = blockDim.x; \
    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH; \
    extern __shared__ double shared_memory[]; \
    __shared__ int ntasks, pair_ij, pair_kl0; \
    __shared__ int cell_j, ish_cell0, jsh_cell0, i0, j0; \
    __shared__ double ri[3]; \
    __shared__ double rjri[3]; \
    __shared__ int expi; \
    __shared__ int expj;

#define LAUNCH_JKMATRIX_KERNEL(KERNEL) \
    KERNEL<<<workers, threads, buflen*sizeof(double)>>>( \
    *envs, *kmat, *bounds, \
    pair_ij_mapping, pair_kl_mapping, supcell_shl, Ts_ij_lookup, \
    nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl, \
    s_cond_ij, s_cond_kl, diffuse_exps, dm_penalty, pool, head)

#endif // USE_SYCL
