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

#pragma once

#include "gint.h"

#ifdef USE_SYCL

extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_bpcache;

// Generated with GINTinit_index1d_xyz
inline constexpr int c_idx[TOT_NF*3] = {
  0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 4, 3, 3,
  2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1,
  1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2,
  2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2,
  1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4,
  3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2,
  1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0,
  6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2,
  0, 1, 2, 3, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 0, 1, 0,
  1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 0, 1, 0, 1, 2,
  0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6};

inline constexpr int c_l_locs[GPU_LMAX+2] = {0, 1, 4, 10, 20, 35, 56, 84};

#else // USE_SYCL
//extern __constant__ GINTEnvVars c_envs;
extern __constant__ BasisProdCache c_bpcache;
//extern __constant__ int16_t c_idx4c[NFffff*3];

extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];
#endif // USE_SYCL

// Abstracts 2D kernel thread-index setup for task_ij/task_kl kernels. Used 79x across gint/.
#ifdef USE_SYCL
#define KERNEL_SETUP() \
    auto item = syclex::this_work_item::get_nd_item<2>(); \
    const int task_ij = item.get_global_id(1); \
    const int task_kl = item.get_global_id(0); \
    const auto& c_bpcache = s_bpcache.get();
#else
#define KERNEL_SETUP() \
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x; \
    const int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
#endif

// Abstracts 2D kernel local thread-index setup for threadIdx_x/blockDim_x kernels. Used 9x across gint/.
#ifdef USE_SYCL
#define KERNEL_SETUP_LOCAL() \
    auto item = syclex::this_work_item::get_nd_item<2>(); \
    const int threadIdx_x = item.get_local_id(1); \
    const int blockDim_x = item.get_local_range(1); \
    const auto& c_bpcache = s_bpcache.get();
#else
#define KERNEL_SETUP_LOCAL() \
    const int threadIdx_x = threadIdx.x; \
    const int blockDim_x = blockDim.x;
#endif
