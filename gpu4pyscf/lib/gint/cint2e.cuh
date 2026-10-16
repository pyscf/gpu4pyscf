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
// sycl_device.hpp comes via gint.h.
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
// Backend-agnostic kernel macros (setup_context, threadIdx_x, DYNAMIC_SHARED_PTR,
// LAUNCH_KERNEL_*). Included here so every gint TU gets them regardless of include order.
#include "gsycl/gpu_compat.h"

#ifdef USE_SYCL

extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_gint_bpcache;
// Kernel bodies keep CUDA's `c_bpcache.member` spelling (device code only;
// host uploads go through TABLE_FILL, which names s_gint_bpcache directly).
#define c_bpcache (::s_gint_bpcache.get())

// Generated with GINTinit_index1d_xyz
// Look into constant.cu for details
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

// Provides task_ij/task_kl grid indices for 3D-launched kernels. Backend-agnostic:
// setup_context() + index math via threadIdx_x/blockDim_x macros (gpu_compat.h).
// Launch with make_block(THREADSX, THREADSY) + make_grid(...) so x/y lanes match.
#define KERNEL_SETUP() \
    setup_context(); \
    const int task_ij = blockIdx_x * blockDim_x + threadIdx_x; \
    const int task_kl = blockIdx_y * blockDim_y + threadIdx_y;

