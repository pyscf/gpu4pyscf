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

extern SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_bpcache;

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
    GINT_CACHE_REF(); \
    const int task_ij = blockIdx_x * blockDim_x + threadIdx_x; \
    const int task_kl = blockIdx_y * blockDim_y + threadIdx_y;

// Cache reference for gpu_compat-style (rank-3) kernels that spell out index
// setup inline via setup_context(). CUDA uses the __constant__ symbol directly;
// SYCL binds the device_global. Keeps .cu files free of backend branches.
#ifdef USE_SYCL
#define GINT_CACHE_REF() \
    const auto& c_bpcache = s_bpcache.get()
#else
#define GINT_CACHE_REF()
#endif

// Backend-agnostic constant-cache symbol for CONSTANT_MEMCPY upload sites.
// Usage: CONSTANT_MEMCPY(GINT_BPCACHE_SYM, bpcache, sizeof(BasisProdCache));
#ifdef USE_SYCL
#define GINT_BPCACHE_SYM s_bpcache
#define GINT_BPCACHE_DEFINE() \
    SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_bpcache
#else
#define GINT_BPCACHE_SYM c_bpcache
#define GINT_BPCACHE_DEFINE() \
    __constant__ BasisProdCache c_bpcache
#endif

// Backend-agnostic upload of host BasisProdCache to constant memory on a
// caller-provided stream. Usage: GINT_BPCACHE_UPLOAD(stream, bpcache);
#ifdef USE_SYCL
#define GINT_BPCACHE_UPLOAD(stream, src) \
    (stream).memcpy(s_bpcache, src, sizeof(BasisProdCache)).wait()
#else
#define GINT_BPCACHE_UPLOAD(stream, src) \
    do { checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, src, sizeof(BasisProdCache))); (void)(stream); } while (0)
#endif

// Context + cache setup for intra-block direct-write helpers. Use threadIdx_x /
// blockDim_x macros directly at use sites (1D block span THREADSX*THREADSY).
#define KERNEL_SETUP_LOCAL() \
    setup_context(); \
    GINT_CACHE_REF();
