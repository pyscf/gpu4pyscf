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

#pragma once

#include "gsycl/gpu_compat.h"

template <typename T>
__host__ __device__ T distance_squared(const T x, const T y, const T z) {
  return x * x + y * y + z * z;
}

#ifdef USE_SYCL
inline
#endif
__device__ __forceinline__
void multiply(double aR, double aI, double bR, double bI, double &cR, double &cI)
{
    double outR = aR * bR - aI * bI;
    double outI = aR * bI + aI * bR;
    cR = outR;
    cI = outI;
}

#ifdef USE_SYCL
inline
#endif
__device__ __forceinline__
double reduce(double val, double *swap, int thread_id)
{
#ifdef USE_SYCL
    setup_context();
#endif
    constexpr int WARP_SIZE = 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = thread_id % WARP_SIZE;
    int warp = thread_id / WARP_SIZE;
#ifdef USE_SYCL
    // The callers invoke reduce() repeatedly (once per matrix element of a
    // shell-pair block) with the SAME `swap` scratch. Without this barrier
    // the warps of iteration n+1 can overwrite swap[] while warp 0 is still
    // reading it for iteration n. On NVIDIA the warp scheduler happens to
    // hide this; on Intel GPUs it produces run-to-run nondeterministic
    // results (observed: multigrid nonorth matrices varying by up to 1e-4
    // between identical calls, only for the LI>0/LJ>0 kernels, i.e. exactly
    // those that call reduce() more than once).
    __syncthreads();
#endif
    if (lane == 0) {
        swap[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        val = (thread_id < 8) ? swap[lane] : 0.;
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    // Without this barrier the other warps can enter the next reduce and
    // overwrite swap[] before warp 0 has read it.
    __syncthreads();
    return val;
}
