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

#include "../../warp_size.h"

constexpr int MGRID_SUBGROUP_SIZE = WARP_SIZE;
static_assert(MGRID_SUBGROUP_SIZE == 32 || MGRID_SUBGROUP_SIZE == 64);

template <typename T>
__host__ __device__ T distance_squared(const T x, const T y, const T z) {
  return x * x + y * y + z * z;
}

__device__ __forceinline__
void multiply(double aR, double aI, double bR, double bI, double &cR, double &cI)
{
    double outR = aR * bR - aI * bI;
    double outI = aR * bI + aI * bR;
    cR = outR;
    cI = outI;
}

__device__ __forceinline__
double reduce(double val, double *swap, int thread_id)
{
    for (int offset = MGRID_SUBGROUP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(__activemask(), val, offset, MGRID_SUBGROUP_SIZE);
    }
    int lane = thread_id % MGRID_SUBGROUP_SIZE;
    int warp = thread_id / MGRID_SUBGROUP_SIZE;
    if (lane == 0) {
        swap[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        int nwarps = blockDim.x * blockDim.y * blockDim.z / MGRID_SUBGROUP_SIZE;
        val = (lane < nwarps) ? swap[lane] : 0.;
        for (int offset = MGRID_SUBGROUP_SIZE/2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(__activemask(), val, offset, MGRID_SUBGROUP_SIZE);
        }
    }
    // Without this barrier the other warps can enter the next reduce and
    // overwrite swap[] before warp 0 has read it.
    __syncthreads();
    return val;
}
