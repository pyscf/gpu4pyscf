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
    constexpr int WARP_SIZE = 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = thread_id % WARP_SIZE;
    int warp = thread_id / WARP_SIZE;
    if (lane == 0) {
        swap[warp] = val;
    }
    __syncthreads();

    val = (thread_id < 8) ? swap[lane] : 0.;
    for (int offset = 4; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
