/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

template <int ANG> __forceinline__ __device__
void gto_cartesian(double values[], const double fx, const double fy, const double fz)
{
    if constexpr (ANG == 0) {
        values[0] = 1;
    } else if constexpr (ANG == 1) {
        values[0] = fx;
        values[1] = fy;
        values[2] = fz;
    } else if constexpr (ANG == 2) {
        values[0] = fx * fx;
        values[1] = fx * fy;
        values[2] = fx * fz;
        values[3] = fy * fy;
        values[4] = fy * fz;
        values[5] = fz * fz;
    } else if constexpr (ANG == 3) {
        double xx = fx * fx;
        double yy = fy * fy;
        double zz = fz * fz;
        values[0] = xx * fx;
        values[1] = xx * fy;
        values[2] = xx * fz;
        values[3] = fx * yy;
        values[4] = fx * fy * fz;
        values[5] = fx * zz;
        values[6] = yy * fy;
        values[7] = yy * fz;
        values[8] = fy * zz;
        values[9] = fz * zz;
    } else if constexpr (ANG == 4) {
        double xx = fx * fx;
        double yy = fy * fy;
        double zz = fz * fz;
        double xxx = xx * fx;
        double yyy = yy * fy;
        double zzz = zz * fz;
        values[0 ] = xxx * fx;
        values[1 ] = xxx * fy;
        values[2 ] = xxx * fz;
        values[3 ] = xx * yy;
        values[4 ] = xx * fy * fz;
        values[5 ] = xx * zz;
        values[6 ] = fx * yyy;
        values[7 ] = fx * yy * fz;
        values[8 ] = fx * fy * zz;
        values[9 ] = fx * zzz;
        values[10] = yyy * fy;
        values[11] = yyy * fz;
        values[12] = yy * zz;
        values[13] = fy * zzz;
        values[14] = fz * zzz;
    }
}
