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

#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]

namespace gpu4pyscf::gpbc::multi_grid {

template <typename T>
__host__ __device__ T distance_squared(const T x, const T y, const T z) {
  return x * x + y * y + z * z;
}

template <typename T, int ANG> __device__ constexpr T common_fac_sp() {
  if constexpr (ANG == 0) {
    return 0.282094791773878143;
  } else if constexpr (ANG == 1) {
    return 0.488602511902919921;
  } else {
    return 1.0;
  }
}

template <typename T, int ANG> __device__ T log_common_fac_sp() {
  if constexpr (ANG == 0) {
    return -1.26551212348464540;
  } else if constexpr (ANG == 1) {
    return -0.71620597915059055;
  } else {
    return 0;
  }
}

} // namespace gpu4pyscf::gpbc::multi_grid
