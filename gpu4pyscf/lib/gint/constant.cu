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

#include "gint.h"
#include "cint2e.cuh"

// SYCL store behind the c_bpcache accessor (cint2e.cuh). CUDA's
// __constant__ c_bpcache lives in constant_tables.cu (built only under !USE_SYCL).
#ifdef USE_SYCL
SYCL_EXTERNAL sycl_device_global<BasisProdCache> s_gint_bpcache;
#endif
