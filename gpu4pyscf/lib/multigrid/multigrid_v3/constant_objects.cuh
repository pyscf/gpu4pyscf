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

#ifdef USE_SYCL
#include <sycl_device.hpp>

extern SYCL_EXTERNAL sycl_device_global<double[9]> s_c_lattice_vectors;
extern SYCL_EXTERNAL sycl_device_global<double[9]> s_c_reciprocal_lattice_vectors; // norm to 1
extern SYCL_EXTERNAL sycl_device_global<double[9]> s_c_dxyz_dabc;

#define c_lattice_vectors            (s_c_lattice_vectors.get())
#define c_reciprocal_lattice_vectors (s_c_reciprocal_lattice_vectors.get())
#define c_dxyz_dabc                  (s_c_dxyz_dabc.get())

// c_nf / c_div_nf are defined unconditionally in gvhf-rys/vhf.cuh, which
// every multigrid_v3 TU includes. Under USE_SYCL __constant__ expands to
// `inline constexpr`, so defining them here too is an ODR redefinition
// error. Inherit vhf.cuh's tables (identical values) instead.

// CUDA float2 stand-in. sycl::float2's element accessors are methods
// (v.x()), not members (v.x), so aliasing to it would break every
// `.x`/`.y` site in screen.cu and the eval_*_v2/strain_grad kernels. This
// POD keeps both branches identical and supports brace-init assignment.
struct alignas(8) float2 { float x, y; };
#else
extern __constant__ double c_lattice_vectors[9];
extern __constant__ double c_reciprocal_lattice_vectors[9]; // norm to 1
extern __constant__ double c_dxyz_dabc[9];
extern __constant__ int c_nf[];
extern __constant__ float c_div_nf[];
#endif

#define NBAS_MAX        16777216
