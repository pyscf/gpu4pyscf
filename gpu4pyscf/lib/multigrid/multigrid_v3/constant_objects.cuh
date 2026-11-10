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

// gpu_compat.h (via this header chain) provides CUDA_VERSION (aft_eval_*); the runtime API shims live in gsycl/sycl_device.hpp.
#include <cuComplex.h>
#include "gpu_compat.h"

// All constant memory of multigrid_v3 is declared here.
#ifdef USE_SYCL
// c_nf / c_div_nf: constexpr tables defined in vhf.cuh (CUDA declares them below).
#include "gvhf-rys/vhf.cuh"
// __constant__ is constexpr under SYCL, which the host cannot update, so the
// host-written tables are device_globals. Being inline they need no
// definition in any one .cu; cudaMemcpyToSymbol and c_xxx[i] work on them
// exactly as on CUDA.
inline sycl_device_global<double[9]> c_lattice_vectors;
inline sycl_device_global<double[9]> c_reciprocal_lattice_vectors; // norm to 1
inline sycl_device_global<double[9]> c_dxyz_dabc;
#else
// Defined in drivers.cu
extern __constant__ double c_lattice_vectors[9];
extern __constant__ double c_reciprocal_lattice_vectors[9]; // norm to 1
extern __constant__ double c_dxyz_dabc[9];
extern __constant__ int c_nf[];
extern __constant__ float c_div_nf[];
#endif

#define NBAS_MAX        16777216
