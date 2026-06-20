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
namespace gpu4pyscf::gpbc::multi_grid {

#ifdef USE_SYCL
#include <sycl_device.hpp>

extern SYCL_EXTERNAL sycl_device_global<double[9]> s_lattice_vectors;
extern SYCL_EXTERNAL sycl_device_global<double[9]> s_reciprocal_lattice_vectors;
extern SYCL_EXTERNAL sycl_device_global<double[9]> s_dxyz_dabc;
extern SYCL_EXTERNAL sycl_device_global<double[3]> s_reciprocal_norm;

// Bare references in kernels (e.g. dxyz_dabc[i]) resolve to the raw
// pointer obtained from the device_global, matching the v1 multigrid
// convention (auto x = s_xxx.get(); x[i]).
#define lattice_vectors            (s_lattice_vectors.get())
#define reciprocal_lattice_vectors (s_reciprocal_lattice_vectors.get())
#define dxyz_dabc                  (s_dxyz_dabc.get())
#define reciprocal_norm            (s_reciprocal_norm.get())
#else
extern __constant__ double lattice_vectors[9];
extern __constant__ double reciprocal_lattice_vectors[9];
extern __constant__ double dxyz_dabc[9];
extern __constant__ double reciprocal_norm[3];
#endif

} // namespace gpu4pyscf::gpbc::multi_grid
