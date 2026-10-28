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

#include "gsycl/gpu_compat.h"
#include "gvhf-rys/vhf.cuh"   // c_nf / c_div_nf on both backends
#include <cuComplex.h>        // gsycl/cuComplex.h shim under SYCL

extern TABLE_DEFINE(double, s_c_lattice_vectors, c_lattice_vectors, 9);
extern TABLE_DEFINE(double, s_c_reciprocal_lattice_vectors, c_reciprocal_lattice_vectors, 9); // norm to 1
extern TABLE_DEFINE(double, s_c_dxyz_dabc, c_dxyz_dabc, 9);

#define NBAS_MAX        16777216
