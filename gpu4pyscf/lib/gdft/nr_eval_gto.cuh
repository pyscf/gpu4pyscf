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

#pragma once

// Kernel-side & launch-config macros to unify CUDA and SYCL in gdft.
// All variants in ONE ifdef block - single pair of #ifdef / #else.
#ifdef USE_SYCL

#define SHARED_ARRAY(T, name, SIZE)                                   \
    using name##_tile_t = T[SIZE];                                    \
    name##_tile_t& name = *sycl::ext::oneapi::                        \
        group_local_memory_for_overwrite<name##_tile_t>(item.get_group());

#define MAKE_RANGE_2D(X, Y)  sycl::range<2>((Y), (X))
#define MAKE_RANGE_3D(X, Y, Z)  sycl::range<3>((Z), (Y), (X))

#define BLOCKS_SET_Y(val)  (blocks[0] = (val))
#define BLOCKS_GET_Y()     (blocks[0])

#else

#define SHARED_ARRAY(T, name, SIZE)  __shared__ T name[SIZE];

#define MAKE_RANGE_2D(X, Y)  dim3((X), (Y))
#define MAKE_RANGE_3D(X, Y, Z)  dim3((X), (Y), (Z))

#define BLOCKS_SET_Y(val)  (blocks.y = (val))
#define BLOCKS_GET_Y()     (blocks.y)

#endif // USE_SYCL

typedef struct {
    int natm;
    int nbas;
    int *bas_atom;
    int *bas_exp;
    int *bas_coeff;
    double *env;
    double *atom_coordx;
} GTOValEnvVars;

typedef struct {
    int ngrids;
    int nbas;
    int nao;
    int bas_off;
    int nprim;
    int *ao_loc;
    int *bas_indices;
    double fac;
    double *gridx;
    double *data;
} BasOffsets;

#define C_ATOM          0
#define C_EXP           1
#define C_COEFF         2
#define C_BAS_SLOTS     3
#define NBAS_MAX        6000

