/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

#include <stdint.h>

#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#define warpSize 32 // (builtin SYCL not available) Needed for the files: ./lib/pbc/fill_int3c2e_v2.cu, ./lib/pbc/unrolled_int3c2e.cu
#else
#include <cuda_runtime.h>
#endif

#define WARP_SIZE       32
// corresponding to 256 threads
#define WARPS           8
#define THREADS         (WARP_SIZE*WARPS)
#define IMG_MASK_SLOTS  1024
#define L_AUX_MAX       6
#define L_AUX1          7
#define SPTASKS_PER_BLOCK       32
#define IMG_BLOCK       16384
#define PI_FAC          34.98683665524972497

typedef struct {
    int li;
    int lj;
    int lk;
    int nroots;
    int nfi;
    int nfj;
    int nfk;
    int kprim;
    int stride_j;
    int stride_k;
    int g_size;
    int nbas_aux;
    int nksh;
    int ksh0;
    int naux;
    int n_prim_pairs;
    int n_ctr_pairs;
    uint32_t *bas_ij_idx;
    int *pair_mapping;
    uint32_t *img_offsets; // offset img_idx for each shell-pair
    int *img_idx; // indices of img_coords in each shell-pair
} PBCInt3c2eBounds;

typedef struct {
    int *bas_ij_idx;
    // the bas_ij_idx offset for each blockIdx.x
    int *shl_pair_offsets;
    // gout_stride for for each (li,lj) pattern
    int *gout_stride_lookup;
} PBCInt2c2eBounds;
