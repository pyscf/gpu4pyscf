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

// WARP_SIZE: compile-time constant used for shared-memory sizing.
// `warpSize` (HIP/CUDA device-runtime built-in) is not constexpr,
// so we keep a literal here. Guarded so the build can override
// it (e.g. -DWARP_SIZE=64) for future wider-wavefront targets.
#ifndef WARP_SIZE
#define WARP_SIZE       32
#endif
// corresponding to 256 threads
#define WARPS           8
// Block size is fixed at 256 threads and must NOT scale with WARP_SIZE. The
// int3c2e / int1e kernels here are thread-indexed (sp_id = tid % nsp_per_block)
// and reduce through shared memory, and their host/device shared-memory budget
// assumes 256 threads. Deriving THREADS from WARP_SIZE would make a wave64
// block 512 threads, doubling shared memory past the 64KB LDS limit (invalid
// device image) and desyncing host/device sizing. 256 threads is 8 warps on
// wave32 or 4 warps on wave64; the warp count is irrelevant to correctness here.
#define THREADS         256
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
