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

#define WARP_SIZE       32
#define FT_AO_THREADS   (WARP_SIZE*4)
#define NG_PER_BLOCK    32

typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t nfij;
    uint8_t g_size;
    uint8_t stride_i;
    uint8_t stride_j;
    uint8_t iprim;
    uint8_t jprim;
    int npairs_ij;
    int ngrids;
    int *bas_ij_idx;
    double *grids;
    int *img_offsets; // offset AFTIntEnvVars.img_idx for each shell-pair
    int *img_idx; // indices of img_coords in each shell-pair
} AFTBoundsInfo;

typedef struct {
    int ngrids;
    // The effective basis pair Id = ish*nbas+jsh
    int *bas_ij_idx;
    // the bas_ij_idx offset for each blockIdx.x
    int *shl_pair_offsets;
    // the AO-pair offset (address) in the output tensor for each blockIdx.x
    int *ao_pair_loc;
    // offset AFTIntEnvVars.img_idx for each shell-pair
    int *img_offsets;
    // indices of img_coords in each shell-pair
    int *img_idx;
    // gout_stride for for each (li,lj) pattern
    int *gout_stride_lookup;
    double *grids;
} BDivAFTBoundsInfo;
