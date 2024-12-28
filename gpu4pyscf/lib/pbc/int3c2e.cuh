/*
 * Copyright 2024 The PySCF Developers. All Rights Reserved.
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

#include <stdint.h>

#define WARP_SIZE       32
// corresponding to 256 threads
#define WARPS           8
#define IMG_MASK_SLOTS  1024
#define L_AUX_MAX       6
#define SPTAKS_PER_BLOCK        32

#ifndef HAVE_DEFINED_PBCINT3CENVVAS_H
#define HAVE_DEFINED_PBCINT3CENVVAS_H
typedef struct {
    uint16_t natm; // in bvk-cell
    uint16_t nbas; // in bvk-cell
    uint16_t bvk_ncells; // number of images in the BvK cell
    uint16_t nimgs; // number of images in lattice sum
    int *atm;
    int *bas;
    double *env;
    int *ao_loc; // in bvk-cell
    double *img_coords; // vectors in lattice sum
    float log_cutoff;
} PBCInt3c2eEnvVars;

typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t lk;
    uint8_t nroots;
    uint8_t nfi;
    uint8_t nfij;
    uint8_t nfk;
    uint8_t iprim;
    uint8_t jprim;
    uint8_t kprim;
    uint8_t stride_i;
    uint8_t stride_j;
    uint8_t stride_k;
    uint8_t g_size;
    uint16_t nrow;
    uint16_t ncol;
    uint16_t naux;
    uint16_t nksh;
    uint16_t ish0;
    uint16_t jsh0;
    uint16_t ksh0;
    int npairs_ij;
    int *bas_ij_idx;
    int *img_idx; // indices of img_coords in each shell-pair
    int *img_offsets; // offset img_idx for each shell-pair
} PBCInt3c2eBounds;

#ifdef __CUDACC__
extern __constant__ int c_g_pair_idx[];
extern __constant__ int c_g_pair_offsets[];
extern __constant__ int c_g_cart_idx[];
#endif
#endif
