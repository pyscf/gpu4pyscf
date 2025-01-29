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
#define BATCHES_PER_BLOCK       16
#define L_AUX_MAX       6

#ifndef HAVE_DEFINED_INT3CENVVAS_H
#define HAVE_DEFINED_INT3CENVVAS_H
typedef struct {
    uint16_t natm;
    uint16_t nbas;
    int *atm;
    int *bas;
    double *env;
    int *ao_loc;
    float log_cutoff;
} Int3c2eEnvVars;

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
    uint16_t naux;
    uint16_t nksh;
    uint16_t ksh0;
    int nshl_pair;
    // The effective basis pair Id = ish*nbas+jsh
    int *bas_ij_idx;
} Int3c2eBounds;

typedef struct {
    int naux;
    // The effective basis pair Id = ish*nbas+jsh
    int *bas_ij_idx;
    // the bas_ij_idx offset for each blockIdx.x
    int *shl_pair_offsets;
    // the AO-pair offset (address) in the output tensor for each blockIdx.x
    int *ao_pair_loc;
    // the auxiliary function offset (address) in the output tensor for each blockIdx.y
    int *ksh_offsets;
    // nst_per_block for each (li,lj,lk) pattern
    int *nst_lookup;
} BDiv3c2eBounds;

#ifdef __CUDACC__
extern __constant__ int c_g_pair_idx[];
extern __constant__ int c_g_pair_offsets[];
extern __constant__ int c_g_cart_idx[];
#endif
#endif
