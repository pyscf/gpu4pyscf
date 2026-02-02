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
    int natm;
    int nbas;
    int *atm;
    int *bas;
    double *env;
    int *ao_loc;
    float log_cutoff;
} Int3c2eEnvVars;

typedef struct {
    int li;
    int lj;
    int lk;
    int nroots;
    int nfi;
    int nfij;
    int nfk;
    int iprim;
    int jprim;
    int kprim;
    int stride_i;
    int stride_j;
    int stride_k;
    int g_size;
    int naux;
    int nksh;
    int ksh0;
    int nshl_pair;
    // The effective basis pair Id = ish*nbas+jsh
    int *bas_ij_idx;
} Int3c2eBounds;

typedef struct {
    int naux;
    int aux_sh_offset;
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

#ifdef USE_SYCL
#include "gint/sycl_device.hpp"

// these two vars are defined in gvhf-rys/vhf.cuh
#define LMAX            4
#define LMAX1           (LMAX+1)

extern SYCL_EXTERNAL sycl_device_global<int[3675]> s_g_pair_idx; // corresponding to LMAX=4
extern SYCL_EXTERNAL sycl_device_global<int[LMAX1*LMAX1]> s_g_pair_offsets;
extern SYCL_EXTERNAL sycl_device_global<int[252]> s_g_cart_idx; // corresponding to LMAX=6
#endif // USE_SYCL

#endif
