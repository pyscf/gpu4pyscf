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

#include <stdint.h>

#define WARP_SIZE       32
#define WARPS           8
#define THREADS         (WARP_SIZE*WARPS)
#define LMAX            4

#define PRIMBAS_SLOTS   4
#define PRIMBAS_ANG     0
#define PRIMBAS_EXP     1
#define PRIMBAS_COEFF   2
#define PRIMBAS_COORD   3

#define MIN(x, y)       ((x) < (y) ? (x) : (y))
#define MAX(x, y)       ((x) > (y) ? (x) : (y))

// an index associated to enumerate([_ for t in range(L+1) for u in range(L+1-t)])
//#define ADDR2(l, t, u)  ((l+1)*(l+2)/2 - (l-(t)+1)*(l-(t)+2)/2 + (u))
#define ADDR2(l, t, u)  ((t)*((l)*2+3-(t))/2 + (u))
// an index associated to enumerate([_ for t in range(L+1) for u in range(L+1-t) for v in range(L+1-t-u)])
#define ADDR3(l, t, u, v) \
        ((l+1)*(l+2)*(l+3)/6 - ((l)-(t)+1)*((l)-(t)+2)*((l)-(t)+3)/6 + \
         ((l)-(t)+1)*((l)-(t)+2)/2 - ((l)-(t)-(u)+1)*((l)-(t)-(u)+2)/2 + (v))

#ifndef HAVE_DEFINED_MGRIDENVVAS_H
#define HAVE_DEFINED_MGRIDENVVAS_H
typedef struct {
    int nbas_i;
    int nbas_j;
    int nao;
    int *bas; // the supmol._bas, shaped as [:,PRIMBAS_SLOTS]
    double *env; // the supmol._env
    // ao_loc points to the addresses of the original contracted GTOs, not the
    // uncontracted GTOs. The adjcent values in ao_loc may point to the same
    // address. (ao_loc[n+1] - ao_loc[n]) cannot be used as the dimension for
    // each shell.
    int *ao_loc;
    double *lattice_params;
} MGridEnvVars;

typedef struct {
    int nshl_pair;
    int *bas_ij_idx;
    int ngrid_radius;
    int mesh[3];
} MGridBounds;

typedef struct {
    uint8_t x;
    uint8_t y;
} Fold2Index;

typedef struct {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t _padding;
} Fold3Index;

#ifdef __CUDACC__
extern __constant__ Fold2Index c_i_in_fold2idx[];
extern __constant__ Fold3Index c_i_in_fold3idx[];
#endif
#endif
