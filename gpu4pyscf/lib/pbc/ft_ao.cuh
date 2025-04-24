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

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#ifndef HAVE_DEFINED_AFTENVVAS_H
#define HAVE_DEFINED_AFTENVVAS_H
typedef struct {
    uint16_t cell0_natm; // in bvk-cell
    uint16_t cell0_nbas; // in bvk-cell
    uint16_t bvk_ncells; // number of images in the BvK cell
    uint16_t nimgs; // number of images in lattice sum
    int *atm;
    int *bas;
    double *env;
    int *ao_loc; // in bvk-cell
    double *img_coords; // vectors in lattice sum
} AFTIntEnvVars;

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
#endif
