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
typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t lk;
    uint8_t ll;
    int npairs_ij;
    int npairs_kl;
    int *pair_ij_mapping; // the significant ij pairs, mapping to i*nao+j
    int *pair_kl_mapping;
    int *pair_ij_loc; // offsets to the input dm_xyz for each ij pair
    int *pair_kl_loc;
    float *qd_ij_max; // largest dm_cond*q_cond within each block for ij pair
    float *qd_kl_max;
    float *q_cond;
    float cutoff; // cutoff to screening schwarz estimation q_ij+q_kl
} MDBoundsInfo;
