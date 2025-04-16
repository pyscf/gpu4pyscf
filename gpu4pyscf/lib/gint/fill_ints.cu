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

#include <stdio.h>
#include "gint/gint.h"
#include "gint/cint2e.cuh"

__device__
void GINTwrite_ints_s2(ERITensor eri, double* __restrict__ gout,
                       int ish, int jsh, int ksh, int lsh)
{
    int *ao_loc = c_bpcache.ao_loc;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh  ] - eri.ao_offsets_l;
    int l1 = ao_loc[lsh+1] - eri.ao_offsets_l;
    int i, j, k, l, n;
    double s;
    double* __restrict__ peri;
    for (n = 0, l = l0; l < l1; ++l) {
        for (k = k0; k < k1; ++k) {
            peri = eri.data + l * lstride + k * kstride;
            for (j = j0; j < j1; ++j) {
                for (i = i0; i < i1; ++i, ++n) {
                    s = gout[n];
                    peri[i+jstride*j] = s;
                    //peri[j+jstride*i] = s;
                }
            }
        }
    }
}


__device__
void GINTwrite_ints_sph_s2(ERITensor eri, double* __restrict__ gout,
                       int ish, int jsh, int ksh, int lsh)
{
    int *ao_loc = c_bpcache.ao_loc;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    size_t lstride = eri.stride_l;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;
    int l0 = ao_loc[lsh  ] - eri.ao_offsets_l;
    int l1 = ao_loc[lsh+1] - eri.ao_offsets_l;
    int i, j, k, l, n;
    double s;
    double* __restrict__ peri;
    for (n = 0, l = l0; l < l1; ++l) {
        for (k = k0; k < k1; ++k) {
            peri = eri.data + l * lstride + k * kstride;
            for (j = j0; j < j1; ++j) {
                for (i = i0; i < i1; ++i, ++n) {
                    s = gout[n];
                    peri[i+jstride*j] = s;
                    //peri[j+jstride*i] = s;
                }
            }
        }
    }
}
