/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
