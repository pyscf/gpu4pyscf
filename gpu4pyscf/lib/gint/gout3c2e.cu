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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "g2e.h"
#include "cint2e.cuh"

template <int NROOTS> __device__
static void GINTgout3c2e_ip(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ f, double* __restrict__ g)
{
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    const int nfi = (li+1)*(li+2)/2;
    const int nfj = (lj+1)*(lj+2)/2;
    const int nfk = (lk+1)*(lk+2)/2;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[lk] + ik;
        const int loc_j = c_l_locs[lj] + ij;
        const int loc_i = c_l_locs[li] + ii;

        int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        double sx = gout[3*i + 0];
        double sy = gout[3*i + 1];
        double sz = gout[3*i + 2];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n) {
            sx += f[ix+n] * g[iy+n] * g[iz+n];
            sy += g[ix+n] * f[iy+n] * g[iz+n];
            sz += g[ix+n] * g[iy+n] * f[iz+n];
        }
        gout[3*i + 0] = sx;
        gout[3*i + 1] = sy;
        gout[3*i + 2] = sz;
    }}}
}


template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTgout3c2e_ip(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ f, double* __restrict__ g)
{
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[LK] + ik;
        const int loc_j = c_l_locs[LJ] + ij;
        const int loc_i = c_l_locs[LI] + ii;

        int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        double sx = gout[3*i + 0];
        double sy = gout[3*i + 1];
        double sz = gout[3*i + 2];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n) {
            sx += f[ix+n] * g[iy+n] * g[iz+n];
            sy += g[ix+n] * f[iy+n] * g[iz+n];
            sz += g[ix+n] * g[iy+n] * f[iz+n];
        }
        gout[3*i + 0] = sx;
        gout[3*i + 1] = sy;
        gout[3*i + 2] = sz;
    }}}
}


template <int NROOTS> __device__
static void GINTgout3c2e(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g)
{
    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    const int nfi = (li+1)*(li+2)/2;
    const int nfj = (lj+1)*(lj+2)/2;
    const int nfk = (lk+1)*(lk+2)/2;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[lk] + ik;
        const int loc_j = c_l_locs[lj] + ij;
        const int loc_i = c_l_locs[li] + ii;

        int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
        int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
        int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

        double s = gout[i];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n) {
            s += g[ix+n] * g[iy+n] * g[iz+n];
        }
        gout[i] = s;
    }}}
}


__device__
static void GINTwrite_int3c2e(ERITensor eri, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;

    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;

    double s;
    double* __restrict__ peri;
    for (int n = 0, k = k0; k < k1; ++k) {
        peri = eri.data + k * kstride;
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                s = gout[n];
                peri[i+jstride*j] = s;
            }
        }
    }
}


__device__
static void GINTwrite_int3c2e_ip(ERITensor eri, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
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

    double* __restrict__ px_eri;
    double* __restrict__ py_eri;
    double* __restrict__ pz_eri;

    for (int n = 0, k = k0; k < k1; ++k) {
        px_eri = eri.data + 0 * lstride + k * kstride;
        py_eri = eri.data + 1 * lstride + k * kstride;
        pz_eri = eri.data + 2 * lstride + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                double sx = gout[3 * n];
                double sy = gout[3 * n + 1];
                double sz = gout[3 * n + 2];

                int off = i + jstride * j;
                px_eri[off] = sx;
                py_eri[off] = sy;
                pz_eri[off] = sz;
            }
        }
    }
}

__device__
static void GINTwrite_int3c2e_ipip(ERITensor eri, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
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

    for (int n = 0, k = k0; k < k1; ++k) {
    for (int j = j0; j < j1; ++j) {
    for (int i = i0; i < i1; ++i, ++n) {
        for (int ix = 0; ix < 9; ix++){
            int off = ix*lstride + k*kstride + j*jstride + i;
            eri.data[off] = gout[9*n + ix];
        }
    }}}
}
