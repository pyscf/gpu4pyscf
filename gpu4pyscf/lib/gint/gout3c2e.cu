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
static void GINTgout3c2e_ipip(GINTEnvVars envs, double* __restrict__ gout, 
    double* __restrict__ g0, double* __restrict__ g1, double* __restrict__ g2, double* __restrict__ g3)
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

        double sxx = gout[9*i + 0];
        double sxy = gout[9*i + 1];
        double sxz = gout[9*i + 2];
        double syx = gout[9*i + 3];
        double syy = gout[9*i + 4];
        double syz = gout[9*i + 5];
        double szx = gout[9*i + 6];
        double szy = gout[9*i + 7];
        double szz = gout[9*i + 8];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n, ++ix, ++iy, ++iz) {
            sxx += g3[ix] * g0[iy] * g0[iz];
            sxy += g2[ix] * g1[iy] * g0[iz];
            sxz += g2[ix] * g0[iy] * g1[iz];
            syx += g1[ix] * g2[iy] * g0[iz];
            syy += g0[ix] * g3[iy] * g0[iz];
            syz += g0[ix] * g2[iy] * g1[iz];
            szx += g1[ix] * g0[iy] * g2[iz];
            szy += g0[ix] * g1[iy] * g2[iz];
            szz += g0[ix] * g0[iy] * g3[iz];
        }
        gout[9*i + 0] = sxx;
        gout[9*i + 1] = sxy;
        gout[9*i + 2] = sxz;
        gout[9*i + 3] = syx;
        gout[9*i + 4] = syy;
        gout[9*i + 5] = syz;
        gout[9*i + 6] = szx;
        gout[9*i + 7] = szy;
        gout[9*i + 8] = szz;
    }}}
}


template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTgout3c2e_ipip(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g0, double* __restrict__ g1, double* __restrict__ g2,
double* __restrict__ g3)
{
    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int nfi = (LI+1)*(LI+2)/2;
    const int nfj = (LJ+1)*(LJ+2)/2;
    const int nfk = (LK+1)*(LK+2)/2;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[LK] + ik;
        const int loc_j = c_l_locs[LJ] + ij;
        const int loc_i = c_l_locs[LI] + ii;

        int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
        int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
        int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

        double sxx = gout[9*i + 0];
        double sxy = gout[9*i + 1];
        double sxz = gout[9*i + 2];
        double syx = gout[9*i + 3];
        double syy = gout[9*i + 4];
        double syz = gout[9*i + 5];
        double szx = gout[9*i + 6];
        double szy = gout[9*i + 7];
        double szz = gout[9*i + 8];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n, ++ix, ++iy, ++iz) {
            sxx += g3[ix] * g0[iy] * g0[iz];
            sxy += g2[ix] * g1[iy] * g0[iz];
            sxz += g2[ix] * g0[iy] * g1[iz];
            syx += g1[ix] * g2[iy] * g0[iz];
            syy += g0[ix] * g3[iy] * g0[iz];
            syz += g0[ix] * g2[iy] * g1[iz];
            szx += g1[ix] * g0[iy] * g2[iz];
            szy += g0[ix] * g1[iy] * g2[iz];
            szz += g0[ix] * g0[iy] * g3[iz];
        }
        gout[9*i + 0] = sxx;
        gout[9*i + 1] = sxy;
        gout[9*i + 2] = sxz;
        gout[9*i + 3] = syx;
        gout[9*i + 4] = syy;
        gout[9*i + 5] = syz;
        gout[9*i + 6] = szx;
        gout[9*i + 7] = szy;
        gout[9*i + 8] = szz;
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

template <int NROOTS> __device__
static void GINTwrite_int3c2e_ipip_direct(GINTEnvVars envs, ERITensor eri, 
    double* __restrict__ g0, double* __restrict__ g1, double* __restrict__ g2, double* __restrict__ g3, 
    const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    const int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    const int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    const int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;

    double* __restrict__ pxx_eri;
    double* __restrict__ pxy_eri;
    double* __restrict__ pxz_eri;
    double* __restrict__ pyx_eri;
    double* __restrict__ pyy_eri;
    double* __restrict__ pyz_eri;
    double* __restrict__ pzx_eri;
    double* __restrict__ pzy_eri;
    double* __restrict__ pzz_eri;

    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    for (int n = 0, k = k0; k < k1; ++k) {
        pxx_eri = eri.data + 0 * lstride + k * kstride;
        pxy_eri = eri.data + 1 * lstride + k * kstride;
        pxz_eri = eri.data + 2 * lstride + k * kstride;
        pyx_eri = eri.data + 3 * lstride + k * kstride;
        pyy_eri = eri.data + 4 * lstride + k * kstride;
        pyz_eri = eri.data + 5 * lstride + k * kstride;
        pzx_eri = eri.data + 6 * lstride + k * kstride;
        pzy_eri = eri.data + 7 * lstride + k * kstride;
        pzz_eri = eri.data + 8 * lstride + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                const int loc_k = c_l_locs[lk] + (k-k0);
                const int loc_j = c_l_locs[lj] + (j-j0);
                const int loc_i = c_l_locs[li] + (i-i0);

                int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
                int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
                int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

                double eri_xx = 0;
                double eri_xy = 0;
                double eri_xz = 0;
                double eri_yx = 0;
                double eri_yy = 0;
                double eri_yz = 0;
                double eri_zx = 0;
                double eri_zy = 0;
                double eri_zz = 0;
                for (int ir = 0; ir < NROOTS; ++ir, ++ix, ++iy, ++iz){
                    double g0_x = g0[ix];
                    double g0_y = g0[iy];
                    double g0_z = g0[iz];
                    eri_xx += g3[ix] * g0_y   * g0_z  ;
                    eri_xy += g2[ix] * g1[iy] * g0_z  ;
                    eri_xz += g2[ix] * g0_y   * g1[iz];
                    eri_yx += g1[ix] * g2[iy] * g0_z  ;
                    eri_yy += g0_x   * g3[iy] * g0_z  ;
                    eri_yz += g0_x   * g2[iy] * g1[iz];
                    eri_zx += g1[ix] * g0_y   * g2[iz];
                    eri_zy += g0_x   * g1[iy] * g2[iz];
                    eri_zz += g0_x   * g0_y   * g3[iz];
                }
                int off = i+jstride*j;
                pxx_eri[off] += eri_xx;
                pxy_eri[off] += eri_xy;
                pxz_eri[off] += eri_xz;
                pyx_eri[off] += eri_yx;
                pyy_eri[off] += eri_yy;
                pyz_eri[off] += eri_yz;
                pzx_eri[off] += eri_zx;
                pzy_eri[off] += eri_zy;
                pzz_eri[off] += eri_zz;
            }
        }
    }
}

template <int LI, int LJ, int LK> __device__
static void GINTwrite_int3c2e_ipip_direct(GINTEnvVars envs, ERITensor eri, 
    double* __restrict__ g0, double* __restrict__ g1, double* __restrict__ g2, double* __restrict__ g3, 
    const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    
    const int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    
    double* __restrict__ pxx_eri;
    double* __restrict__ pxy_eri;
    double* __restrict__ pxz_eri;
    double* __restrict__ pyx_eri;
    double* __restrict__ pyy_eri;
    double* __restrict__ pyz_eri;
    double* __restrict__ pzx_eri;
    double* __restrict__ pzy_eri;
    double* __restrict__ pzz_eri;

    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    constexpr int NROOTS = (LI+LJ+LK+2)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    for (int ik = 0; ik < nfk; ++ik) {
        int k = k0 + ik;
        pxx_eri = eri.data + 0 * lstride + k * kstride;
        pxy_eri = eri.data + 1 * lstride + k * kstride;
        pxz_eri = eri.data + 2 * lstride + k * kstride;
        pyx_eri = eri.data + 3 * lstride + k * kstride;
        pyy_eri = eri.data + 4 * lstride + k * kstride;
        pyz_eri = eri.data + 5 * lstride + k * kstride;
        pzx_eri = eri.data + 6 * lstride + k * kstride;
        pzy_eri = eri.data + 7 * lstride + k * kstride;
        pzz_eri = eri.data + 8 * lstride + k * kstride;

        for (int ij = 0; ij < nfj; ++ij) {
            for (int ii = 0; ii < nfi; ++ii) {
                const int loc_k = c_l_locs[LK] + ik;
                const int loc_j = c_l_locs[LJ] + ij;
                const int loc_i = c_l_locs[LI] + ii;

                int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
                int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
                int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

                double eri_xx = 0;
                double eri_xy = 0;
                double eri_xz = 0;
                double eri_yx = 0;
                double eri_yy = 0;
                double eri_yz = 0;
                double eri_zx = 0;
                double eri_zy = 0;
                double eri_zz = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir, ++ix, ++iy, ++iz){
                    double g0_x = g0[ix];
                    double g0_y = g0[iy];
                    double g0_z = g0[iz];
                    eri_xx += g3[ix] * g0_y   * g0_z  ;
                    eri_xy += g2[ix] * g1[iy] * g0_z  ;
                    eri_xz += g2[ix] * g0_y   * g1[iz];
                    eri_yx += g1[ix] * g2[iy] * g0_z  ;
                    eri_yy += g0_x   * g3[iy] * g0_z  ;
                    eri_yz += g0_x   * g2[iy] * g1[iz];
                    eri_zx += g1[ix] * g0_y   * g2[iz];
                    eri_zy += g0_x   * g1[iy] * g2[iz];
                    eri_zz += g0_x   * g0_y   * g3[iz];
                }
                const int off = (ii+i0)+jstride*(ij+j0);
                pxx_eri[off] += eri_xx;
                pxy_eri[off] += eri_xy;
                pxz_eri[off] += eri_xz;
                pyx_eri[off] += eri_yx;
                pyy_eri[off] += eri_yy;
                pyz_eri[off] += eri_yz;
                pzx_eri[off] += eri_zx;
                pzy_eri[off] += eri_zy;
                pzz_eri[off] += eri_zz;
            }
        }
    }
}


template <int LI, int LJ, int LK> __device__
static void GINTwrite_int3c2e_ip_direct(GINTEnvVars envs, ERITensor eri, double* f, double* g, int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;

    const int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;

    double* __restrict__ px_eri;
    double* __restrict__ py_eri;
    double* __restrict__ pz_eri;

    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    constexpr int NROOTS = (LI+LJ+LK+1)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    for (int ik = 0, n = 0; ik < nfk; ++ik) {
        int k = ik + k0;
        px_eri = eri.data + 0 * lstride + k * kstride;
        py_eri = eri.data + 1 * lstride + k * kstride;
        pz_eri = eri.data + 2 * lstride + k * kstride;

        for (int ij = 0; ij < nfj; ++ij) {
            for (int ii = 0; ii < nfi; ++ii, ++n) {
                const int loc_k = c_l_locs[LK] + ik;
                const int loc_j = c_l_locs[LJ] + ij;
                const int loc_i = c_l_locs[LI] + ii;

                int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
                int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
                int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

                double eri_x = 0;
                double eri_y = 0;
                double eri_z = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir, ++ix, ++iy, ++iz){
                    eri_x += f[ix] * g[iy] * g[iz];
                    eri_y += g[ix] * f[iy] * g[iz];
                    eri_z += g[ix] * g[iy] * f[iz];
                }
                int off = (ii+i0)+jstride*(ij+j0);
                px_eri[off] += eri_x;
                py_eri[off] += eri_y;
                pz_eri[off] += eri_z;
            }
        }
    }
}

template <int NROOTS> __device__
static void GINTwrite_int3c2e_ip_direct(GINTEnvVars envs, ERITensor eri, double* f, double* g, int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    const int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    const int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    const int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;
    
    double* __restrict__ px_eri;
    double* __restrict__ py_eri;
    double* __restrict__ pz_eri;

    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;
    
    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    for (int n = 0, k = k0; k < k1; ++k) {
        px_eri = eri.data + 0 * lstride + k * kstride;
        py_eri = eri.data + 1 * lstride + k * kstride;
        pz_eri = eri.data + 2 * lstride + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                const int loc_k = c_l_locs[lk] + (k-k0);
                const int loc_j = c_l_locs[lj] + (j-j0);
                const int loc_i = c_l_locs[li] + (i-i0);

                int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
                int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
                int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

                double eri_x = 0;
                double eri_y = 0;
                double eri_z = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir, ++ix, ++iy, ++iz){
                    eri_x += f[ix] * g[iy] * g[iz];
                    eri_y += g[ix] * f[iy] * g[iz];
                    eri_z += g[ix] * g[iy] * f[iz];
                }
                int off = i+jstride*j;
                px_eri[off] += eri_x;
                py_eri[off] += eri_y;
                pz_eri[off] += eri_z;
            }
        }
    }
}

template <int NROOTS> __device__
static void GINTmemset_int3c2e(GINTEnvVars envs, ERITensor eri, int ish, int jsh, int ksh)
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
    
    double* __restrict__ p_eri;

    for (int n = 0, k = k0; k < k1; ++k) {
        p_eri = eri.data + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                p_eri[i+jstride*j] = 0;
            }
        }
    }
}

template <int NROOTS> __device__
static void GINTwrite_int3c2e_direct(GINTEnvVars envs, ERITensor eri, double* g, int ish, int jsh, int ksh)
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

    double* __restrict__ p_eri;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    for (int n = 0, k = k0; k < k1; ++k) {
        p_eri = eri.data + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                const int loc_k = c_l_locs[lk] + (k-k0);
                const int loc_j = c_l_locs[lj] + (j-j0);
                const int loc_i = c_l_locs[li] + (i-i0);

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double eri = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    eri += g[ix + ir] * g[iy + ir] * g[iz + ir];
                }
                int off = i+jstride*j;
                p_eri[off] += eri;
            }
        }
    }
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
    
    double* __restrict__ pxx_eri;
    double* __restrict__ pxy_eri;
    double* __restrict__ pxz_eri;
    double* __restrict__ pyx_eri;
    double* __restrict__ pyy_eri;
    double* __restrict__ pyz_eri;
    double* __restrict__ pzx_eri;
    double* __restrict__ pzy_eri;
    double* __restrict__ pzz_eri;

    for (int n = 0, k = k0; k < k1; ++k) {
        pxx_eri = eri.data + 0 * lstride + k * kstride;
        pxy_eri = eri.data + 1 * lstride + k * kstride;
        pxz_eri = eri.data + 2 * lstride + k * kstride;
        pyx_eri = eri.data + 3 * lstride + k * kstride;
        pyy_eri = eri.data + 4 * lstride + k * kstride;
        pyz_eri = eri.data + 5 * lstride + k * kstride;
        pzx_eri = eri.data + 6 * lstride + k * kstride;
        pzy_eri = eri.data + 7 * lstride + k * kstride;
        pzz_eri = eri.data + 8 * lstride + k * kstride;

        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i, ++n) {
                double sxx = gout[9 * n];
                double sxy = gout[9 * n + 1];
                double sxz = gout[9 * n + 2];
                double syx = gout[9 * n + 3];
                double syy = gout[9 * n + 4];
                double syz = gout[9 * n + 5];
                double szx = gout[9 * n + 6];
                double szy = gout[9 * n + 7];
                double szz = gout[9 * n + 8];

                int off = i + jstride * j;
                pxx_eri[off] = sxx;
                pxy_eri[off] = sxy;
                pxz_eri[off] = sxz;
                pyx_eri[off] = syx;
                pyy_eri[off] = syy;
                pyz_eri[off] = syz;
                pzx_eri[off] = szx;
                pzy_eri[off] = szy;
                pzz_eri[off] = szz;
            }
        }
    }
}
