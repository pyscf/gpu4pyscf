/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
   if (NROOTS < 8) {
        int nf = envs.nf;
        int16_t *idx = c_idx4c;
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        int i, n, ix, iy, iz;

        for (i = 0; i < nf; i++) {
            ix = idx[i];
            iy = idy[i];
            iz = idz[i];

            double sx = gout[3*i + 0];
            double sy = gout[3*i + 1];
            double sz = gout[3*i + 2];
#pragma unroll
            for (n = 0; n < NROOTS; ++n) {
                sx += f[ix+n] * g[iy+n] * g[iz+n];
                sy += g[ix+n] * f[iy+n] * g[iz+n];
                sz += g[ix+n] * g[iy+n] * f[iz+n];
            }
            gout[3*i + 0] = sx;
            gout[3*i + 1] = sy;
            gout[3*i + 2] = sz;
        }
   }
   else {
        int nf = envs.nf;
        int16_t *idx = c_idx4c;
        if (nf > NFhgg) {
            idx = envs.idx;
        }
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        double sx, sy, sz;
        int i, n, ix, iy, iz;

        for (i = 0; i < nf; i++) {
            ix = idx[i];
            iy = idy[i];
            iz = idz[i];
            sx = gout[3*i + 0];
            sy = gout[3*i + 1];
            sz = gout[3*i + 2];
#pragma unroll
            for (n = 0; n < NROOTS; ++n) {
                sx += f[ix+n] * g[iy+n] * g[iz+n];
                sy += g[ix+n] * f[iy+n] * g[iz+n];
                sz += g[ix+n] * g[iy+n] * f[iz+n];
            }
            gout[3*i + 0] = sx;
            gout[3*i + 1] = sy;
            gout[3*i + 2] = sz;
        }
    }
}

template <int NROOTS> __device__
static void GINTgout3c2e_ipip(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g0, double* __restrict__ g1, double* __restrict__ g2,
double* __restrict__ g3)
{
    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    int i, n, ix, iy, iz;

    for (i = 0; i < nf; i++) {
        ix = idx[i];
        iy = idy[i];
        iz = idz[i];

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
        for (n = 0; n < NROOTS; ++n) {
            sxx += g3[ix+n] * g0[iy+n] * g0[iz+n];
            sxy += g2[ix+n] * g1[iy+n] * g0[iz+n];
            sxz += g2[ix+n] * g0[iy+n] * g1[iz+n];
            syx += g1[ix+n] * g2[iy+n] * g0[iz+n];
            syy += g0[ix+n] * g3[iy+n] * g0[iz+n];
            syz += g0[ix+n] * g2[iy+n] * g1[iz+n];
            szx += g1[ix+n] * g0[iy+n] * g2[iz+n];
            szy += g0[ix+n] * g1[iy+n] * g2[iz+n];
            szz += g0[ix+n] * g0[iy+n] * g3[iz+n];
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
    }
}

template <int NROOTS> __device__
static void GINTgout3c2e(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g)
{
    if (NROOTS < 8) {
        int nf = envs.nf;
        int16_t *idx = c_idx4c;
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        double s;
        int i, n, ix, iy, iz;

        for (i = 0; i < nf; i++) {
            ix = idx[i];
            iy = idy[i];
            iz = idz[i];
            s = gout[i];
#pragma unroll
            for (n = 0; n < NROOTS; ++n) {
                s += g[ix+n] * g[iy+n] * g[iz+n];
            }
            gout[i] = s;
        }
    } else {
        int nf = envs.nf;
        int16_t *idx = c_idx4c;
        if (nf > NFffff) {
            idx = envs.idx;
        }
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        double s;
        int i, n, ix, iy, iz;

        for (i = 0; i < nf; i++) {
            ix = idx[i];
            iy = idy[i];
            iz = idz[i];
            s = gout[i];
#pragma unroll
            for (n = 0; n < NROOTS; ++n) {
                s += g[ix+n] * g[iy+n] * g[iz+n];
            }
            gout[i] = s;
        }
    }
}

template <int NROOTS> __device__
static void GINTwrite_int3c2e_ipip_direct(GINTEnvVars envs, ERITensor eri, double* g0, double* g1, double* g2, double* g3, int ish, int jsh, int ksh)
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
    int i, j, k, n;
    double* __restrict__ pxx_eri;
    double* __restrict__ pxy_eri;
    double* __restrict__ pxz_eri;
    double* __restrict__ pyx_eri;
    double* __restrict__ pyy_eri;
    double* __restrict__ pyz_eri;
    double* __restrict__ pzx_eri;
    double* __restrict__ pzy_eri;
    double* __restrict__ pzz_eri;

    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    int ix, iy, iz, off;

    for (n = 0, k = k0; k < k1; ++k) {
        pxx_eri = eri.data + 0 * lstride + k * kstride;
        pxy_eri = eri.data + 1 * lstride + k * kstride;
        pxz_eri = eri.data + 2 * lstride + k * kstride;
        pyx_eri = eri.data + 3 * lstride + k * kstride;
        pyy_eri = eri.data + 4 * lstride + k * kstride;
        pyz_eri = eri.data + 5 * lstride + k * kstride;
        pzx_eri = eri.data + 6 * lstride + k * kstride;
        pzy_eri = eri.data + 7 * lstride + k * kstride;
        pzz_eri = eri.data + 8 * lstride + k * kstride;

        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                ix = idx[n];
                iy = idy[n];
                iz = idz[n];

                double eri_xx = 0;
                double eri_xy = 0;
                double eri_xz = 0;
                double eri_yx = 0;
                double eri_yy = 0;
                double eri_yz = 0;
                double eri_zx = 0;
                double eri_zy = 0;
                double eri_zz = 0;
                for (int ir = 0; ir < NROOTS; ++ir){
                    eri_xx += g3[ix + ir] * g0[iy + ir] * g0[iz + ir];
                    eri_xy += g2[ix + ir] * g1[iy + ir] * g0[iz + ir];
                    eri_xz += g2[ix + ir] * g0[iy + ir] * g1[iz + ir];
                    eri_yx += g1[ix + ir] * g2[iy + ir] * g0[iz + ir];
                    eri_yy += g0[ix + ir] * g3[iy + ir] * g0[iz + ir];
                    eri_yz += g0[ix + ir] * g2[iy + ir] * g1[iz + ir];
                    eri_zx += g1[ix + ir] * g0[iy + ir] * g2[iz + ir];
                    eri_zy += g0[ix + ir] * g1[iy + ir] * g2[iz + ir];
                    eri_zz += g0[ix + ir] * g0[iy + ir] * g3[iz + ir];
                }
                off = i+jstride*j;
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


template <int NROOTS> __device__
static void GINTwrite_int3c2e_ip_direct(GINTEnvVars envs, ERITensor eri, double* f, double* g, int ish, int jsh, int ksh)
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
    int i, j, k, n;
    double* __restrict__ px_eri;
    double* __restrict__ py_eri;
    double* __restrict__ pz_eri;

    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    int ix, iy, iz, off;

    for (n = 0, k = k0; k < k1; ++k) {
        px_eri = eri.data + 0 * lstride + k * kstride;
        py_eri = eri.data + 1 * lstride + k * kstride;
        pz_eri = eri.data + 2 * lstride + k * kstride;

        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                ix = idx[n];
                iy = idy[n];
                iz = idz[n];

                double eri_x = 0;
                double eri_y = 0;
                double eri_z = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    eri_x += f[ix + ir] * g[iy + ir] * g[iz + ir];
                    eri_y += g[ix + ir] * f[iy + ir] * g[iz + ir];
                    eri_z += g[ix + ir] * g[iy + ir] * f[iz + ir];
                }
                off = i+jstride*j;
                px_eri[off] += eri_x;
                py_eri[off] += eri_y;
                pz_eri[off] += eri_z;
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
    size_t lstride = eri.stride_l;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;
    int i, j, k, n;
    double* __restrict__ p_eri;

    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    int ix, iy, iz, off;

    for (n = 0, k = k0; k < k1; ++k) {
        p_eri = eri.data + 0 * lstride + k * kstride;

        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                ix = idx[n];
                iy = idy[n];
                iz = idz[n];

                double eri = 0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    eri += g[ix + ir] * g[iy + ir] * g[iz + ir];
                }
                off = i+jstride*j;
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
    size_t lstride = eri.stride_l;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;
    int i, j, k, l, n;
    double s;
    double* __restrict__ peri;
    for (n = 0, k = k0; k < k1; ++k) {
        peri = eri.data + l * lstride + k * kstride;
        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
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
    int i, j, k, n, off;
    double sx, sy, sz;
    double* __restrict__ px_eri;
    double* __restrict__ py_eri;
    double* __restrict__ pz_eri;

    for (n = 0, k = k0; k < k1; ++k) {
        px_eri = eri.data + 0 * lstride + k * kstride;
        py_eri = eri.data + 1 * lstride + k * kstride;
        pz_eri = eri.data + 2 * lstride + k * kstride;

        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                sx = gout[3 * n];
                sy = gout[3 * n + 1];
                sz = gout[3 * n + 2];

                off = i + jstride * j;
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
    int i, j, k, n, off;
    double* __restrict__ pxx_eri;
    double* __restrict__ pxy_eri;
    double* __restrict__ pxz_eri;
    double* __restrict__ pyx_eri;
    double* __restrict__ pyy_eri;
    double* __restrict__ pyz_eri;
    double* __restrict__ pzx_eri;
    double* __restrict__ pzy_eri;
    double* __restrict__ pzz_eri;

    for (n = 0, k = k0; k < k1; ++k) {
        pxx_eri = eri.data + 0 * lstride + k * kstride;
        pxy_eri = eri.data + 1 * lstride + k * kstride;
        pxz_eri = eri.data + 2 * lstride + k * kstride;
        pyx_eri = eri.data + 3 * lstride + k * kstride;
        pyy_eri = eri.data + 4 * lstride + k * kstride;
        pyz_eri = eri.data + 5 * lstride + k * kstride;
        pzx_eri = eri.data + 6 * lstride + k * kstride;
        pzy_eri = eri.data + 7 * lstride + k * kstride;
        pzz_eri = eri.data + 8 * lstride + k * kstride;

        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                double sxx = gout[9 * n];
                double sxy = gout[9 * n + 1];
                double sxz = gout[9 * n + 2];
                double syx = gout[9 * n + 3];
                double syy = gout[9 * n + 4];
                double syz = gout[9 * n + 5];
                double szx = gout[9 * n + 6];
                double szy = gout[9 * n + 7];
                double szz = gout[9 * n + 8];

                off = i + jstride * j;
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
