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

#include <math.h>
#include "cint2e.cuh"

// This function assumes i_l >= j_l
template <int NROOTS>
__device__
static void GINTg1e(double* __restrict__ g, const double* __restrict__ grid_point, const int ish, const int jsh, const int prim_ij,
                    const int i_l, const int j_l, const double omega)
{
    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;
    const double aij = a12[prim_ij];
    const double eij = e12[prim_ij];
    const double Px  = x12[prim_ij];
    const double Py  = y12[prim_ij];
    const double Pz  = z12[prim_ij];
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];

    const double PCx = Px - Cx;
    const double PCy = Py - Cy;
    const double PCz = Pz - Cz;
    double a0 = aij;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / a0 * eij;
    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);

    const double* __restrict__ u = uw;
    const double* __restrict__ w = u + NROOTS;
    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    double* __restrict__ gx = g;
    double* __restrict__ gy = g + g_size;
    double* __restrict__ gz = g + g_size * 2;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];
    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;

    // int nmax = envs.li_ceil + envs.lj_ceil;
    // int mmax = envs.lk_ceil + envs.ll_ceil;
    // int ijmin = envs.ijmin;
    // int klmin = envs.klmin;
    // int dm = envs.stride_klmax;
    // int dn = envs.stride_ijmax;
    // int di = envs.stride_ijmax;
    // int dj = envs.stride_ijmin;
    // int dk = envs.stride_klmax;
    // int dl = envs.stride_klmin;
    // int dij = envs.g_size_ij;
    // int i, k;
    // int j, l, m, n, off;
    // double tmpb0;
    // double s0x, s1x, s2x, t0x, t1x;
    // double s0y, s1y, s2y, t0y, t1y;
    // double s0z, s1z, s2z, t0z, t1z;
    // double u2, tmp1, tmp2, tmp3, tmp4;
    // double b00, b10, b01, c00x, c00y, c00z, c0px, c0py, c0pz;

    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0;
        gy[i_root] = prefactor;
        gz[i_root] = w[i_root];

        const double u2 = a0 * u[i_root];
        const double t2 = u2 / (u2 + a0);
        const double b10 = 0.5 / a0 * (1.0 - t2);
        const double c00x = PAx - t2 * PCx;
        const double c00y = PAy - t2 * PCy;
        const double c00z = PAz - t2 * PCz;

        if (i_l + j_l > 0) {
            double s0x = gx[i_root]; // i - 1
            double s0y = gy[i_root];
            double s0z = gz[i_root];
            double s1x = c00x * s0x; // i
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[i_root + 1 * NROOTS] = s1x;
            gy[i_root + 1 * NROOTS] = s1y;
            gz[i_root + 1 * NROOTS] = s1z;
            for (int i_rys = 1; i_rys < i_l + j_l; i_rys++) {
                const double s2x = c00x * s1x + i_rys * b10 * s0x; // i + 1
                const double s2y = c00y * s1y + i_rys * b10 * s0y;
                const double s2z = c00z * s1z + i_rys * b10 * s0z;
                gx[i_root + (i_rys+1) * NROOTS] = s2x;
                gy[i_root + (i_rys+1) * NROOTS] = s2y;
                gz[i_root + (i_rys+1) * NROOTS] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }
    }

    if (j_l > 0) {
        const double Bx = bas_x[jsh];
        const double By = bas_y[jsh];
        const double Bz = bas_z[jsh];
        const double ABx = Ax - Bx;
        const double ABy = Ay - By;
        const double ABz = Az - Bz;

        for (int i_root = 0; i_root < NROOTS; i_root++) {
            for (int j_rys = 0; j_rys < j_l; j_rys++) {
                for (int i_rys = i_l + j_l - j_rys - 1; i_rys >= 0; i_rys--) {
                    gx[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gx[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABx * gx[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gy[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gy[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABy * gy[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gz[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gz[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABz * gz[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                }
            }
        }
    }
}

