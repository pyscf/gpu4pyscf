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

#include <math.h>
#include "cint2e.cuh"

// This function assumes i_l >= j_l
template <int NROOTS>
__device__
static void GINT_g1e(double* __restrict__ g, const double* __restrict__ grid_point,
                     const int ish, const int jsh, const int prim_ij,
                     const int i_l, const int j_l, const double charge_exponent, const double omega)
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
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
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

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0;
        gy[i_root] = prefactor;
        gz[i_root] = w[i_root];

        const double u2 = a0 * u[i_root];
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const double b10 = 0.5 / aij * (1.0 - qt2_over_p_plus_q);
        const double c00x = PAx - qt2_over_p_plus_q * PCx;
        const double c00y = PAy - qt2_over_p_plus_q * PCy;
        const double c00z = PAz - qt2_over_p_plus_q * PCz;

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

        for (int j_rys = 0; j_rys < j_l; j_rys++) {
            for (int i_rys = i_l + j_l - j_rys - 1; i_rys >= 0; i_rys--) {
                for (int i_root = 0; i_root < NROOTS; i_root++) {
                    gx[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gx[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABx * gx[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gy[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gy[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABy * gy[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gz[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gz[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABz * gz[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                }
            }
        }
    }
}

// This function assumes i_l >= j_l
template <int NROOTS>
__device__
static void GINT_g1e_save_u2(double* __restrict__ g, double* __restrict__ u2_save, const double* __restrict__ grid_point,
                             const int ish, const int jsh, const int prim_ij,
                             const int i_l, const int j_l, const double charge_exponent, const double omega)
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
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
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

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0;
        gy[i_root] = prefactor;
        gz[i_root] = w[i_root];

        const double u2 = a0 * u[i_root];
        u2_save[i_root] = charge_exponent > 0.0 ? u2 * charge_exponent / (u2 + charge_exponent) : u2;
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const double b10 = 0.5 / aij * (1.0 - qt2_over_p_plus_q);
        const double c00x = PAx - qt2_over_p_plus_q * PCx;
        const double c00y = PAy - qt2_over_p_plus_q * PCy;
        const double c00z = PAz - qt2_over_p_plus_q * PCz;

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

        for (int j_rys = 0; j_rys < j_l; j_rys++) {
            for (int i_rys = i_l + j_l - j_rys - 1; i_rys >= 0; i_rys--) {
                for (int i_root = 0; i_root < NROOTS; i_root++) {
                    gx[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gx[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABx * gx[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gy[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gy[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABy * gy[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                    gz[i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS] = gz[i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS] + ABz * gz[i_root + (i_rys + j_rys * (i_l+1)) * NROOTS];
                }
            }
        }
    }
}

template <int L_SUM>
__device__
static void GINT_g1e_without_hrr(double* __restrict__ g, const double grid_x, const double grid_y, const double grid_z,
                                 const int ish, const int prim_ij, const double charge_exponent, const double omega)
{
    constexpr int NROOTS = L_SUM / 2 + 1;

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

    const double PCx = Px - grid_x;
    const double PCy = Py - grid_y;
    const double PCz = Pz - grid_z;

    double a0 = aij;
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);

    const double* __restrict__ u = uw;
    const double* __restrict__ w = u + NROOTS;
    constexpr int g_size = NROOTS * (L_SUM + 1);
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

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0;
        gy[i_root] = prefactor;
        gz[i_root] = w[i_root];

        const double u2 = a0 * u[i_root];
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const double b10 = 0.5 / aij * (1.0 - qt2_over_p_plus_q);
        const double c00x = PAx - qt2_over_p_plus_q * PCx;
        const double c00y = PAy - qt2_over_p_plus_q * PCy;
        const double c00z = PAz - qt2_over_p_plus_q * PCz;

        if constexpr (L_SUM > 0) {
            double s0x = gx[i_root]; // i - 1
            double s0y = gy[i_root];
            double s0z = gz[i_root];
            double s1x = c00x * s0x; // i
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[i_root + 1 * NROOTS] = s1x;
            gy[i_root + 1 * NROOTS] = s1y;
            gz[i_root + 1 * NROOTS] = s1z;
#pragma unroll
            for (int i_rys = 1; i_rys < L_SUM; i_rys++) {
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

}

template <int L_SUM>
__device__
static void GINT_g1e_without_hrr_save_u2(double* __restrict__ g, double* __restrict__ u2_save, const double grid_x, const double grid_y, const double grid_z,
                                         const int ish, const int prim_ij, const double charge_exponent, const double omega)
{
    constexpr int NROOTS = L_SUM / 2 + 1;

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

    const double PCx = Px - grid_x;
    const double PCy = Py - grid_y;
    const double PCz = Pz - grid_z;

    double a0 = aij;
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);

    const double* __restrict__ u = uw;
    const double* __restrict__ w = u + NROOTS;
    constexpr int g_size = NROOTS * (L_SUM + 1);
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

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0;
        gy[i_root] = prefactor;
        gz[i_root] = w[i_root];

        const double u2 = a0 * u[i_root];
        u2_save[i_root] = charge_exponent > 0.0 ? u2 * charge_exponent / (u2 + charge_exponent) : u2;
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const double b10 = 0.5 / aij * (1.0 - qt2_over_p_plus_q);
        const double c00x = PAx - qt2_over_p_plus_q * PCx;
        const double c00y = PAy - qt2_over_p_plus_q * PCy;
        const double c00z = PAz - qt2_over_p_plus_q * PCz;

        if constexpr (L_SUM > 0) {
            double s0x = gx[i_root]; // i - 1
            double s0y = gy[i_root];
            double s0z = gz[i_root];
            double s1x = c00x * s0x; // i
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[i_root + 1 * NROOTS] = s1x;
            gy[i_root + 1 * NROOTS] = s1y;
            gz[i_root + 1 * NROOTS] = s1z;
#pragma unroll
            for (int i_rys = 1; i_rys < L_SUM; i_rys++) {
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

}
