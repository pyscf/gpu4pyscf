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

#include "gint.h"

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ipip1_charge_contracted(const double* g, double* local_output, const double minus_two_a, const double prefactor, const int i_l, const int j_l)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 2 + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const int loc_j = c_l_locs[j_l] + j;
            const int loc_i = c_l_locs[i_l] + i;
            const int ix = idx[loc_i];
            const int iy = idy[loc_i];
            const int iz = idz[loc_i];
            const int jx = idx[loc_j];
            const int jy = idy[loc_j];
            const int jz = idz[loc_j];
            const int gx_offset = ix + jx * (i_l + 2 + 1);
            const int gy_offset = iy + jy * (i_l + 2 + 1);
            const int gz_offset = iz + jz * (i_l + 2 + 1);

            double d2eri_dAxdAx = 0;
            double d2eri_dAxdAy = 0;
            double d2eri_dAxdAz = 0;
            double d2eri_dAydAy = 0;
            double d2eri_dAydAz = 0;
            double d2eri_dAzdAz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_minus_2 = (ix >= 2 ? gx[(gx_offset - 2) * NROOTS + i_root] : 0);
                const double gy_minus_2 = (iy >= 2 ? gy[(gy_offset - 2) * NROOTS + i_root] : 0);
                const double gz_minus_2 = (iz >= 2 ? gz[(gz_offset - 2) * NROOTS + i_root] : 0);
                const double gx_minus_1 = (ix >= 1 ? gx[(gx_offset - 1) * NROOTS + i_root] : 0);
                const double gy_minus_1 = (iy >= 1 ? gy[(gy_offset - 1) * NROOTS + i_root] : 0);
                const double gz_minus_1 = (iz >= 1 ? gz[(gz_offset - 1) * NROOTS + i_root] : 0);
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double gx_1 = gx[(gx_offset + 1) * NROOTS + i_root];
                const double gy_1 = gy[(gy_offset + 1) * NROOTS + i_root];
                const double gz_1 = gz[(gz_offset + 1) * NROOTS + i_root];
                const double gx_2 = gx[(gx_offset + 2) * NROOTS + i_root];
                const double gy_2 = gy[(gy_offset + 2) * NROOTS + i_root];
                const double gz_2 = gz[(gz_offset + 2) * NROOTS + i_root];
                const double dgx_dAx = ix * gx_minus_1 + minus_two_a * gx_1;
                const double dgy_dAy = iy * gy_minus_1 + minus_two_a * gy_1;
                const double dgz_dAz = iz * gz_minus_1 + minus_two_a * gz_1;
                const double d2gx_dAx2 = ix * (ix - 1) * gx_minus_2 + minus_two_a * (2 * ix + 1) * gx_0 + minus_two_a * minus_two_a * gx_2;
                const double d2gy_dAy2 = iy * (iy - 1) * gy_minus_2 + minus_two_a * (2 * iy + 1) * gy_0 + minus_two_a * minus_two_a * gy_2;
                const double d2gz_dAz2 = iz * (iz - 1) * gz_minus_2 + minus_two_a * (2 * iz + 1) * gz_0 + minus_two_a * minus_two_a * gz_2;
                d2eri_dAxdAx += d2gx_dAx2 * gy_0 * gz_0;
                d2eri_dAxdAy += dgx_dAx * dgy_dAy * gz_0;
                d2eri_dAxdAz += dgx_dAx * gy_0 * dgz_dAz;
                d2eri_dAydAy += gx_0 * d2gy_dAy2 * gz_0;
                d2eri_dAydAz += gx_0 * dgy_dAy * dgz_dAz;
                d2eri_dAzdAz += gx_0 * gy_0 * d2gz_dAz2;
            }
            local_output[i + j * n_density_elements_i + 0 * n_density_elements_ij] += d2eri_dAxdAx * prefactor;
            local_output[i + j * n_density_elements_i + 1 * n_density_elements_ij] += d2eri_dAxdAy * prefactor;
            local_output[i + j * n_density_elements_i + 2 * n_density_elements_ij] += d2eri_dAxdAz * prefactor;
            local_output[i + j * n_density_elements_i + 3 * n_density_elements_ij] += d2eri_dAydAy * prefactor;
            local_output[i + j * n_density_elements_i + 4 * n_density_elements_ij] += d2eri_dAydAz * prefactor;
            local_output[i + j * n_density_elements_i + 5 * n_density_elements_ij] += d2eri_dAzdAz * prefactor;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ipip1_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                                    const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                                    const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const double* __restrict__ a_exponents = c_bpcache.a1;

    constexpr int l_sum_max = (NROOTS - 1) * 2 + 1;
    constexpr int l_i_max_density_elements = (l_sum_max + 1) / 2;
    constexpr int l_j_max_density_elements = l_sum_max - l_i_max_density_elements;
    double output_cache[(l_i_max_density_elements + 1) * (l_i_max_density_elements + 2) / 2
                        * (l_j_max_density_elements + 1) * (l_j_max_density_elements + 2) / 2
                        * 6] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[GSIZE_INT3C_1E];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l + 2, j_l, charge_exponent, omega);
            const double minus_two_a = -2.0 * a_exponents[ij];
            GINTwrite_int3c1e_ipip1_charge_contracted<NROOTS>(g, output_cache, minus_two_a, charge, i_l, j_l);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const double d2eri_dAxdAx = output_cache[i + j * n_density_elements_i + 0 * n_density_elements_ij];
            const double d2eri_dAxdAy = output_cache[i + j * n_density_elements_i + 1 * n_density_elements_ij];
            const double d2eri_dAxdAz = output_cache[i + j * n_density_elements_i + 2 * n_density_elements_ij];
            const double d2eri_dAydAy = output_cache[i + j * n_density_elements_i + 3 * n_density_elements_ij];
            const double d2eri_dAydAz = output_cache[i + j * n_density_elements_i + 4 * n_density_elements_ij];
            const double d2eri_dAzdAz = output_cache[i + j * n_density_elements_i + 5 * n_density_elements_ij];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), d2eri_dAxdAx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), d2eri_dAxdAy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), d2eri_dAxdAz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 4 * stride_ij), d2eri_dAydAy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 5 * stride_ij), d2eri_dAydAz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 8 * stride_ij), d2eri_dAzdAz);
        }
    }
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ipvip1_charge_contracted(const double* g, double* local_output, const double minus_two_a, const double minus_two_b, const double prefactor, const int i_l, const int j_l)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1 + 1) * (j_l + 1 + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const int loc_j = c_l_locs[j_l] + j;
            const int loc_i = c_l_locs[i_l] + i;
            const int ix = idx[loc_i];
            const int iy = idy[loc_i];
            const int iz = idz[loc_i];
            const int jx = idx[loc_j];
            const int jy = idy[loc_j];
            const int jz = idz[loc_j];
            const int j_offset = i_l + 1 + 1;

            double d2eri_dAxdBx = 0;
            double d2eri_dAxdBy = 0;
            double d2eri_dAxdBz = 0;
            double d2eri_dAydBx = 0;
            double d2eri_dAydBy = 0;
            double d2eri_dAydBz = 0;
            double d2eri_dAzdBx = 0;
            double d2eri_dAzdBy = 0;
            double d2eri_dAzdBz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_i_minus_1_j_minus_1 = ix * jx * (ix >= 1 && jx >= 1 ? gx[(ix - 1 + (jx - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gy_i_minus_1_j_minus_1 = iy * jy * (iy >= 1 && jy >= 1 ? gy[(iy - 1 + (jy - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gz_i_minus_1_j_minus_1 = iz * jz * (iz >= 1 && jz >= 1 ? gz[(iz - 1 + (jz - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gx_i_minus_1_j_1 = ix * minus_two_b * (ix >= 1 ? gx[(ix - 1 + (jx + 1) * j_offset) * NROOTS + i_root] : 0);
                const double gy_i_minus_1_j_1 = iy * minus_two_b * (iy >= 1 ? gy[(iy - 1 + (jy + 1) * j_offset) * NROOTS + i_root] : 0);
                const double gz_i_minus_1_j_1 = iz * minus_two_b * (iz >= 1 ? gz[(iz - 1 + (jz + 1) * j_offset) * NROOTS + i_root] : 0);
                const double gx_i_1_j_minus_1 = jx * minus_two_a * (jx >= 1 ? gx[(ix + 1 + (jx - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gy_i_1_j_minus_1 = jy * minus_two_a * (jy >= 1 ? gy[(iy + 1 + (jy - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gz_i_1_j_minus_1 = jz * minus_two_a * (jz >= 1 ? gz[(iz + 1 + (jz - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gx_i_1_j_1 = minus_two_a * minus_two_b * gx[(ix + 1 + (jx + 1) * j_offset) * NROOTS + i_root];
                const double gy_i_1_j_1 = minus_two_a * minus_two_b * gy[(iy + 1 + (jy + 1) * j_offset) * NROOTS + i_root];
                const double gz_i_1_j_1 = minus_two_a * minus_two_b * gz[(iz + 1 + (jz + 1) * j_offset) * NROOTS + i_root];
                const double gx_0 = gx[(ix + jx * j_offset) * NROOTS + i_root];
                const double gy_0 = gy[(iy + jy * j_offset) * NROOTS + i_root];
                const double gz_0 = gz[(iz + jz * j_offset) * NROOTS + i_root];
                const double gx_i_1_j_0 = minus_two_a * gx[(ix + 1 + jx * j_offset) * NROOTS + i_root];
                const double gy_i_1_j_0 = minus_two_a * gy[(iy + 1 + jy * j_offset) * NROOTS + i_root];
                const double gz_i_1_j_0 = minus_two_a * gz[(iz + 1 + jz * j_offset) * NROOTS + i_root];
                const double gx_i_minus_1_j_0 = ix * (ix >= 1 ? gx[(ix - 1 + jx * j_offset) * NROOTS + i_root] : 0);
                const double gy_i_minus_1_j_0 = iy * (iy >= 1 ? gy[(iy - 1 + jy * j_offset) * NROOTS + i_root] : 0);
                const double gz_i_minus_1_j_0 = iz * (iz >= 1 ? gz[(iz - 1 + jz * j_offset) * NROOTS + i_root] : 0);
                const double gx_i_0_j_1 = minus_two_b * gx[(ix + (jx + 1) * j_offset) * NROOTS + i_root];
                const double gy_i_0_j_1 = minus_two_b * gy[(iy + (jy + 1) * j_offset) * NROOTS + i_root];
                const double gz_i_0_j_1 = minus_two_b * gz[(iz + (jz + 1) * j_offset) * NROOTS + i_root];
                const double gx_i_0_j_minus_1 = jx * (jx >= 1 ? gx[(ix + (jx - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gy_i_0_j_minus_1 = jy * (jy >= 1 ? gy[(iy + (jy - 1) * j_offset) * NROOTS + i_root] : 0);
                const double gz_i_0_j_minus_1 = jz * (jz >= 1 ? gz[(iz + (jz - 1) * j_offset) * NROOTS + i_root] : 0);

                d2eri_dAxdBx += (gx_i_minus_1_j_minus_1 + gx_i_minus_1_j_1 + gx_i_1_j_minus_1 + gx_i_1_j_1) * gy_0 * gz_0;
                d2eri_dAxdBy += (gx_i_minus_1_j_0 + gx_i_1_j_0) * (gy_i_0_j_minus_1 + gy_i_0_j_1) * gz_0;
                d2eri_dAxdBz += (gx_i_minus_1_j_0 + gx_i_1_j_0) * gy_0 * (gz_i_0_j_minus_1 + gz_i_0_j_1);
                d2eri_dAydBx += (gx_i_0_j_minus_1 + gx_i_0_j_1) * (gy_i_minus_1_j_0 + gy_i_1_j_0) * gz_0;
                d2eri_dAydBy += gx_0 * (gy_i_minus_1_j_minus_1 + gy_i_minus_1_j_1 + gy_i_1_j_minus_1 + gy_i_1_j_1) * gz_0;
                d2eri_dAydBz += gx_0 * (gy_i_minus_1_j_0 + gy_i_1_j_0) * (gz_i_0_j_minus_1 + gz_i_0_j_1);
                d2eri_dAzdBx += (gx_i_0_j_minus_1 + gx_i_0_j_1) * gy_0 * (gz_i_minus_1_j_0 + gz_i_1_j_0);
                d2eri_dAzdBy += gx_0 * (gy_i_0_j_minus_1 + gy_i_0_j_1) * (gz_i_minus_1_j_0 + gz_i_1_j_0);
                d2eri_dAzdBz += gx_0 * gy_0 * (gz_i_minus_1_j_minus_1 + gz_i_minus_1_j_1 + gz_i_1_j_minus_1 + gz_i_1_j_1);
            }
            local_output[i + j * n_density_elements_i + 0 * n_density_elements_ij] += d2eri_dAxdBx * prefactor;
            local_output[i + j * n_density_elements_i + 1 * n_density_elements_ij] += d2eri_dAxdBy * prefactor;
            local_output[i + j * n_density_elements_i + 2 * n_density_elements_ij] += d2eri_dAxdBz * prefactor;
            local_output[i + j * n_density_elements_i + 3 * n_density_elements_ij] += d2eri_dAydBx * prefactor;
            local_output[i + j * n_density_elements_i + 4 * n_density_elements_ij] += d2eri_dAydBy * prefactor;
            local_output[i + j * n_density_elements_i + 5 * n_density_elements_ij] += d2eri_dAydBz * prefactor;
            local_output[i + j * n_density_elements_i + 6 * n_density_elements_ij] += d2eri_dAzdBx * prefactor;
            local_output[i + j * n_density_elements_i + 7 * n_density_elements_ij] += d2eri_dAzdBy * prefactor;
            local_output[i + j * n_density_elements_i + 8 * n_density_elements_ij] += d2eri_dAzdBz * prefactor;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ipvip1_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                                     const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                                     const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const double* __restrict__ a_exponents = c_bpcache.a1;
    const double* __restrict__ b_exponents = c_bpcache.a2;

    constexpr int l_sum_max = (NROOTS - 1) * 2 + 1;
    constexpr int l_i_max_density_elements = (l_sum_max + 1) / 2;
    constexpr int l_j_max_density_elements = l_sum_max - l_i_max_density_elements;
    double output_cache[(l_i_max_density_elements + 1) * (l_i_max_density_elements + 2) / 2
                        * (l_j_max_density_elements + 1) * (l_j_max_density_elements + 2) / 2
                        * 9] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[GSIZE_INT3C_1E];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l + 1, j_l + 1, charge_exponent, omega);
            const double minus_two_a = -2.0 * a_exponents[ij];
            const double minus_two_b = -2.0 * b_exponents[ij];
            GINTwrite_int3c1e_ipvip1_charge_contracted<NROOTS>(g, output_cache, minus_two_a, minus_two_b, charge, i_l, j_l);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const double d2eri_dAxdBx = output_cache[i + j * n_density_elements_i + 0 * n_density_elements_ij];
            const double d2eri_dAxdBy = output_cache[i + j * n_density_elements_i + 1 * n_density_elements_ij];
            const double d2eri_dAxdBz = output_cache[i + j * n_density_elements_i + 2 * n_density_elements_ij];
            const double d2eri_dAydBx = output_cache[i + j * n_density_elements_i + 3 * n_density_elements_ij];
            const double d2eri_dAydBy = output_cache[i + j * n_density_elements_i + 4 * n_density_elements_ij];
            const double d2eri_dAydBz = output_cache[i + j * n_density_elements_i + 5 * n_density_elements_ij];
            const double d2eri_dAzdBx = output_cache[i + j * n_density_elements_i + 6 * n_density_elements_ij];
            const double d2eri_dAzdBy = output_cache[i + j * n_density_elements_i + 7 * n_density_elements_ij];
            const double d2eri_dAzdBz = output_cache[i + j * n_density_elements_i + 8 * n_density_elements_ij];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), d2eri_dAxdBx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), d2eri_dAxdBy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), d2eri_dAxdBz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 3 * stride_ij), d2eri_dAydBx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 4 * stride_ij), d2eri_dAydBy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 5 * stride_ij), d2eri_dAydBz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 6 * stride_ij), d2eri_dAzdBx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 7 * stride_ij), d2eri_dAzdBy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 8 * stride_ij), d2eri_dAzdBz);
        }
    }
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ip1ip2_charge_contracted(const double* g, double* local_output, const double minus_two_a, const double* u2, const double* AC, const double prefactor, const int i_l, const int j_l)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 2 + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const double ACx = AC[0];
    const double ACy = AC[1];
    const double ACz = AC[2];

    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const int loc_j = c_l_locs[j_l] + j;
            const int loc_i = c_l_locs[i_l] + i;
            const int ix = idx[loc_i];
            const int iy = idy[loc_i];
            const int iz = idz[loc_i];
            const int jx = idx[loc_j];
            const int jy = idy[loc_j];
            const int jz = idz[loc_j];
            const int gx_offset = ix + jx * (i_l + 2 + 1);
            const int gy_offset = iy + jy * (i_l + 2 + 1);
            const int gz_offset = iz + jz * (i_l + 2 + 1);

            double d2eri_dAxdCx = 0;
            double d2eri_dAxdCy = 0;
            double d2eri_dAxdCz = 0;
            double d2eri_dAydCx = 0;
            double d2eri_dAydCy = 0;
            double d2eri_dAydCz = 0;
            double d2eri_dAzdCx = 0;
            double d2eri_dAzdCy = 0;
            double d2eri_dAzdCz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_minus_1 = (ix >= 1 ? gx[(gx_offset - 1) * NROOTS + i_root] : 0);
                const double gy_minus_1 = (iy >= 1 ? gy[(gy_offset - 1) * NROOTS + i_root] : 0);
                const double gz_minus_1 = (iz >= 1 ? gz[(gz_offset - 1) * NROOTS + i_root] : 0);
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double gx_1 = gx[(gx_offset + 1) * NROOTS + i_root];
                const double gy_1 = gy[(gy_offset + 1) * NROOTS + i_root];
                const double gz_1 = gz[(gz_offset + 1) * NROOTS + i_root];
                const double gx_2 = gx[(gx_offset + 2) * NROOTS + i_root];
                const double gy_2 = gy[(gy_offset + 2) * NROOTS + i_root];
                const double gz_2 = gz[(gz_offset + 2) * NROOTS + i_root];

                const double two_u2 = 2.0 * u2[i_root];
                const double dgx_dAx = ix * gx_minus_1 + minus_two_a * gx_1;
                const double dgy_dAy = iy * gy_minus_1 + minus_two_a * gy_1;
                const double dgz_dAz = iz * gz_minus_1 + minus_two_a * gz_1;
                const double dgx_dCx = two_u2 * (ACx * gx_0 + gx_1);
                const double dgy_dCy = two_u2 * (ACy * gy_0 + gy_1);
                const double dgz_dCz = two_u2 * (ACz * gz_0 + gz_1);
                const double d2gx_dAxdCx = two_u2 * (ix * ACx * gx_minus_1 + ix * gx_0 + minus_two_a * ACx * gx_1 + minus_two_a * gx_2);
                const double d2gy_dAydCy = two_u2 * (iy * ACy * gy_minus_1 + iy * gy_0 + minus_two_a * ACy * gy_1 + minus_two_a * gy_2);
                const double d2gz_dAzdCz = two_u2 * (iz * ACz * gz_minus_1 + iz * gz_0 + minus_two_a * ACz * gz_1 + minus_two_a * gz_2);

                d2eri_dAxdCx += - d2gx_dAxdCx * gy_0 * gz_0;
                d2eri_dAxdCy += - dgx_dAx * dgy_dCy * gz_0;
                d2eri_dAxdCz += - dgx_dAx * gy_0 * dgz_dCz;
                d2eri_dAydCx += - dgx_dCx * dgy_dAy * gz_0;
                d2eri_dAydCy += - gx_0 * d2gy_dAydCy * gz_0;
                d2eri_dAydCz += - gx_0 * dgy_dAy * dgz_dCz;
                d2eri_dAzdCx += - dgx_dCx * gy_0 * dgz_dAz;
                d2eri_dAzdCy += - gx_0 * dgy_dCy * dgz_dAz;
                d2eri_dAzdCz += - gx_0 * gy_0 * d2gz_dAzdCz;
            }
            local_output[i + j * n_density_elements_i + 0 * n_density_elements_ij] += d2eri_dAxdCx * prefactor;
            local_output[i + j * n_density_elements_i + 1 * n_density_elements_ij] += d2eri_dAxdCy * prefactor;
            local_output[i + j * n_density_elements_i + 2 * n_density_elements_ij] += d2eri_dAxdCz * prefactor;
            local_output[i + j * n_density_elements_i + 3 * n_density_elements_ij] += d2eri_dAydCx * prefactor;
            local_output[i + j * n_density_elements_i + 4 * n_density_elements_ij] += d2eri_dAydCy * prefactor;
            local_output[i + j * n_density_elements_i + 5 * n_density_elements_ij] += d2eri_dAydCz * prefactor;
            local_output[i + j * n_density_elements_i + 6 * n_density_elements_ij] += d2eri_dAzdCx * prefactor;
            local_output[i + j * n_density_elements_i + 7 * n_density_elements_ij] += d2eri_dAzdCy * prefactor;
            local_output[i + j * n_density_elements_i + 8 * n_density_elements_ij] += d2eri_dAzdCz * prefactor;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ip1ip2_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                                     const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                                     const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const double* __restrict__ a_exponents = c_bpcache.a1;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    constexpr int l_sum_max = (NROOTS - 1) * 2 + 1;
    constexpr int l_i_max_density_elements = (l_sum_max + 1) / 2;
    constexpr int l_j_max_density_elements = l_sum_max - l_i_max_density_elements;
    double output_cache[(l_i_max_density_elements + 1) * (l_i_max_density_elements + 2) / 2
                        * (l_j_max_density_elements + 1) * (l_j_max_density_elements + 2) / 2
                        * 9] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double Cx = grid_point[0];
        const double Cy = grid_point[1];
        const double Cz = grid_point[2];
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        const double AC[3] { Ax - Cx, Ay - Cy, Az - Cz };

        double g[GSIZE_INT3C_1E];
        double u2[NROOTS];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e_save_u2<NROOTS>(g, u2, grid_point, ish, jsh, ij, i_l + 2, j_l, charge_exponent, omega);
            const double minus_two_a = -2.0 * a_exponents[ij];
            GINTwrite_int3c1e_ip1ip2_charge_contracted<NROOTS>(g, output_cache, minus_two_a, u2, AC, charge, i_l, j_l);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
    const int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
    for (int j = 0; j < n_density_elements_j; j++) {
        for (int i = 0; i < n_density_elements_i; i++) {
            const double d2eri_dAxdCx = output_cache[i + j * n_density_elements_i + 0 * n_density_elements_ij];
            const double d2eri_dAxdCy = output_cache[i + j * n_density_elements_i + 1 * n_density_elements_ij];
            const double d2eri_dAxdCz = output_cache[i + j * n_density_elements_i + 2 * n_density_elements_ij];
            const double d2eri_dAydCx = output_cache[i + j * n_density_elements_i + 3 * n_density_elements_ij];
            const double d2eri_dAydCy = output_cache[i + j * n_density_elements_i + 4 * n_density_elements_ij];
            const double d2eri_dAydCz = output_cache[i + j * n_density_elements_i + 5 * n_density_elements_ij];
            const double d2eri_dAzdCx = output_cache[i + j * n_density_elements_i + 6 * n_density_elements_ij];
            const double d2eri_dAzdCy = output_cache[i + j * n_density_elements_i + 7 * n_density_elements_ij];
            const double d2eri_dAzdCz = output_cache[i + j * n_density_elements_i + 8 * n_density_elements_ij];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), d2eri_dAxdCx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), d2eri_dAxdCy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), d2eri_dAxdCz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 3 * stride_ij), d2eri_dAydCx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 4 * stride_ij), d2eri_dAydCy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 5 * stride_ij), d2eri_dAydCz);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 6 * stride_ij), d2eri_dAzdCx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 7 * stride_ij), d2eri_dAzdCy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 8 * stride_ij), d2eri_dAzdCz);
        }
    }
}

template <int L_SUM>
__global__
static void GINTfill_int3c1e_ipip2_density_contracted_kernel_general(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                                     const BasisProdOffsets offsets, const int nprim_ij,
                                                                     const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int NROOTS = (L_SUM + 2) / 2 + 1;

    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_grid >= ngrids) {
        return;
    }

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double d2eri_dCxdCx_pair_sum = 0.0;
    double d2eri_dCxdCy_pair_sum = 0.0;
    double d2eri_dCxdCz_pair_sum = 0.0;
    double d2eri_dCydCy_pair_sum = 0.0;
    double d2eri_dCydCz_pair_sum = 0.0;
    double d2eri_dCzdCz_pair_sum = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {

        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
        const int* bas_pair2bra = c_bpcache.bas_pair2bra;
        // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
        const int ish = bas_pair2bra[bas_ij];
        // const int jsh = bas_pair2ket[bas_ij];
        const int nbas = c_bpcache.nbas;
        const double* __restrict__ bas_x = c_bpcache.bas_coords;
        const double* __restrict__ bas_y = bas_x + nbas;
        const double* __restrict__ bas_z = bas_y + nbas;
        const double Ax = bas_x[ish];
        const double Ay = bas_y[ish];
        const double Az = bas_z[ish];

        const double ACx = Ax - Cx;
        const double ACy = Ay - Cy;
        const double ACz = Az - Cz;

        double D_hermite[(L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6];
#pragma unroll
        for (int i_t = 0; i_t < (L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6; i_t++) {
            D_hermite[i_t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + i_t * hermite_density_offsets.n_pair_of_angular_pair];
        }

        double d2eri_dCxdCx = 0.0;
        double d2eri_dCxdCy = 0.0;
        double d2eri_dCxdCz = 0.0;
        double d2eri_dCydCy = 0.0;
        double d2eri_dCydCz = 0.0;
        double d2eri_dCzdCz = 0.0;
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            double g[NROOTS * (L_SUM + 2 + 1) * 3];
            double u2[NROOTS];
            GINT_g1e_without_hrr_save_u2<L_SUM + 2>(g, u2, Cx, Cy, Cz, ish, ij, charge_exponent, omega);

            const double* __restrict__ gx = g;
            const double* __restrict__ gy = g + NROOTS * (L_SUM + 2 + 1);
            const double* __restrict__ gz = g + NROOTS * (L_SUM + 2 + 1) * 2;

#pragma unroll
            for (int i_x = 0, i_t = 0; i_x <= L_SUM; i_x++) {
#pragma unroll
                for (int i_y = 0; i_x + i_y <= L_SUM; i_y++) {
#pragma unroll
                    for (int i_z = 0; i_x + i_y + i_z <= L_SUM; i_z++, i_t++) {
                        double d2eri_dCxdCx_per_hermite = 0.0;
                        double d2eri_dCxdCy_per_hermite = 0.0;
                        double d2eri_dCxdCz_per_hermite = 0.0;
                        double d2eri_dCydCy_per_hermite = 0.0;
                        double d2eri_dCydCz_per_hermite = 0.0;
                        double d2eri_dCzdCz_per_hermite = 0.0;
#pragma unroll
                        for (int i_root = 0; i_root < NROOTS; i_root++) {
                            const double gx_0 = gx[i_root + NROOTS * i_x];
                            const double gy_0 = gy[i_root + NROOTS * i_y];
                            const double gz_0 = gz[i_root + NROOTS * i_z];
                            const double gx_1 = gx[i_root + NROOTS * (i_x + 1)];
                            const double gy_1 = gy[i_root + NROOTS * (i_y + 1)];
                            const double gz_1 = gz[i_root + NROOTS * (i_z + 1)];
                            const double gx_2 = gx[i_root + NROOTS * (i_x + 2)];
                            const double gy_2 = gy[i_root + NROOTS * (i_y + 2)];
                            const double gz_2 = gz[i_root + NROOTS * (i_z + 2)];
                            const double two_u2 = 2.0 * u2[i_root];
                            const double dgx_dCx = two_u2 * (gx_1 + ACx * gx_0);
                            const double dgy_dCy = two_u2 * (gy_1 + ACy * gy_0);
                            const double dgz_dCz = two_u2 * (gz_1 + ACz * gz_0);
                            const double d2gx_dCx2 = two_u2 * (-gx_0 + two_u2 * (gx_2 + ACx * gx_1 * 2 + ACx * ACx * gx_0));
                            const double d2gy_dCy2 = two_u2 * (-gy_0 + two_u2 * (gy_2 + ACy * gy_1 * 2 + ACy * ACy * gy_0));
                            const double d2gz_dCz2 = two_u2 * (-gz_0 + two_u2 * (gz_2 + ACz * gz_1 * 2 + ACz * ACz * gz_0));
                            d2eri_dCxdCx_per_hermite += d2gx_dCx2 * gy_0 * gz_0;
                            d2eri_dCxdCy_per_hermite += dgx_dCx * dgy_dCy * gz_0;
                            d2eri_dCxdCz_per_hermite += dgx_dCx * gy_0 * dgz_dCz;
                            d2eri_dCydCy_per_hermite += gx_0 * d2gy_dCy2 * gz_0;
                            d2eri_dCydCz_per_hermite += gx_0 * dgy_dCy * dgz_dCz;
                            d2eri_dCzdCz_per_hermite += gx_0 * gy_0 * d2gz_dCz2;
                        }
                        const double D_t = D_hermite[i_t];
                        d2eri_dCxdCx += d2eri_dCxdCx_per_hermite * D_t;
                        d2eri_dCxdCy += d2eri_dCxdCy_per_hermite * D_t;
                        d2eri_dCxdCz += d2eri_dCxdCz_per_hermite * D_t;
                        d2eri_dCydCy += d2eri_dCydCy_per_hermite * D_t;
                        d2eri_dCydCz += d2eri_dCydCz_per_hermite * D_t;
                        d2eri_dCzdCz += d2eri_dCzdCz_per_hermite * D_t;
                    }
                }
            }
        }
        d2eri_dCxdCx_pair_sum += d2eri_dCxdCx;
        d2eri_dCxdCy_pair_sum += d2eri_dCxdCy;
        d2eri_dCxdCz_pair_sum += d2eri_dCxdCz;
        d2eri_dCydCy_pair_sum += d2eri_dCydCy;
        d2eri_dCydCz_pair_sum += d2eri_dCydCz;
        d2eri_dCzdCz_pair_sum += d2eri_dCzdCz;
    }
    atomicAdd(output + task_grid + ngrids * 0, d2eri_dCxdCx_pair_sum);
    atomicAdd(output + task_grid + ngrids * 1, d2eri_dCxdCy_pair_sum);
    atomicAdd(output + task_grid + ngrids * 2, d2eri_dCxdCz_pair_sum);
    atomicAdd(output + task_grid + ngrids * 4, d2eri_dCydCy_pair_sum);
    atomicAdd(output + task_grid + ngrids * 5, d2eri_dCydCz_pair_sum);
    atomicAdd(output + task_grid + ngrids * 8, d2eri_dCzdCz_pair_sum);
}
