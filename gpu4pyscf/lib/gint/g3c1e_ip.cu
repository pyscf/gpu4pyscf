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
static void GINTwrite_int3c1e_ip(const double* g, double* output, const double minus_two_a, const double* u2, const double* AC, const int ish, const int jsh, const int i_grid,
                                 const int i_l, const int j_l, const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j, const int ngrids)
{
    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish  ] - ao_offsets_i;
    const int i1 = ao_loc[ish+1] - ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - ao_offsets_j;

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1 + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    for (int j = j0; j < j1; j++) {
        for (int i = i0; i < i1; i++) {
            const int loc_j = c_l_locs[j_l] + (j-j0);
            const int loc_i = c_l_locs[i_l] + (i-i0);
            const int ix = idx[loc_i];
            const int iy = idy[loc_i];
            const int iz = idz[loc_i];
            const int jx = idx[loc_j];
            const int jy = idy[loc_j];
            const int jz = idz[loc_j];
            const int gx_offset = ix + jx * (i_l + 1 + 1);
            const int gy_offset = iy + jy * (i_l + 1 + 1);
            const int gz_offset = iz + jz * (i_l + 1 + 1);

            double deri_dAx = 0;
            double deri_dAy = 0;
            double deri_dAz = 0;
            double deri_dCx = 0;
            double deri_dCy = 0;
            double deri_dCz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double gx_1 = gx[(gx_offset + 1) * NROOTS + i_root];
                const double gy_1 = gy[(gy_offset + 1) * NROOTS + i_root];
                const double gz_1 = gz[(gz_offset + 1) * NROOTS + i_root];
                const double dgx_dAx = (ix > 0 ? ix * gx[(gx_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gx_1;
                const double dgy_dAy = (iy > 0 ? iy * gy[(gy_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gy_1;
                const double dgz_dAz = (iz > 0 ? iz * gz[(gz_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gz_1;
                deri_dAx += dgx_dAx * gy_0 * gz_0;
                deri_dAy += gx_0 * dgy_dAy * gz_0;
                deri_dAz += gx_0 * gy_0 * dgz_dAz;
                const double minus_two_u2 = -2.0 * u2[i_root];
                const double dgx_dCx = minus_two_u2 * (gx_1 + AC[0] * gx_0);
                const double dgy_dCy = minus_two_u2 * (gy_1 + AC[1] * gy_0);
                const double dgz_dCz = minus_two_u2 * (gz_1 + AC[2] * gz_0);
                deri_dCx += dgx_dCx * gy_0 * gz_0;
                deri_dCy += gx_0 * dgy_dCy * gz_0;
                deri_dCz += gx_0 * gy_0 * dgz_dCz;
            }
            output[i + j * stride_j + i_grid * stride_ij + 0 * stride_ij * ngrids] += deri_dAx;
            output[i + j * stride_j + i_grid * stride_ij + 1 * stride_ij * ngrids] += deri_dAy;
            output[i + j * stride_j + i_grid * stride_ij + 2 * stride_ij * ngrids] += deri_dAz;
            output[i + j * stride_j + i_grid * stride_ij + 3 * stride_ij * ngrids] += deri_dCx;
            output[i + j * stride_j + i_grid * stride_ij + 4 * stride_ij * ngrids] += deri_dCy;
            output[i + j * stride_j + i_grid * stride_ij + 5 * stride_ij * ngrids] += deri_dCz;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ip_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                               const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                               const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
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

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    const double AC[3] { Ax - Cx, Ay - Cy, Az - Cz };

    double g[GSIZE_INT3C_1E];
    double u2[NROOTS];

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        GINT_g1e_save_u2<NROOTS>(g, u2, grid_point, ish, jsh, ij, i_l + 1, j_l, charge_exponent, omega);
        const double minus_two_a = -2.0 * a_exponents[ij];
        GINTwrite_int3c1e_ip<NROOTS>(g, output, minus_two_a, u2, AC, ish, jsh, task_grid, i_l, j_l, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, ngrids);
    }
}

template <int LI, int LJ>
__device__
static void GINTwrite_int3c1e_ip1_charge_contracted(const double* g, double* local_output, const double minus_two_a, const double prefactor)
{
    constexpr int NROOTS = (LI + LJ + 1) / 2 + 1;

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (LI + 1 + 1) * (LJ + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    constexpr int n_density_elements_i = (LI + 1) * (LI + 2) / 2;
    constexpr int n_density_elements_j = (LJ + 1) * (LJ + 2) / 2;
    constexpr int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
#pragma unroll
    for (int j = 0; j < n_density_elements_j; j++) {
#pragma unroll
        for (int i = 0; i < n_density_elements_i; i++) {
            const int loc_j = c_l_locs[LJ] + j;
            const int loc_i = c_l_locs[LI] + i;
            const int ix = idx[loc_i];
            const int iy = idy[loc_i];
            const int iz = idz[loc_i];
            const int jx = idx[loc_j];
            const int jy = idy[loc_j];
            const int jz = idz[loc_j];
            const int gx_offset = ix + jx * (LI + 1 + 1);
            const int gy_offset = iy + jy * (LI + 1 + 1);
            const int gz_offset = iz + jz * (LI + 1 + 1);

            double deri_dAx = 0;
            double deri_dAy = 0;
            double deri_dAz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double dgx_dAx = (ix > 0 ? ix * gx[(gx_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gx[(gx_offset + 1) * NROOTS + i_root];
                const double dgy_dAy = (iy > 0 ? iy * gy[(gy_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gy[(gy_offset + 1) * NROOTS + i_root];
                const double dgz_dAz = (iz > 0 ? iz * gz[(gz_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gz[(gz_offset + 1) * NROOTS + i_root];
                deri_dAx += dgx_dAx * gy_0 * gz_0;
                deri_dAy += gx_0 * dgy_dAy * gz_0;
                deri_dAz += gx_0 * gy_0 * dgz_dAz;
            }
            local_output[i + j * n_density_elements_i + 0 * n_density_elements_ij] += deri_dAx * prefactor;
            local_output[i + j * n_density_elements_i + 1 * n_density_elements_ij] += deri_dAy * prefactor;
            local_output[i + j * n_density_elements_i + 2 * n_density_elements_ij] += deri_dAz * prefactor;
        }
    }
}

template <int LI, int LJ>
__global__
static void GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                                  const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                                  const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int L_SUM = LI + LJ;
    constexpr int NROOTS = (L_SUM + 1) / 2 + 1;

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

    constexpr int n_density_elements_i = (LI + 1) * (LI + 2) / 2;
    constexpr int n_density_elements_j = (LJ + 1) * (LJ + 2) / 2;
    double output_cache[n_density_elements_i * n_density_elements_j * 3] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[3 * NROOTS * (LI + 1 + 1) * (LJ + 1)];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, LI + 1, LJ, charge_exponent, omega);
            const double minus_two_a = -2.0 * a_exponents[ij];
            GINTwrite_int3c1e_ip1_charge_contracted<LI, LJ>(g, output_cache, minus_two_a, charge);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    constexpr int n_density_elements_ij = n_density_elements_i * n_density_elements_j;
#pragma unroll
    for (int j = 0; j < n_density_elements_j; j++) {
#pragma unroll
        for (int i = 0; i < n_density_elements_i; i++) {
            const double deri_dAx = output_cache[i + j * n_density_elements_i + 0 * n_density_elements_ij];
            const double deri_dAy = output_cache[i + j * n_density_elements_i + 1 * n_density_elements_ij];
            const double deri_dAz = output_cache[i + j * n_density_elements_i + 2 * n_density_elements_ij];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), deri_dAx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), deri_dAy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), deri_dAz);
        }
    }
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ip1_charge_contracted(const double* g, double* local_output, const double minus_two_a, const double prefactor, const int i_l, const int j_l)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1 + 1) * (j_l + 1);
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
            const int gx_offset = ix + jx * (i_l + 1 + 1);
            const int gy_offset = iy + jy * (i_l + 1 + 1);
            const int gz_offset = iz + jz * (i_l + 1 + 1);

            double deri_dAx = 0;
            double deri_dAy = 0;
            double deri_dAz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double dgx_dAx = (ix > 0 ? ix * gx[(gx_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gx[(gx_offset + 1) * NROOTS + i_root];
                const double dgy_dAy = (iy > 0 ? iy * gy[(gy_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gy[(gy_offset + 1) * NROOTS + i_root];
                const double dgz_dAz = (iz > 0 ? iz * gz[(gz_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gz[(gz_offset + 1) * NROOTS + i_root];
                deri_dAx += dgx_dAx * gy_0 * gz_0;
                deri_dAy += gx_0 * dgy_dAy * gz_0;
                deri_dAz += gx_0 * gy_0 * dgz_dAz;
            }
            local_output[i + j * n_density_elements_i + 0 * n_density_elements_ij] += deri_dAx * prefactor;
            local_output[i + j * n_density_elements_i + 1 * n_density_elements_ij] += deri_dAy * prefactor;
            local_output[i + j * n_density_elements_i + 2 * n_density_elements_ij] += deri_dAz * prefactor;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ip1_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
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
                        * 3] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[GSIZE_INT3C_1E];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l + 1, j_l, charge_exponent, omega);
            const double minus_two_a = -2.0 * a_exponents[ij];
            GINTwrite_int3c1e_ip1_charge_contracted<NROOTS>(g, output_cache, minus_two_a, charge, i_l, j_l);
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
            const double deri_dAx = output_cache[i + j * n_density_elements_i + 0 * n_density_elements_ij];
            const double deri_dAy = output_cache[i + j * n_density_elements_i + 1 * n_density_elements_ij];
            const double deri_dAz = output_cache[i + j * n_density_elements_i + 2 * n_density_elements_ij];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), deri_dAx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), deri_dAy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), deri_dAz);
        }
    }
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ip1_density_contracted(const double* g, double* output, const double minus_two_a, const double* density, const int* aoslice, const int nao,
                                                     const int ish, const int jsh, const int i_grid, const int i_l, const int j_l, const int ngrids)
{
    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];

    const int i_atom = aoslice[ish];

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1 + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
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
            const int gx_offset = ix + jx * (i_l + 1 + 1);
            const int gy_offset = iy + jy * (i_l + 1 + 1);
            const int gz_offset = iz + jz * (i_l + 1 + 1);

            double deri_dAx = 0;
            double deri_dAy = 0;
            double deri_dAz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double dgx_dAx = (ix > 0 ? ix * gx[(gx_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gx[(gx_offset + 1) * NROOTS + i_root];
                const double dgy_dAy = (iy > 0 ? iy * gy[(gy_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gy[(gy_offset + 1) * NROOTS + i_root];
                const double dgz_dAz = (iz > 0 ? iz * gz[(gz_offset - 1) * NROOTS + i_root] : 0) + minus_two_a * gz[(gz_offset + 1) * NROOTS + i_root];
                deri_dAx += dgx_dAx * gy_0 * gz_0;
                deri_dAy += gx_0 * dgy_dAy * gz_0;
                deri_dAz += gx_0 * gy_0 * dgz_dAz;
            }
            const double Dij = density[(i + i0) + (j + j0) * nao];
            deri_dAx *= Dij;
            deri_dAy *= Dij;
            deri_dAz *= Dij;
            atomicAdd(output + (i_grid + ngrids * (i_atom * 3 + 0)), deri_dAx);
            atomicAdd(output + (i_grid + ngrids * (i_atom * 3 + 1)), deri_dAy);
            atomicAdd(output + (i_grid + ngrids * (i_atom * 3 + 2)), deri_dAz);
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ip1_density_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                                   const double* density, const int* aoslice, const int nao,
                                                                   const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const double* __restrict__ a_exponents = c_bpcache.a1;

    const double* grid_point = grid_points + task_grid * 3;
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double g[GSIZE_INT3C_1E];

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l + 1, j_l, charge_exponent, omega);
        const double minus_two_a = -2.0 * a_exponents[ij];
        GINTwrite_int3c1e_ip1_density_contracted<NROOTS>(g, output, minus_two_a, density, aoslice, nao, ish, jsh, task_grid, i_l, j_l, ngrids);
    }
}

template <int L_SUM>
__global__
static void GINTfill_int3c1e_ip2_density_contracted_kernel_general(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                                   const BasisProdOffsets offsets, const int nprim_ij,
                                                                   const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int NROOTS = (L_SUM + 1) / 2 + 1;

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

    double deri_dCx_pair_sum = 0.0;
    double deri_dCy_pair_sum = 0.0;
    double deri_dCz_pair_sum = 0.0;
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

        double D_hermite[(L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6];
#pragma unroll
        for (int i_t = 0; i_t < (L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6; i_t++) {
            D_hermite[i_t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + i_t * hermite_density_offsets.n_pair_of_angular_pair];
        }

        double deri_dCx_per_pair = 0.0;
        double deri_dCy_per_pair = 0.0;
        double deri_dCz_per_pair = 0.0;
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            double g[NROOTS * (L_SUM + 1 + 1) * 3];
            double u2[NROOTS];
            GINT_g1e_without_hrr_save_u2<L_SUM + 1>(g, u2, Cx, Cy, Cz, ish, ij, charge_exponent, omega);

            const double* __restrict__ gx = g;
            const double* __restrict__ gy = g + NROOTS * (L_SUM + 1 + 1);
            const double* __restrict__ gz = g + NROOTS * (L_SUM + 1 + 1) * 2;

#pragma unroll
            for (int i_x = 0, i_t = 0; i_x <= L_SUM; i_x++) {
#pragma unroll
                for (int i_y = 0; i_x + i_y <= L_SUM; i_y++) {
#pragma unroll
                    for (int i_z = 0; i_x + i_y + i_z <= L_SUM; i_z++, i_t++) {
                        double deri_dCx_per_hermite = 0.0;
                        double deri_dCy_per_hermite = 0.0;
                        double deri_dCz_per_hermite = 0.0;
#pragma unroll
                        for (int i_root = 0; i_root < NROOTS; i_root++) {
                            const double gx_0 = gx[i_root + NROOTS * i_x];
                            const double gy_0 = gy[i_root + NROOTS * i_y];
                            const double gz_0 = gz[i_root + NROOTS * i_z];
                            const double gx_1 = gx[i_root + NROOTS * (i_x + 1)];
                            const double gy_1 = gy[i_root + NROOTS * (i_y + 1)];
                            const double gz_1 = gz[i_root + NROOTS * (i_z + 1)];
                            const double minus_two_u2 = -2.0 * u2[i_root];
                            const double dgx_dCx = minus_two_u2 * (gx_1 + (Ax - Cx) * gx_0);
                            const double dgy_dCy = minus_two_u2 * (gy_1 + (Ay - Cy) * gy_0);
                            const double dgz_dCz = minus_two_u2 * (gz_1 + (Az - Cz) * gz_0);
                            deri_dCx_per_hermite += dgx_dCx * gy_0 * gz_0;
                            deri_dCy_per_hermite += gx_0 * dgy_dCy * gz_0;
                            deri_dCz_per_hermite += gx_0 * gy_0 * dgz_dCz;
                        }
                        const double D_t = D_hermite[i_t];
                        deri_dCx_per_pair += deri_dCx_per_hermite * D_t;
                        deri_dCy_per_pair += deri_dCy_per_hermite * D_t;
                        deri_dCz_per_pair += deri_dCz_per_hermite * D_t;
                    }
                }
            }
        }
        deri_dCx_pair_sum += deri_dCx_per_pair;
        deri_dCy_pair_sum += deri_dCy_per_pair;
        deri_dCz_pair_sum += deri_dCz_per_pair;
    }
    atomicAdd(output + task_grid + ngrids * 0, deri_dCx_pair_sum);
    atomicAdd(output + task_grid + ngrids * 1, deri_dCy_pair_sum);
    atomicAdd(output + task_grid + ngrids * 2, deri_dCz_pair_sum);
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_ip2_charge_contracted(const double* g, double* output, const double minus_two_a, const double* u2, const double* AC, const double prefactor,
                                                    const int ish, const int jsh, const int i_grid, const int i_l, const int j_l,
                                                    const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j, const int* gridslice, const int ngrids)
{
    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;

    const int i_atom = gridslice[i_grid];

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1 + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const int n_density_elements_i = (i_l + 1) * (i_l + 2) / 2;
    const int n_density_elements_j = (j_l + 1) * (j_l + 2) / 2;
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
            const int gx_offset = ix + jx * (i_l + 1 + 1);
            const int gy_offset = iy + jy * (i_l + 1 + 1);
            const int gz_offset = iz + jz * (i_l + 1 + 1);

            double deri_dCx = 0;
            double deri_dCy = 0;
            double deri_dCz = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                const double gx_0 = gx[gx_offset * NROOTS + i_root];
                const double gy_0 = gy[gy_offset * NROOTS + i_root];
                const double gz_0 = gz[gz_offset * NROOTS + i_root];
                const double gx_1 = gx[(gx_offset + 1) * NROOTS + i_root];
                const double gy_1 = gy[(gy_offset + 1) * NROOTS + i_root];
                const double gz_1 = gz[(gz_offset + 1) * NROOTS + i_root];
                const double minus_two_u2 = -2.0 * u2[i_root];
                const double dgx_dCx = minus_two_u2 * (gx_1 + AC[0] * gx_0);
                const double dgy_dCy = minus_two_u2 * (gy_1 + AC[1] * gy_0);
                const double dgz_dCz = minus_two_u2 * (gz_1 + AC[2] * gz_0);
                deri_dCx += dgx_dCx * gy_0 * gz_0;
                deri_dCy += gx_0 * dgy_dCy * gz_0;
                deri_dCz += gx_0 * gy_0 * dgz_dCz;
            }
            deri_dCx *= prefactor;
            deri_dCy *= prefactor;
            deri_dCz *= prefactor;

            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij + i_atom * 3 * stride_ij), deri_dCx);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij + i_atom * 3 * stride_ij), deri_dCy);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij + i_atom * 3 * stride_ij), deri_dCz);
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_ip2_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                                  const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j, const int* gridslice,
                                                                  const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
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
        GINT_g1e_save_u2<NROOTS>(g, u2, grid_point, ish, jsh, ij, i_l + 1, j_l, charge_exponent, omega);
        const double minus_two_a = -2.0 * a_exponents[ij];
        GINTwrite_int3c1e_ip2_charge_contracted<NROOTS>(g, output, minus_two_a, u2, AC, charge, ish, jsh, task_grid, i_l, j_l, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, ngrids);
    }
}
