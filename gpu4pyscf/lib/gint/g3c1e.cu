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
static void GINTwrite_int3c1e(const double* g, double* output, const int ish, const int jsh, const int i_grid,
                              const int i_l, const int j_l, const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j)
{
    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish  ] - ao_offsets_i;
    const int i1 = ao_loc[ish+1] - ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - ao_offsets_j;

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    for (int j = j0; j < j1; j++) {
        for (int i = i0; i < i1; i++) {
            const int loc_j = c_l_locs[j_l] + (j-j0);
            const int loc_i = c_l_locs[i_l] + (i-i0);

            int ix = idx[loc_i] + idx[loc_j] * (i_l + 1);
            int iy = idy[loc_i] + idy[loc_j] * (i_l + 1);
            int iz = idz[loc_i] + idz[loc_j] * (i_l + 1);

            ix = ix * NROOTS;
            iy = iy * NROOTS;
            iz = iz * NROOTS;

            double eri = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                eri += gx[ix + i_root] * gy[iy + i_root] * gz[iz + i_root];
            }
            output[i + j * stride_j + i_grid * stride_ij] += eri;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
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

    const double* grid_point = grid_points + task_grid * 3;
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double g[GSIZE_INT3C_1E];

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l, j_l, charge_exponent, omega);
        GINTwrite_int3c1e<NROOTS>(g, output, ish, jsh, task_grid, i_l, j_l, stride_j, stride_ij, ao_offsets_i, ao_offsets_j);
    }
}

template <int LI, int LJ>
__device__
static void GINTwrite_int3c1e_charge_contracted(const double* g, double* local_output, const double prefactor)
{
    constexpr int NROOTS = (LI + LJ) / 2 + 1;

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (LI + 1) * (LJ + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    constexpr int n_density_elements_i = (LI + 1) * (LI + 2) / 2;
    constexpr int n_density_elements_j = (LJ + 1) * (LJ + 2) / 2;
#pragma unroll
    for (int j = 0; j < n_density_elements_j; j++) {
#pragma unroll
        for (int i = 0; i < n_density_elements_i; i++) {
            const int loc_j = c_l_locs[LJ] + j;
            const int loc_i = c_l_locs[LI] + i;

            const int ix = (idx[loc_i] + idx[loc_j] * (LI + 1)) * NROOTS;
            const int iy = (idy[loc_i] + idy[loc_j] * (LI + 1)) * NROOTS;
            const int iz = (idz[loc_i] + idz[loc_j] * (LI + 1)) * NROOTS;

            double eri = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                eri += gx[ix + i_root] * gy[iy + i_root] * gz[iz + i_root];
            }
            local_output[i + j * n_density_elements_i] += eri * prefactor;
        }
    }
}

template <int LI, int LJ>
__global__
static void GINTfill_int3c1e_charge_contracted_kernel_expanded(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                              const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                              const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int L_SUM = LI + LJ;
    constexpr int NROOTS = L_SUM / 2 + 1;

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

    constexpr int n_density_elements_i = (LI + 1) * (LI + 2) / 2;
    constexpr int n_density_elements_j = (LJ + 1) * (LJ + 2) / 2;
    double output_cache[n_density_elements_i * n_density_elements_j] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[3 * NROOTS * (LI + 1) * (LJ + 1)];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, LI, LJ, charge_exponent, omega);
            GINTwrite_int3c1e_charge_contracted<LI, LJ>(g, output_cache, charge);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
#pragma unroll
    for (int j = 0; j < n_density_elements_j; j++) {
#pragma unroll
        for (int i = 0; i < n_density_elements_i; i++) {
            const double eri_grid_sum = output_cache[i + j * n_density_elements_i];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j), eri_grid_sum);
        }
    }
}

template <int NROOTS>
__device__
static void GINTwrite_int3c1e_charge_contracted(const double* g, double* local_output, const double prefactor, const int i_l, const int j_l)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    for (int j = 0; j < (j_l + 1) * (j_l + 2) / 2; j++) {
        for (int i = 0; i < (i_l + 1) * (i_l + 2) / 2; i++) {
            const int loc_j = c_l_locs[j_l] + j;
            const int loc_i = c_l_locs[i_l] + i;

            const int ix = (idx[loc_i] + idx[loc_j] * (i_l + 1)) * NROOTS;
            const int iy = (idy[loc_i] + idy[loc_j] * (i_l + 1)) * NROOTS;
            const int iz = (idz[loc_i] + idz[loc_j] * (i_l + 1)) * NROOTS;

            double eri = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                eri += gx[ix + i_root] * gy[iy + i_root] * gz[iz + i_root];
            }
            local_output[i + j * ((i_l + 1) * (i_l + 2) / 2)] += eri * prefactor;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTfill_int3c1e_charge_contracted_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
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

    constexpr int l_sum_max = (NROOTS - 1) * 2 + 1;
    constexpr int l_i_max_density_elements = (l_sum_max + 1) / 2;
    constexpr int l_j_max_density_elements = l_sum_max - l_i_max_density_elements;
    double output_cache[(l_i_max_density_elements + 1) * (l_i_max_density_elements + 2) / 2
                        * (l_j_max_density_elements + 1) * (l_j_max_density_elements + 2) / 2] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double g[GSIZE_INT3C_1E];

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l, j_l, charge_exponent, omega);
            GINTwrite_int3c1e_charge_contracted<NROOTS>(g, output_cache, charge, i_l, j_l);
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    for (int j = 0; j < (j_l + 1) * (j_l + 2) / 2; j++) {
        for (int i = 0; i < (i_l + 1) * (i_l + 2) / 2; i++) {
            const double eri_grid_sum = output_cache[i + j * ((i_l + 1) * (i_l + 2) / 2)];
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j), eri_grid_sum);
        }
    }
}

template <int L_SUM>
__global__
static void GINTfill_int3c1e_density_contracted_kernel_general(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                               const BasisProdOffsets offsets, const int nprim_ij,
                                                               const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int NROOTS = L_SUM / 2 + 1;

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

    double eri_with_density_pair_sum = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {
        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
        const int* bas_pair2bra = c_bpcache.bas_pair2bra;
        // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
        const int ish = bas_pair2bra[bas_ij];
        // const int jsh = bas_pair2ket[bas_ij];

        double D_hermite[(L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6];
#pragma unroll
        for (int i_t = 0; i_t < (L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6; i_t++) {
            D_hermite[i_t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + i_t * hermite_density_offsets.n_pair_of_angular_pair];
        }

        double eri_with_density_per_pair = 0.0;
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            double g[NROOTS * (L_SUM + 1) * 3];
            GINT_g1e_without_hrr<L_SUM>(g, Cx, Cy, Cz, ish, ij, charge_exponent, omega);

            double eri_with_density_per_primitive = 0.0;
#pragma unroll
            for (int i_x = 0, i_t = 0; i_x <= L_SUM; i_x++) {
#pragma unroll
                for (int i_y = 0; i_x + i_y <= L_SUM; i_y++) {
#pragma unroll
                    for (int i_z = 0; i_x + i_y + i_z <= L_SUM; i_z++, i_t++) {
                        double eri_per_hermite = 0.0;
#pragma unroll
                        for (int i_root = 0; i_root < NROOTS; i_root++) {
                            const double gx = g[i_root + NROOTS * i_x];
                            const double gy = g[i_root + NROOTS * i_y + NROOTS * (L_SUM + 1)];
                            const double gz = g[i_root + NROOTS * i_z + NROOTS * (L_SUM + 1) * 2];
                            eri_per_hermite += gx * gy * gz;
                        }
                        const double D_t = D_hermite[i_t];
                        eri_with_density_per_primitive += eri_per_hermite * D_t;
                    }
                }
            }

            eri_with_density_per_pair += eri_with_density_per_primitive;
        }
        eri_with_density_pair_sum += eri_with_density_per_pair;
    }
    atomicAdd(output + task_grid, eri_with_density_pair_sum);
}
