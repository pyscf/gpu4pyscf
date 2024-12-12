/* Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

template <int NROOTS>
__global__
static void GINTfill_int3c1e_density_contracted_kernel_general(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                               const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                               const double omega, const double* grid_points, const double* charge_exponents)
{
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

        constexpr int l_max = (NROOTS - 1) * 2 + 1;
        double D_hermite[(l_max + 1) * (l_max + 2) * (l_max + 3) / 6];
        const int l = i_l + j_l;
        for (int i_t = 0; i_t < (l + 1) * (l + 2) * (l + 3) / 6; i_t++) {
            D_hermite[i_t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + i_t * hermite_density_offsets.n_pair_of_angular_pair];
        }

        double eri_with_density_per_pair = 0.0;
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            double g[NROOTS * (l_max + 1) * 3];
            GINT_g1e_without_hrr<NROOTS>(g, Cx, Cy, Cz, ish, ij, l, charge_exponent, omega);

            double eri_with_density_per_primitive = 0.0;
            for (int i_x = 0, i_t = 0; i_x <= l; i_x++) {
                for (int i_y = 0; i_x + i_y <= l; i_y++) {
                    for (int i_z = 0; i_x + i_y + i_z <= l; i_z++, i_t++) {
                        const double D_t = D_hermite[i_t];
#pragma unroll
                        for (int i_root = 0; i_root < NROOTS; i_root++) {
                            const double gx = g[i_root + NROOTS * i_x];
                            const double gy = g[i_root + NROOTS * i_y + NROOTS * (l + 1)];
                            const double gz = g[i_root + NROOTS * i_z + NROOTS * (l + 1) * 2];
                            eri_with_density_per_primitive += gx * gy * gz * D_t;
                        }
                    }
                }
            }

            eri_with_density_per_pair += eri_with_density_per_primitive;
        }
        eri_with_density_pair_sum += eri_with_density_per_pair;
    }
    atomicAdd(output + task_grid, eri_with_density_pair_sum);
}
