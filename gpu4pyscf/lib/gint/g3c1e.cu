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
#include "config.h" // For SQRTPI

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

    const int16_t* cart_component_ix = c_idx4c;
    const int16_t* cart_component_iy = c_idx4c + GPU_CART_MAX;
    const int16_t* cart_component_iz = c_idx4c + GPU_CART_MAX * 2;
    const int16_t* cart_component_jx = c_idx4c + GPU_CART_MAX * 3;
    const int16_t* cart_component_jy = c_idx4c + GPU_CART_MAX * 4;
    const int16_t* cart_component_jz = c_idx4c + GPU_CART_MAX * 5;

    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    for (int j = j0; j < j1; j++) {
        for (int i = i0; i < i1; i++) {
            const int ix = cart_component_ix[i - i0] + cart_component_jx[j - j0] * (i_l+1);
            const int iy = cart_component_iy[i - i0] + cart_component_jy[j - j0] * (i_l+1);
            const int iz = cart_component_iz[i - i0] + cart_component_jz[j - j0] * (i_l+1);

            double eri = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                eri += gx[ix * NROOTS + i_root] * gy[iy * NROOTS + i_root] * gz[iz * NROOTS + i_root];
            }
            output[i + j * stride_j + i_grid * stride_ij] += eri;
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
void GINTfill_int3c1e_kernel_general(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                     const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                     const double omega, const double* grid_points)
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

    double g[GSIZE_INT3C_1E];

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        GINTg1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l, j_l, omega);
        GINTwrite_int3c1e<NROOTS>(g, output, ish, jsh, task_grid, i_l, j_l, stride_j, stride_ij, ao_offsets_i, ao_offsets_j);
    }
}

template <int NROOTS>
__global__
void GINT_int3c1e_density_contracted_kernel_general(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                    const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                    const double omega, const double* grid_points)
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
    // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    // const int jsh = bas_pair2ket[bas_ij];

    const double* grid_point = grid_points + task_grid * 3;

    constexpr int l_max = (NROOTS - 1) * 2 + 1;
    double D_hermite[(l_max + 1) * (l_max + 2) * (l_max + 3) / 6];
    const int l = i_l + j_l;
    for (int i_t = 0; i_t < (l + 1) * (l + 2) * (l + 3) / 6; i_t++) {
        D_hermite[i_t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + i_t * hermite_density_offsets.n_pair_of_angular_pair];
    }

    double eri_with_density = 0.0;
    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        double g[NROOTS * (l_max + 1) * 3];
        GINT_g1e_without_hrr<NROOTS>(g, grid_point, ish, ij, l, omega);

        double eri_with_density_primitive = 0.0;
        for (int i_x = 0, i_t = 0; i_x <= l; i_x++)
            for (int i_y = 0; i_x + i_y <= l; i_y++)
                for (int i_z = 0; i_x + i_y + i_z <= l; i_z++, i_t++)
                    for (int i_root = 0; i_root < NROOTS; i_root++) {
                        const double gx = g[i_root + NROOTS * i_x];
                        const double gy = g[i_root + NROOTS * i_y + NROOTS * (l + 1)];
                        const double gz = g[i_root + NROOTS * i_z + NROOTS * (l + 1) * 2];
                        eri_with_density_primitive += gx * gy * gz * D_hermite[i_t];
                    }

        eri_with_density += eri_with_density_primitive;
    }
    atomicAdd(output + task_grid, eri_with_density);
}
