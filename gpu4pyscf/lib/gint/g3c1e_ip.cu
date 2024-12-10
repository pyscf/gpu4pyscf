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
