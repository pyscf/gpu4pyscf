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

#include <stdbool.h>

// void GINTinit_J_density_reorder(const double* D_matrix, double* D_pair_ordered, const int n_dm, const int n_ao, const int n_pair_type,
//                                 const int* bas_pair2shls, const int* bas_pairs_locs, const int* density_offset, const int* ao_loc)
// {
//     const int n_bas_pairs = bas_pairs_locs[n_pair_type];
//     const int n_density_cache = density_offset[n_pair_type];
//     for (int i_dm = 0; i_dm < n_dm; i_dm++) {
//         for (int i_pair_type = 0; i_pair_type < n_pair_type; i_pair_type++) {
//             const int n_pair_of_angular_pair_type = bas_pairs_locs[i_pair_type + 1] - bas_pairs_locs[i_pair_type];
//             for (int i_pair = bas_pairs_locs[i_pair_type]; i_pair < bas_pairs_locs[i_pair_type + 1]; i_pair++) {
//                 const int ish = bas_pair2shls[i_pair];
//                 const int jsh = bas_pair2shls[n_bas_pairs + i_pair];
//                 const int i0 = ao_loc[ish];
//                 const int i1 = ao_loc[ish + 1];
//                 const int j0 = ao_loc[jsh];
//                 const int j1 = ao_loc[jsh + 1];
//                 for (int j = j0; j < j1; j++)
//                     for (int i = i0; i < i1; i++)
//                         D_pair_ordered[i_pair - bas_pairs_locs[i_pair_type] + density_offset[i_pair_type] + ((i-i0) + (j-j0) * (i1-i0)) * n_pair_of_angular_pair_type + i_dm * n_density_cache]
//                             = D_matrix[i + j * n_ao + i_dm * n_ao * n_ao];
//             }
//         }
//     }
// }

// void GINTclean_J_matrix_reorder(const double* J_pair_ordered, double* J_matrix, const int n_dm, const int n_ao, const int n_pair_type,
//                                 const int* bas_pair2shls, const int* bas_pairs_locs, const int* density_offset, const int* ao_loc)
// {
//     const int n_bas_pairs = bas_pairs_locs[n_pair_type];
//     const int n_density_cache = density_offset[n_pair_type];
//     for (int i_dm = 0; i_dm < n_dm; i_dm++) {
//         for (int i_pair_type = 0; i_pair_type < n_pair_type; i_pair_type++) {
//             const int n_pair_of_angular_pair_type = bas_pairs_locs[i_pair_type + 1] - bas_pairs_locs[i_pair_type];
//             for (int i_pair = bas_pairs_locs[i_pair_type]; i_pair < bas_pairs_locs[i_pair_type + 1]; i_pair++) {
//                 const int ish = bas_pair2shls[i_pair];
//                 const int jsh = bas_pair2shls[n_bas_pairs + i_pair];
//                 const int i0 = ao_loc[ish];
//                 const int i1 = ao_loc[ish + 1];
//                 const int j0 = ao_loc[jsh];
//                 const int j1 = ao_loc[jsh + 1];
//                 for (int j = j0; j < j1; j++)
//                     for (int i = i0; i < i1; i++)
//                         J_matrix[i + j * n_ao + i_dm * n_ao * n_ao]
//                             += J_pair_ordered[i_pair - bas_pairs_locs[i_pair_type] + density_offset[i_pair_type] + ((i-i0) + (j-j0) * (i1-i0)) * n_pair_of_angular_pair_type + i_dm * n_density_cache];
//             }
//         }
//     }
// }

// Copied from gpu4pyscf/lib/gvhf-rys/cart2xyz.c
static int _LEN_CART0[] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};

static int _BINOMIAL_COEF[] = {
    1,
    1,   1,
    1,   2,   1,
    1,   3,   3,   1,
    1,   4,   6,   4,   1,
    1,   5,  10,  10,   5,   1,
    1,   6,  15,  20,  15,   6,   1,
    1,   7,  21,  35,  35,  21,   7,   1,
    1,   8,  28,  56,  70,  56,  28,   8,   1,
    1,   9,  36,  84, 126, 126,  84,  36,   9,   1,
    1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1,
    1,  11,  55, 165, 330, 462, 462, 330, 165,  55,  11,   1,
    1,  12,  66, 220, 495, 792, 924, 792, 495, 220,  66,  12,   1,
    1,  13,  78, 286, 715,1287,1716,1716,1287, 715, 286,  78,  13,   1,
    1,  14,  91, 364,1001,2002,3003,3432,3003,2002,1001, 364,  91,  14,   1,
    1,  15, 105, 455,1365,3003,5005,6435,6435,5005,3003,1365, 455, 105,  15,   1,
};

#define BINOMIAL(n, i)  (_BINOMIAL_COEF[_LEN_CART0[n]+i])
// Copied from gpu4pyscf/lib/gvhf-rys/cart2xyz.c ended

int hermite_xyz_to_t_index(const int x, const int y, const int z, const int l)
{
    // This index is consistent with the following for loop, and recover t from (x,y,z):
    // for (int x = 0, t = 0; x <= l; x++)
    //     for (int y = 0; x + y <= l; y++)
    //         for (int z = 0; x + y + z <= l; z++, t++)
    return ((l + 1) * (l + 2) * (l + 3) - (l + 1 - x) * (l + 2 - x) * (l + 3 - x)) / 6 + ((l - x + 1) * (l - x + 2) - (l - x + 1 - y) * (l - x + 2 - y)) / 2 + z;
}

void GINTinit_J_density_rys_preprocess(const double* D_matrix, double* D_pair_ordered, const int n_dm, const int n_ao, const int n_pair_type,
                                       const int* bas_pair2shls, const int* bas_pairs_locs, const int* l_ij, const int* density_offset, const int* ao_loc,
                                       const double* bas_coords, const bool symmetric)
{
    const int n_bas_pairs = bas_pairs_locs[n_pair_type];
    const int n_total_hermite_density = density_offset[n_pair_type];
    for (int i_dm = 0; i_dm < n_dm; i_dm++) {
        for (int i_pair_type = 0; i_pair_type < n_pair_type; i_pair_type++) {
            const int n_pair_of_angular_pair_type = bas_pairs_locs[i_pair_type + 1] - bas_pairs_locs[i_pair_type];
            const int li = l_ij[i_pair_type];
            const int lj = l_ij[i_pair_type + n_pair_type];
            const int l = li + lj;
            for (int i_pair = bas_pairs_locs[i_pair_type]; i_pair < bas_pairs_locs[i_pair_type + 1]; i_pair++) {
                const int ish = bas_pair2shls[i_pair];
                const int jsh = bas_pair2shls[n_bas_pairs + i_pair];
                const double Ax = bas_coords[ish * 3 + 0];
                const double Ay = bas_coords[ish * 3 + 1];
                const double Az = bas_coords[ish * 3 + 2];
                const double Bx = bas_coords[jsh * 3 + 0];
                const double By = bas_coords[jsh * 3 + 1];
                const double Bz = bas_coords[jsh * 3 + 2];
                const int i0 = ao_loc[ish];
                const int j0 = ao_loc[jsh];

                double D_hermite[(l + 1) * (l + 2) * (l + 3) / 6];
                for (int i_x = 0, i_t = 0; i_x <= l; i_x++)
                    for (int i_y = 0; i_x + i_y <= l; i_y++)
                        for (int i_z = 0; i_x + i_y + i_z <= l; i_z++, i_t++) {
                            D_hermite[i_t] = 0.0;
                        }

                // Assuming canonical order of density matrix for each angular momentum pair
                for (int i_x_i = li, i_density_i = 0; i_x_i >= 0; i_x_i--)
                    for (int i_y_i = li - i_x_i; i_y_i >= 0; i_y_i--, i_density_i++) {
                        const int i_z_i = li - i_x_i - i_y_i;
                        for (int i_x_j = lj, i_density_j = 0; i_x_j >= 0; i_x_j--)
                            for (int i_y_j = lj - i_x_j; i_y_j >= 0; i_y_j--, i_density_j++) {
                                const int i_z_j = lj - i_x_j - i_y_j;

                                const double D_cartesian = ((!symmetric) || i0 == j0) ?
                                                            D_matrix[(i0 + i_density_i) + (j0 + i_density_j) * n_ao + i_dm * n_ao * n_ao] :
                                                            D_matrix[(i0 + i_density_i) + (j0 + i_density_j) * n_ao + i_dm * n_ao * n_ao] + D_matrix[(j0 + i_density_j) + (i0 + i_density_i) * n_ao + i_dm * n_ao * n_ao];

                                // The next piece of loops follows from:
                                // I_{k_x, l_x}(t) = \sum_{t_x = 0}^{l_x} \left({\begin{array}{*{20}c} l_x \\ l_x - t_x \end{array}}\right) (C_x - D_x)^{l_x - t_x} I_{k_x + t_x, 0}(t)
                                for (int i_x_t = 0; i_x_t <= i_x_j; i_x_t++)
                                    for (int i_y_t = 0; i_y_t <= i_y_j; i_y_t++)
                                        for (int i_z_t = 0; i_z_t <= i_z_j; i_z_t++) {
                                            double power_AB = 1.0;
                                            for (int i_power = 0; i_power < i_x_j - i_x_t; i_power++) power_AB *= Ax - Bx;
                                            for (int i_power = 0; i_power < i_y_j - i_y_t; i_power++) power_AB *= Ay - By;
                                            for (int i_power = 0; i_power < i_z_j - i_z_t; i_power++) power_AB *= Az - Bz;
                                            D_hermite[hermite_xyz_to_t_index(i_x_t + i_x_i, i_y_t + i_y_i, i_z_t + i_z_i, l)]
                                                += BINOMIAL(i_x_j, i_x_j - i_x_t) * BINOMIAL(i_y_j, i_y_j - i_y_t) * BINOMIAL(i_z_j, i_z_j - i_z_t) * power_AB * D_cartesian;
                                        }
                            }
                    }

                for (int i_x = 0, i_t = 0; i_x <= l; i_x++)
                    for (int i_y = 0; i_x + i_y <= l; i_y++)
                        for (int i_z = 0; i_x + i_y + i_z <= l; i_z++, i_t++) {
                            D_pair_ordered[i_pair - bas_pairs_locs[i_pair_type] + density_offset[i_pair_type] + i_t * n_pair_of_angular_pair_type + i_dm * n_total_hermite_density]
                                = D_hermite[i_t];
                        }
            }
        }
    }
}
