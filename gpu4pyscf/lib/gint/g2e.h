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

#include "gint.h"
#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif
void GINTinit_EnvVars(GINTEnvVars *envs,
                      ContractionProdType *cp_ij, ContractionProdType *cp_kl, int *ng);
void GINTinit_EnvVars_nabla1i(GINTEnvVars *envs,
                              ContractionProdType *cp_ij,
                              ContractionProdType *cp_kl,
                              int *ng);

void GINTinit_2c_gidx(int *idx, int li, int lj);
void GINTinit_2c_gidx_nabla1i(int *idx, int li, int lj);
void GINTinit_4c_idx(int16_t *idx, int *ij_idx, int *kl_idx, GINTEnvVars *envs);
void GINTg2e_index_xyz(int16_t *idx, GINTEnvVars *envs);
void GINTinit_index1d_xyz(int *idx, int *l_locs);
void GINTinit_uw_s1(double *uw_buf, BasisProdOffsets *offsets,
                    GINTEnvVars *envs, BasisProdCache *bpcache);
void GINTinit_uw_s2(double *uw_buf, BasisProdOffsets *offsets,
                    GINTEnvVars *envs, BasisProdCache *bpcache);

void GINTinit_contraction_types(BasisProdCache *bpcache,
                                int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                                int *atm, int natm, int *bas, int nbas, double *env);
void GINTsort_bas_coordinates(double *bas_coords, int *atm, int natm,
                              int *bas, int nbas, double *env);
void GINTinit_aexyz(double *aexyz, BasisProdCache *bpcache, double diag_fac,
                    int *atm, int natm, int *bas, int nbas, double *env);
#ifdef __cplusplus
}
#endif