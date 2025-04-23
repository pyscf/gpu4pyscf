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
void GINTsort_bas_coordinates(double *bas_coords, int *bas_atm, int *atm, int natm,
                              int *bas, int nbas, double *env);
void GINTinit_aexyz(double *aexyz, BasisProdCache *bpcache, double diag_fac,
                    int *atm, int natm, int *bas, int nbas, double *env);
#ifdef __cplusplus
}
#endif
