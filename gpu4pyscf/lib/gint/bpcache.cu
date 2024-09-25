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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

extern "C" {
__host__
void GINTdel_basis_prod(BasisProdCache **pbp)
{
    BasisProdCache *bpcache = *pbp;
    if (bpcache == NULL) {
        return;
    }

    if (bpcache->cptype != NULL) {
        free(bpcache->cptype);
        free(bpcache->primitive_pairs_locs);
    }

    if (bpcache->aexyz != NULL) {
        free(bpcache->aexyz);
    }

    if (bpcache->a12 != NULL) {
        FREE(bpcache->bas_coords);
        FREE(bpcache->bas_pair2bra);
        FREE(bpcache->ao_loc);
        FREE(bpcache->a12);
    }

    free(bpcache);
    *pbp = NULL;
}

__host__
void GINTdel_basis_product_mixed_precision(BasisProdCache **pbp, BasisProductCacheSinglePrecision **pbp_single, BasisProductCacheDoublePrecision **pbp_double)
{
    GINTdel_basis_prod(pbp);

    BasisProductCacheDoublePrecision *bpcache_double = *pbp_double;
    if (bpcache_double->d_a12 != NULL) {
        FREE(bpcache_double->d_a12);
    }
    if (bpcache_double->d_i0 != NULL) {
        FREE(bpcache_double->d_i0);
    }
    free(bpcache_double);
    *pbp_double = NULL;

    BasisProductCacheSinglePrecision *bpcache_single = *pbp_single;
    if (bpcache_single->d_a12 != NULL) {
        FREE(bpcache_single->d_a12);
    }
    free(bpcache_single);
    *pbp_single = NULL;
}

__host__
void GINTinit_basis_prod(BasisProdCache **pbp, double diag_fac, int *ao_loc,
                         int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
                               atm, natm, bas, nbas, env);
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 7);
    GINTinit_aexyz(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize ao_loc on GPU
    DEVICE_INIT(int, d_ao_loc, ao_loc, nbas+1);
    bpcache->ao_loc = d_ao_loc;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    double *bas_coords = (double *)malloc(sizeof(double) * nbas * 3);
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
    DEVICE_INIT(double, d_bas_coords, bas_coords, nbas * 3);
    bpcache->bas_coords = d_bas_coords;
    free(bas_coords);

    // initialize pair data on GPU memory
    DEVICE_INIT(double, d_aexyz, aexyz, n_primitive_pairs * 7);
    DEVICE_INIT(int, d_bas_pair2shls, bas_pair2shls, n_bas_pairs * 2);
    bpcache->a12 = d_aexyz;
    bpcache->e12 = d_aexyz + n_primitive_pairs * 1;
    bpcache->x12 = d_aexyz + n_primitive_pairs * 2;
    bpcache->y12 = d_aexyz + n_primitive_pairs * 3;
    bpcache->z12 = d_aexyz + n_primitive_pairs * 4;
    bpcache->a1  = d_aexyz + n_primitive_pairs * 5;
    bpcache->a2  = d_aexyz + n_primitive_pairs * 6;
    bpcache->bas_pair2bra = d_bas_pair2shls;
    bpcache->bas_pair2ket = d_bas_pair2shls + n_bas_pairs;
}

__host__
void GINTinit_basis_product_mixed_precision(BasisProdCache **pbp,
                                            BasisProductCacheSinglePrecision **pbp_single, BasisProductCacheDoublePrecision **pbp_double,
                                            double diag_fac, int *ao_loc,
                                            int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                                            int *atm, int natm, int *bas, int nbas, double *env)
{
    GINTinit_basis_prod(pbp, diag_fac, ao_loc, bas_pair2shls, bas_pairs_locs, ncptype, atm, natm, bas, nbas, env);

    const BasisProdCache* bpcache = *pbp;
    const int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    const int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];

    BasisProductCacheSinglePrecision *bpcache_single = (BasisProductCacheSinglePrecision *)malloc(sizeof(BasisProductCacheSinglePrecision));
    memset(bpcache_single, 0, sizeof(BasisProductCacheSinglePrecision));
    *pbp_single = bpcache_single;

    BasisProductCacheDoublePrecision *bpcache_double = (BasisProductCacheDoublePrecision *)malloc(sizeof(BasisProductCacheDoublePrecision));
    memset(bpcache_double, 0, sizeof(BasisProductCacheDoublePrecision));
    *pbp_double = bpcache_double;

    double *h_aexyz_double = (double *)malloc((n_primitive_pairs * 6 + n_bas_pairs * 3) * sizeof(double));
    int *h_i0i1j0j1 = (int *)malloc(n_bas_pairs * 4 * sizeof(int));
    GINTinit_populate_pair_data(h_aexyz_double, h_i0i1j0j1, bpcache, diag_fac, atm, natm, bas, nbas, ao_loc, env);
    DEVICE_INIT(double, d_aexyz_double, h_aexyz_double, n_primitive_pairs * 6 + n_bas_pairs * 3);
    DEVICE_INIT(int, d_i0i1j0j1, h_i0i1j0j1, n_bas_pairs * 4);
    bpcache_double->d_a12 = d_aexyz_double;
    bpcache_double->d_e12 = d_aexyz_double + n_primitive_pairs * 1;
    bpcache_double->d_x12 = d_aexyz_double + n_primitive_pairs * 2;
    bpcache_double->d_y12 = d_aexyz_double + n_primitive_pairs * 3;
    bpcache_double->d_z12 = d_aexyz_double + n_primitive_pairs * 4;
    bpcache_double->d_a1  = d_aexyz_double + n_primitive_pairs * 5;
    bpcache_double->d_x1  = d_aexyz_double + n_primitive_pairs * 6;
    bpcache_double->d_y1  = d_aexyz_double + n_primitive_pairs * 6 + n_bas_pairs;
    bpcache_double->d_z1  = d_aexyz_double + n_primitive_pairs * 6 + n_bas_pairs * 2;
    bpcache_double->d_i0  = d_i0i1j0j1;
    bpcache_double->d_i1  = d_i0i1j0j1 + n_bas_pairs * 1;
    bpcache_double->d_j0  = d_i0i1j0j1 + n_bas_pairs * 2;
    bpcache_double->d_j1  = d_i0i1j0j1 + n_bas_pairs * 3;

    float *h_aexyz_single = (float *)malloc((n_primitive_pairs * 6 + n_bas_pairs * 3) * sizeof(float));
    for (int i = 0; i < n_primitive_pairs * 6 + n_bas_pairs * 3; i++)
        h_aexyz_single[i] = static_cast<float>(h_aexyz_double[i]);
    DEVICE_INIT(float, d_aexyz_single, h_aexyz_single, n_primitive_pairs * 6 + n_bas_pairs * 3);
    bpcache_single->d_a12 = d_aexyz_single;
    bpcache_single->d_e12 = d_aexyz_single + n_primitive_pairs * 1;
    bpcache_single->d_x12 = d_aexyz_single + n_primitive_pairs * 2;
    bpcache_single->d_y12 = d_aexyz_single + n_primitive_pairs * 3;
    bpcache_single->d_z12 = d_aexyz_single + n_primitive_pairs * 4;
    bpcache_single->d_a1  = d_aexyz_single + n_primitive_pairs * 5;
    bpcache_single->d_x1  = d_aexyz_single + n_primitive_pairs * 6;
    bpcache_single->d_y1  = d_aexyz_single + n_primitive_pairs * 6 + n_bas_pairs;
    bpcache_single->d_z1  = d_aexyz_single + n_primitive_pairs * 6 + n_bas_pairs * 2;
    bpcache_single->d_i0  = d_i0i1j0j1;
    bpcache_single->d_i1  = d_i0i1j0j1 + n_bas_pairs * 1;
    bpcache_single->d_j0  = d_i0i1j0j1 + n_bas_pairs * 2;
    bpcache_single->d_j1  = d_i0i1j0j1 + n_bas_pairs * 3;

    free(h_aexyz_double);
    free(h_aexyz_single);
    free(h_i0i1j0j1);
}
}

