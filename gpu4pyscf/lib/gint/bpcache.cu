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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"
/*
#include "cint2e.cuh"
#include "fill_ints.cu"
#include "g2e.cu"
#include "rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"
#include "g3c2e.cu"
#include "g3c2e_ip1.cu"
#include "g3c2e_ip2.cu"
*/
extern "C" { __host__
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
        FREE(bpcache->bas_atm);
        FREE(bpcache->bas_pair2bra);
        FREE(bpcache->ao_loc);
        FREE(bpcache->a12);
    }

    free(bpcache);
    *pbp = NULL;
}

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
    int *bas_atm = (int *)malloc(sizeof(int) * nbas);
    GINTsort_bas_coordinates(bas_coords, bas_atm, atm, natm, bas, nbas, env);
    DEVICE_INIT(double, d_bas_coords, bas_coords, nbas * 3);
    DEVICE_INIT(int, d_bas_atm, bas_atm, nbas);
    bpcache->bas_coords = d_bas_coords;
    bpcache->bas_atm = d_bas_atm;
    free(bas_coords);
    free(bas_atm);

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
}

