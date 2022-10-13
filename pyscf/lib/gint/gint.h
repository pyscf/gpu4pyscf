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

#include <stdlib.h>
#include <cint.h>
#include <stddef.h>

// boundaries for gint
// up to g functions
#define GPU_LMAX        4
#define GPU_CART_MAX    15
#define THREADSX        8
#define THREADSY        8
#define THREADS         (THREADSX * THREADSY)
#define SHARED_MEM_NFIJ_MAX     18

// 1 roots upto (ps|ss)  6
// 2 roots upto (pp|ps)  48
// 3 roots upto (dp|pp)  216
// 4 roots upto (dd|dp)  648
// 5 roots upto (fd|dd)  1620
// 6 roots upto (ff|fd)  3456
// 7 roots upto (gf|ff)  6720
// 8 roots upto (gg|gf)  12000
#define GSIZE1       6
#define GSIZE2       48
#define GSIZE3       216
#define GSIZE4       648
#define GSIZE5       1620
#define GSIZE6       3456
#define GSIZE7       6720
#define GSIZE8       12000
#define MAX_GSIZE    GSIZE8

#define GOUTSIZE1    (GSIZE1 + 3)
#define GOUTSIZE2    (GSIZE2 + 27)
#define GOUTSIZE3    (GSIZE3 + 162)
#define GOUTSIZE4    (GSIZE4 + 648)
#define GOUTSIZE5    (GSIZE5 + 2160)
#define GOUTSIZE6    (GSIZE6 + 6000)
#define GOUTSIZE7    (GSIZE7 + 15000)
#define GOUTSIZE8    (GSIZE8 + 15*15*15*10)
#define MAX_GOUTSIZE GOUTSIZE8
// nf=10^4
#define NFffff       10000

#ifndef HAVE_DEFINED_GINTENVVAS_H
#define HAVE_DEFINED_GINTENVVAS_H

typedef struct {
  int stride_j;
  int stride_k;
  int stride_l;
  int stride_xyz;
  int ao_offsets_k;
  int ao_offsets_l;
  int nao;
  double *data;
} ERITensor;

typedef struct {
        int16_t i_l;
        int16_t j_l;
        int16_t k_l;
        int16_t l_l;
        int16_t nfi;
        int16_t nfj;
        int16_t nfk;
        int16_t nfl;
        int nf;
        int nrys_roots;
        int g_size;
        int g_size_ij;
        int16_t ibase;
        int16_t kbase;
        int16_t ijmin;
        int16_t ijmax;
        int16_t klmin;
        int16_t klmax;
        int nao;
        int stride_ijmax;
        int stride_ijmin;
        int stride_klmax;
        int stride_klmin;
        double fac;

        int nprim_ij;
        int nprim_kl;
        int16_t *idx;
        double *uw;
} GINTEnvVars;

// ProdContractionType <-> (li, lj, nprimi, nprimj)
typedef struct {
    int l_bra;  // angular of bra
    int l_ket;  // angular of ket
    int nprim_12;  // nprimi * nprimj
    int npairs;  // nbas_bra * nbas_ket in this contraction type
} ContractionProdType;

typedef struct {
        int ntasks_ij;
        int ntasks_kl;
        int bas_ij;
        int bas_kl;
        int primitive_ij;
        int primitive_kl;
} BasisProdOffsets;

typedef struct {
    int nbas;  // len(bas_coords)
    int ncptype;  // len(cptype)
    ContractionProdType *cptype;
    int *bas_pairs_locs;  // len(bas_pair2bra) = sum(cptype[:].nparis)
    int *primitive_pairs_locs;  // len(a12) = sum(cptype[:].nparis*cptype[:].nprim_12)
    int *bas_pair2shls;
    double *aexyz;

    // Data below held on GPU global memory
    double *bas_coords;  // basis coordinates
    int *bas_pair2bra;
    int *bas_pair2ket;
    int *ao_loc;
    double *a12;
    double *e12;
    double *x12;
    double *y12;
    double *z12;
} BasisProdCache;


typedef void (*FPtr_CPUkernel_jk)(double *g, double **dm, double **v,
                                  int *shls, GINTEnvVars *envs, int *ibuf);

#endif
