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
#include <stddef.h>

// #include <cint.h>
// global parameters in env
// Overall cutoff for integral prescreening, value needs to be ~ln(threshold)
#define PTR_EXPCUTOFF           0
// R_C of (r-R_C) in dipole, GIAO operators
#define PTR_COMMON_ORIG         1
// R_O in 1/|r-R_O|
#define PTR_RINV_ORIG           4
// ZETA parameter for Gaussian charge distribution (Gaussian nuclear model)
#define PTR_RINV_ZETA           7
// omega parameter in range-separated coulomb operator
// LR interaction: erf(omega*r12)/r12 if omega > 0
// SR interaction: erfc(omega*r12)/r12 if omega < 0
#define PTR_RANGE_OMEGA         8
// Yukawa potential and Slater-type geminal e^{-zeta r}
#define PTR_F12_ZETA            9
// Gaussian type geminal e^{-zeta r^2}
#define PTR_GTG_ZETA            10
#define NGRIDS                  11
#define PTR_GRIDS               12
#define PTR_ENV_START           20


// slots of atm
#define CHARGE_OF       0
#define PTR_COORD       1
#define NUC_MOD_OF      2
#define PTR_ZETA        3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS       6


// slots of bas
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define NCTR_OF         3
#define KAPPA_OF        4
#define PTR_EXP         5
#define PTR_COEFF       6
#define RESERVE_BASLOT  7
#define BAS_SLOTS       8

// boundaries for gint
#define GPU_AO_LMAX     4           //l = 0..5 
#define GPU_AUX_LMAX    6   
#define GPU_LMAX        6

#define GPU_AO_NF       15          // up to g orbitals
#define GPU_AUX_NF      28          // > (ANG_MAX*(ANG_MAX+1)/2)
#define GPU_CART_MAX    28  
#define NF_MAX_INT3C    (GPU_AO_NF * GPU_AO_NF * GPU_AUX_NF)

// threads for GPU
#define THREADSX        16
#define THREADSY        16
#define THREADS         (THREADSX * THREADSY)
#define MAX_STREAMS         16
#define SHARED_MEM_NFIJ_MAX     18

// sum of total l indices
#define TOT_NF         (1+3+6+10+15+21+28)

#define POLYFIT_ORDER  9 
#define POLYFIT_ORDER_IP 9

#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096
// 3*nroots*(i*j*k*l)
// 1 roots upto (ps|ss)  6         = 3*1*(2*1*1*1)
// 2 roots upto (pp|ps)  48        = 3*2*(2*2*2*1)
// 3 roots upto (dp|pp)  216       = 3*3*(3*2*2*2)
// 4 roots upto (dd|dp)  648       = 3*4*(3*3*3*2)
// 5 roots upto (fd|dd)  1620      = 3*5*(4*3*3*3)
// 6 roots upto (ff|fd)  3456      = 3*6*(4*4*4*3)
// 7 roots upto (gf|ff)  6720      = 3*7*(5*4*4*4)
// 8 roots upto (gg|gf)  12000     = 3*8*(5*5*5*4)
// 9 roots upto (hg|gg)  20250     = 3*9*(6*5*5*5)
// uncomment this for regular int2e

#define GSIZE1       6
#define GSIZE2       48
#define GSIZE3       216
#define GSIZE4       648
#define GSIZE5       1620
#define GSIZE6       3456
#define GSIZE7       6720
#define GSIZE8       12000
#define GSIZE9       20250
#define MAX_GSIZE    GSIZE9

// s->1, p->3, d->6, f->10, g->15, h->21

#define GOUTSIZE1    (GSIZE1 + 3)
#define GOUTSIZE2    (GSIZE2 + 27)
#define GOUTSIZE3    (GSIZE3 + 162)
#define GOUTSIZE4    (GSIZE4 + 648)
#define GOUTSIZE5    (GSIZE5 + 2160)
#define GOUTSIZE6    (GSIZE6 + 6000)
#define GOUTSIZE7    (GSIZE7 + 15000)
#define GOUTSIZE8    (GSIZE8 + 15*15*15*10)
#define GOUTSIZE9    (GSIZE9 + 21*15*15*15)
#define MAX_GOUTSIZE GOUTSIZE9

#define NABLAGSIZE1  12
#define NABLAGSIZE2  96
#define NABLAGSIZE3  324
#define NABLAGSIZE4  972
#define NABLAGSIZE5  2160
#define NABLAGSIZE6  4608
#define NABLAGSIZE7  8400
#define MAX_NABLAGSIZE NABLAGSIZE7

#define NABLAGOUTSIZE1    (NABLAGSIZE1 + 6)
#define NABLAGOUTSIZE2    (NABLAGSIZE2 + 54)
#define NABLAGOUTSIZE3    (NABLAGSIZE3 + 486)
#define NABLAGOUTSIZE4    (NABLAGSIZE4 + 1944)
#define NABLAGOUTSIZE5    (NABLAGSIZE5 + 7776)
#define NABLAGOUTSIZE6    (NABLAGSIZE6 + 2760)
#define NABLAGOUTSIZE7    (NABLAGSIZE7 + 3600)
#define MAX_NABLAGOUTSIZE NABLAGOUTSIZE7

// 1 roots upto (ps|ss) = (10|00) -> 3*1*(2*1*1) = 6
// 2 roots upto (pp|ps) = (11|10) -> 3*2*(2*2*2) = 48
// 3 roots upto (dd|ps) = (22|10) -> 3*3*(3*3*2) = 162
// 4 roots upto (fd|ds) = (32|20) -> 3*4*(4*3*3) = 432
// 5 roots upto (ff|fs) = (33|30) -> 3*5*(4*4*4) = 960
// 6 roots upto (gg|fs) = (44|30) -> 3*6*(5*5*4) = 1800
// 7 roots upto (hg|gs) = (54|40) -> 3*7*(6*5*5) = 3150
// 8 roots upto (hh|hs) = (55|50) -> 3*8*(6*6*6) = 5184
// 9 roots upto (ii|hs) = (66|50) -> 3*9*(7*7*6) = 7938
#define GSIZE1_INT3C     6
#define GSIZE2_INT3C     48
#define GSIZE3_INT3C     162
#define GSIZE4_INT3C     432
#define GSIZE5_INT3C     960
#define GSIZE6_INT3C     1800
#define GSIZE7_INT3C     3150
#define GSIZE8_INT3C     5184
#define GSIZE9_INT3C     7938
#define MAX_GSIZE_INT3C  GSIZE9_INT3C

// s->1, p->3, d->6, f->10, g->15, h->21, i->28
#define GOUTSIZE1_INT3C     3
#define GOUTSIZE2_INT3C     3*3*3
#define GOUTSIZE3_INT3C     6*6*3
#define GOUTSIZE4_INT3C     10*6*6
#define GOUTSIZE5_INT3C     10*10*10
#define GOUTSIZE6_INT3C     15*15*10
#define GOUTSIZE7_INT3C     21*15*15
#define GOUTSIZE8_INT3C     21*21*21
#define GOUTSIZE9_INT3C     28*28*21
#define MAX_GOUTSIZE_INT3C  GOUTSIZE9_INT3C
/*
// s->1, p->3, d->5, f->7, g->9, h->11, i->13
#define GOUTSIZE1_INT3C_SPH      3
#define GOUTSIZE2_INT3C_SPH      3*3*3
#define GOUTSIZE3_INT3C_SPH      5*5*3
#define GOUTSIZE4_INT3C_SPH      7*5*5
#define GOUTSIZE5_INT3C_SPH      7*7*7
#define GOUTSIZE6_INT3C_SPH      9*9*7
#define GOUTSIZE7_INT3C_SPH      11*9*9
#define GOUTSIZE8_INT3C_SPH      11*11*11
#define GOUTSIZE9_INT3C_SPH      13*13*11
*/
// nf=10^4
#define NFffff       10000
#define NFhgg        GOUTSIZE7_INT3C

#ifndef HAVE_DEFINED_GINTENVVAS_H
#define HAVE_DEFINED_GINTENVVAS_H
typedef struct {
        int16_t i_l;
        int16_t j_l;
        int16_t k_l;
        int16_t l_l;
        int16_t li_ceil;
        int16_t lj_ceil;
        int16_t lk_ceil;
        int16_t ll_ceil;
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
        int stride_i;
        int stride_j;
        int stride_k;
        int stride_l;
        double fac;

        int nprim_ij;
        int nprim_kl;
        int16_t *idx;
        double *uw;
        double omega;
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
        double *log_q_ij;
        double *log_q_kl;
        double log_cutoff;
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
    double *a1;
    double *a2;
} BasisProdCache;


typedef struct {
        int stride_j;
        int stride_k;
        int stride_l;
        int ao_offsets_i;
        int ao_offsets_j;
        int ao_offsets_k;
        int ao_offsets_l;
        int nao;
        double *data;
} ERITensor;

typedef struct {
        int nao;
        int naux;
        int n_dm;
        double *vj;
        double *vk;
        double* __restrict__ dm;
        double* __restrict__ rhoj;
        double* __restrict__ rhok;
        int ao_offsets_i;
        int ao_offsets_j;
        int ao_offsets_k;
        int ao_offsets_l;
        double* __restrict__ dm_sh;
        int nshls;
} JKMatrix;

typedef void (*FPtr_CPUkernel_jk)(double *g, double **dm, double **v,
                                  int *shls, GINTEnvVars *envs, int *ibuf);

#endif
