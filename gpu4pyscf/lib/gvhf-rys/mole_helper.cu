/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include "gvhf-rys/vhf.cuh"

#define THREADS         256
#define TILE_X          16
#define TILE_Y          16
#define ROW_BLKSIZE     (TILE_Y*16)
#define COL_BLKSIZE     4096
#define SHM_BLKSIZE     6
#define NPRIM_MAX       32
#define PTR_PBAS_IDX    4

static __global__
void bra_sorted2cart_kernel(double *out, double *input, double *recontract_coef,
                            int *recontract_bas, int *pbas_idx_recontraction,
                            int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    int thread_id = item.get_local_id(2);
    int col0 = item.get_group(2) * COL_BLKSIZE;
    int c_bas_id = item.get_group(1);
    int count = item.get_group(0);
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX]>(thread_block);
    #else
    int thread_id = threadIdx.x;
    int col0 = blockIdx.x * COL_BLKSIZE;
    int c_bas_id = blockIdx.y;
    int count = blockIdx.z;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX];
    #endif
    int col1 = min(col0 + COL_BLKSIZE, ncol);
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    size_t stride = nfi * ncol;
    double *pgto = input + count * p_nao * ncol;
    if (thread_id < nprim) {
        int p_bas_id = pbas_idx[thread_id];
        p_ao_offsets[thread_id] = p_ao_loc[p_bas_id];
    }
    __syncthreads();

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        size_t c_off = (count * c_nao + c_ao_loc[c_bas_id] + ctr0*nfi) * ncol;
        for (int col_id = col0+thread_id; col_id < col1; col_id += THREADS) {
            for (int i = 0; i < nfi; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = 0;
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double s = pgto[(p_ao_offsets[ip]+i)*ncol+col_id];
                    double *c = coef + ctr0*nprim + ip;
                    for (int n = 0; n < sub_nctr; ++n) {
                        cval[n*THREADS+thread_id] += s * c[n*nprim];
                    }
                }
                double *cgto = out + c_off + i * ncol + col_id;
                for (int n = 0; n < sub_nctr; ++n) {
                    cgto[n*stride] = cval[n*THREADS+thread_id];
                }
            }
        }
    }
}

static __global__
void bra_cart2sorted_kernel(double *out, double *input, double *recontract_coef,
                            int *recontract_bas, int *pbas_idx_recontraction,
                            int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    int thread_id = item.get_local_id(2);
    int col0 = item.get_group(2) * COL_BLKSIZE;
    int c_bas_id = item.get_group(1);
    int count = item.get_group(0);
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX]>(thread_block);
    #else
    int thread_id = threadIdx.x;
    int col0 = blockIdx.x * COL_BLKSIZE;
    int c_bas_id = blockIdx.y;
    int count = blockIdx.z;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX];
    #endif
    int col1 = min(col0 + COL_BLKSIZE, ncol);
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    size_t stride = nfi * ncol;
    double *pgto = out + count * p_nao * ncol;
    if (thread_id < nprim) {
        int p_bas_id = pbas_idx[thread_id];
        p_ao_offsets[thread_id] = p_ao_loc[p_bas_id];
    }
    __syncthreads();

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        size_t c_off = (count * c_nao + c_ao_loc[c_bas_id] + ctr0*nfi) * ncol;
        for (int i = 0; i < nfi; ++i) {
            for (int col_id = col0+thread_id; col_id < col1; col_id += THREADS) {
                double *cgto = input + c_off + i * ncol + col_id;
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = cgto[n*stride];
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double *c = coef + ctr0*nprim + ip;
                    double s = cval[thread_id] * c[0];
                    for (int n = 1; n < sub_nctr; ++n) {
                        s += cval[n*THREADS+thread_id] * c[n*nprim];
                    }
                    pgto[(p_ao_offsets[ip]+i)*ncol+col_id] += s;
                    //atomicAdd(pgto+(p_ao_offsets[ip]+i)*ncol+col_id,  s);
                }
            }
        }
    }
}

static __global__
void bra_sorted2sph_kernel(double *out, double *input, double *recontract_coef,
                           int *recontract_bas, int *pbas_idx_recontraction,
                           int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol)

{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    int thread_id = item.get_local_id(2);
    int col0 = item.get_group(2) * COL_BLKSIZE;
    int c_bas_id = item.get_group(1);
    int count = item.get_group(0);
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX]>(thread_block);
    #else
    int thread_id = threadIdx.x;
    int col0 = blockIdx.x * COL_BLKSIZE;
    int c_bas_id = blockIdx.y;
    int count = blockIdx.z;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX];
    #endif
    int col1 = min(col0 + COL_BLKSIZE, ncol);
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int di = li * 2 + 1;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    double *pgto = input + count * p_nao * ncol;
    if (thread_id < nprim) {
        int p_bas_id = pbas_idx[thread_id];
        p_ao_offsets[thread_id] = p_ao_loc[p_bas_id];
    }
    __syncthreads();

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        for (int col_id = col0+thread_id; col_id < col1; col_id += THREADS) {
            size_t c_off = (count * c_nao + c_ao_loc[c_bas_id] + ctr0*di) * ncol + col_id;
            for (int i = 0; i < nfi; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = 0;
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double s = pgto[(p_ao_offsets[ip]+i)*ncol+col_id];
                    double *c = coef + ctr0*nprim + ip;
                    for (int n = 0; n < sub_nctr; ++n) {
                        cval[n*THREADS+thread_id] += s * c[n*nprim];
                    }
                }
                for (int n = 0; n < n_ctr; ++n) {
                    double *cgto = out + c_off + n * di * ncol;
                    switch (i+nfi*li/3) {
                    case 0: { // l=0, i=0
                        cgto[0*ncol] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 1: { // l=1, i=0
                        cgto[0*ncol] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 2: { // l=1, i=1
                        cgto[1*ncol] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 3: { // l=1, i=2
                        cgto[2*ncol] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 4: { // l=2, i=0
                        cgto[2*ncol] += -0.315391565252520002 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.546274215296039535 * cval[n*THREADS+thread_id];
                    } break;
                    case 5: { // l=2, i=1
                        cgto[0*ncol] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 6: { // l=2, i=2
                        cgto[3*ncol] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 7: { // l=2, i=3
                        cgto[2*ncol] += -0.315391565252520002 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += -0.546274215296039535 * cval[n*THREADS+thread_id];
                    } break;
                    case 8: { // l=2, i=4
                        cgto[1*ncol] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 9: { // l=2, i=5
                        cgto[2*ncol] += 0.630783130505040012 * cval[n*THREADS+thread_id];
                    } break;
                    case 10: { // l=3, i=0
                        cgto[4*ncol] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += 0.590043589926643510 * cval[n*THREADS+thread_id];
                    } break;
                    case 11: { // l=3, i=1
                        cgto[0*ncol] += 1.770130769779930531 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                    } break;
                    case 12: { // l=3, i=2
                        cgto[3*ncol] += -1.119528997770346170 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += 1.445305721320277020 * cval[n*THREADS+thread_id];
                    } break;
                    case 13: { // l=3, i=3
                        cgto[4*ncol] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += -1.770130769779930530 * cval[n*THREADS+thread_id];
                    } break;
                    case 14: { // l=3, i=4
                        cgto[1*ncol] += 2.890611442640554055 * cval[n*THREADS+thread_id];
                    } break;
                    case 15: { // l=3, i=5
                        cgto[4*ncol] += 1.828183197857862944 * cval[n*THREADS+thread_id];
                    } break;
                    case 16: { // l=3, i=6
                        cgto[0*ncol] += -0.590043589926643510 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                    } break;
                    case 17: { // l=3, i=7
                        cgto[3*ncol] += -1.119528997770346170 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += -1.445305721320277020 * cval[n*THREADS+thread_id];
                    } break;
                    case 18: { // l=3, i=8
                        cgto[2*ncol] += 1.828183197857862944 * cval[n*THREADS+thread_id];
                    } break;
                    case 19: { // l=3, i=9
                        cgto[3*ncol] += 0.746352665180230782 * cval[n*THREADS+thread_id];
                    } break;
                    case 20: { // l=4, i=0
                        cgto[4*ncol] += 0.317356640745612911 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += -0.473087347878780002 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 0.625835735449176134 * cval[n*THREADS+thread_id];
                    } break;
                    case 21: { // l=4, i=1
                        cgto[0*ncol] += 2.503342941796704538 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -0.946174695757560014 * cval[n*THREADS+thread_id];
                    } break;
                    case 22: { // l=4, i=2
                        cgto[5*ncol] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += 1.770130769779930531 * cval[n*THREADS+thread_id];
                    } break;
                    case 23: { // l=4, i=3
                        cgto[4*ncol] += 0.634713281491225822 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -3.755014412695056800 * cval[n*THREADS+thread_id];
                    } break;
                    case 24: { // l=4, i=4
                        cgto[1*ncol] += 5.310392309339791593 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                    } break;
                    case 25: { // l=4, i=5
                        cgto[4*ncol] += -2.538853125964903290 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += 2.838524087272680054 * cval[n*THREADS+thread_id];
                    } break;
                    case 26: { // l=4, i=6
                        cgto[0*ncol] += -2.503342941796704530 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -0.946174695757560014 * cval[n*THREADS+thread_id];
                    } break;
                    case 27: { // l=4, i=7
                        cgto[5*ncol] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += -5.310392309339791590 * cval[n*THREADS+thread_id];
                    } break;
                    case 28: { // l=4, i=8
                        cgto[2*ncol] += 5.677048174545360108 * cval[n*THREADS+thread_id];
                    } break;
                    case 29: { // l=4, i=9
                        cgto[5*ncol] += 2.676186174229156671 * cval[n*THREADS+thread_id];
                    } break;
                    case 30: { // l=4, i=10
                        cgto[4*ncol] += 0.317356640745612911 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += 0.473087347878780009 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 0.625835735449176134 * cval[n*THREADS+thread_id];
                    } break;
                    case 31: { // l=4, i=11
                        cgto[1*ncol] += -1.770130769779930530 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                    } break;
                    case 32: { // l=4, i=12
                        cgto[4*ncol] += -2.538853125964903290 * cval[n*THREADS+thread_id];
                        cgto[6*ncol] += -2.838524087272680050 * cval[n*THREADS+thread_id];
                    } break;
                    case 33: { // l=4, i=13
                        cgto[3*ncol] += 2.676186174229156671 * cval[n*THREADS+thread_id];
                    } break;
                    case 34: { // l=4, i=14
                        cgto[4*ncol] += 0.846284375321634430 * cval[n*THREADS+thread_id];
                    } break;
                    case 35: { // l=5, i=0
                        cgto[6*ncol] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -0.489238299435250389 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 0.656382056840170102 * cval[n*THREADS+thread_id];
                    } break;
                    case 36: { // l=5, i=1
                        cgto[0*ncol] += 3.281910284200850514 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -1.467714898305751160 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                    } break;
                    case 37: { // l=5, i=2
                        cgto[5*ncol] += 1.754254836801353946 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += -2.396768392486661870 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += 2.075662314881041278 * cval[n*THREADS+thread_id];
                    } break;
                    case 38: { // l=5, i=3
                        cgto[6*ncol] += 0.905893302391393842 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 0.978476598870500775 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += -6.563820568401701020 * cval[n*THREADS+thread_id];
                    } break;
                    case 39: { // l=5, i=4
                        cgto[1*ncol] += 8.302649259524165115 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 40: { // l=5, i=5
                        cgto[6*ncol] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 3.913906395482003101 * cval[n*THREADS+thread_id];
                    } break;
                    case 41: { // l=5, i=6
                        cgto[0*ncol] += -6.563820568401701020 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -0.978476598870500779 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.905893302391393842 * cval[n*THREADS+thread_id];
                    } break;
                    case 42: { // l=5, i=7
                        cgto[5*ncol] += 3.508509673602707893 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += -12.453973889286247600 * cval[n*THREADS+thread_id];
                    } break;
                    case 43: { // l=5, i=8
                        cgto[2*ncol] += 11.741719186446009300 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                    } break;
                    case 44: { // l=5, i=9
                        cgto[5*ncol] += -4.678012898136943850 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += 4.793536784973323755 * cval[n*THREADS+thread_id];
                    } break;
                    case 45: { // l=5, i=10
                        cgto[6*ncol] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 1.467714898305751163 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 3.281910284200850514 * cval[n*THREADS+thread_id];
                    } break;
                    case 46: { // l=5, i=11
                        cgto[1*ncol] += -8.302649259524165110 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 47: { // l=5, i=12
                        cgto[6*ncol] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -11.741719186446009300 * cval[n*THREADS+thread_id];
                    } break;
                    case 48: { // l=5, i=13
                        cgto[3*ncol] += 9.587073569946647510 * cval[n*THREADS+thread_id];
                    } break;
                    case 49: { // l=5, i=14
                        cgto[6*ncol] += 3.623573209565575370 * cval[n*THREADS+thread_id];
                    } break;
                    case 50: { // l=5, i=15
                        cgto[0*ncol] += 0.656382056840170102 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += 0.489238299435250387 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                    } break;
                    case 51: { // l=5, i=16
                        cgto[5*ncol] += 1.754254836801353946 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += 2.396768392486661877 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += 2.075662314881041278 * cval[n*THREADS+thread_id];
                    } break;
                    case 52: { // l=5, i=17
                        cgto[2*ncol] += -3.913906395482003100 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                    } break;
                    case 53: { // l=5, i=18
                        cgto[5*ncol] += -4.678012898136943850 * cval[n*THREADS+thread_id];
                        cgto[7*ncol] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 54: { // l=5, i=19
                        cgto[4*ncol] += 3.623573209565575370 * cval[n*THREADS+thread_id];
                    } break;
                    case 55: { // l=5, i=20
                        cgto[5*ncol] += 0.935602579627388771 * cval[n*THREADS+thread_id];
                    } break;
                    case 56: { // l=6, i=0
                        cgto[6*ncol] += -0.3178460113381421 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += -0.5045649007287241 * cval[n*THREADS+thread_id];
                        cgto[12*ncol] += 0.6831841051919144 * cval[n*THREADS+thread_id];
                    } break;
                    case 57: { // l=6, i=1
                        cgto[0*ncol] += 4.0991046311514863 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += -2.0182596029148963 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.9212052595149236 * cval[n*THREADS+thread_id];
                    } break;
                    case 58: { // l=6, i=2
                        cgto[7*ncol] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += -2.7636157785447706 * cval[n*THREADS+thread_id];
                        cgto[11*ncol] += 2.3666191622317525 * cval[n*THREADS+thread_id];
                    } break;
                    case 59: { // l=6, i=3
                        cgto[6*ncol] += -0.9535380340144264 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 2.5228245036436201 * cval[n*THREADS+thread_id];
                        cgto[12*ncol] += -10.2477615778787161 * cval[n*THREADS+thread_id];
                    } break;
                    case 60: { // l=6, i=4
                        cgto[1*ncol] += 11.8330958111587634 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -8.2908473356343109 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                    } break;
                    case 61: { // l=6, i=5
                        cgto[6*ncol] += 5.7212282040865583 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 5.0456490072872420 * cval[n*THREADS+thread_id];
                    } break;
                    case 62: { // l=6, i=6
                        cgto[0*ncol] += -13.6636821038382887 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 1.8424105190298472 * cval[n*THREADS+thread_id];
                    } break;
                    case 63: { // l=6, i=7
                        cgto[7*ncol] += 5.8262136251873136 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += 5.5272315570895412 * cval[n*THREADS+thread_id];
                        cgto[11*ncol] += -23.6661916223175268 * cval[n*THREADS+thread_id];
                    } break;
                    case 64: { // l=6, i=8
                        cgto[2*ncol] += 20.1825960291489679 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += -14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 65: { // l=6, i=9
                        cgto[7*ncol] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 66: { // l=6, i=10
                        cgto[6*ncol] += -0.9535380340144264 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 2.5228245036436201 * cval[n*THREADS+thread_id];
                        cgto[12*ncol] += 10.2477615778787161 * cval[n*THREADS+thread_id];
                    } break;
                    case 67: { // l=6, i=11
                        cgto[1*ncol] += -23.6661916223175268 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += -5.5272315570895412 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += 5.8262136251873136 * cval[n*THREADS+thread_id];
                    } break;
                    case 68: { // l=6, i=12
                        cgto[6*ncol] += 11.4424564081731166 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += -30.2738940437234518 * cval[n*THREADS+thread_id];
                    } break;
                    case 69: { // l=6, i=13
                        cgto[3*ncol] += 22.1089262283581647 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                    } break;
                    case 70: { // l=6, i=14
                        cgto[6*ncol] += -7.6283042721154111 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 71: { // l=6, i=15
                        cgto[0*ncol] += 4.0991046311514863 * cval[n*THREADS+thread_id];
                        cgto[2*ncol] += 2.0182596029148963 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += 0.9212052595149236 * cval[n*THREADS+thread_id];
                    } break;
                    case 72: { // l=6, i=16
                        cgto[7*ncol] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += 8.2908473356343109 * cval[n*THREADS+thread_id];
                        cgto[11*ncol] += 11.8330958111587634 * cval[n*THREADS+thread_id];
                    } break;
                    case 73: { // l=6, i=17
                        cgto[2*ncol] += -20.1825960291489679 * cval[n*THREADS+thread_id];
                        cgto[4*ncol] += -14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 74: { // l=6, i=18
                        cgto[7*ncol] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                        cgto[9*ncol] += -22.1089262283581647 * cval[n*THREADS+thread_id];
                    } break;
                    case 75: { // l=6, i=19
                        cgto[4*ncol] += 14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 76: { // l=6, i=20
                        cgto[7*ncol] += 4.6609709001498505 * cval[n*THREADS+thread_id];
                    } break;
                    case 77: { // l=6, i=21
                        cgto[6*ncol] += -0.3178460113381421 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += -0.5045649007287241 * cval[n*THREADS+thread_id];
                        cgto[12*ncol] += -0.6831841051919144 * cval[n*THREADS+thread_id];
                    } break;
                    case 78: { // l=6, i=22
                        cgto[1*ncol] += 2.3666191622317525 * cval[n*THREADS+thread_id];
                        cgto[3*ncol] += 2.7636157785447706 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                    } break;
                    case 79: { // l=6, i=23
                        cgto[6*ncol] += 5.7212282040865583 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[10*ncol] += 5.0456490072872420 * cval[n*THREADS+thread_id];
                    } break;
                    case 80: { // l=6, i=24
                        cgto[3*ncol] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[5*ncol] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                    } break;
                    case 81: { // l=6, i=25
                        cgto[6*ncol] += -7.6283042721154111 * cval[n*THREADS+thread_id];
                        cgto[8*ncol] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 82: { // l=6, i=26
                        cgto[5*ncol] += 4.6609709001498505 * cval[n*THREADS+thread_id];
                    } break;
                    case 83: { // l=6, i=27
                        cgto[6*ncol] += 1.0171072362820548 * cval[n*THREADS+thread_id];
                    } break;
                    }
                }
            }
        }
    }
}

static __global__
void bra_sph2sorted_kernel(double *out, double *input, double *recontract_coef,
                           int *recontract_bas, int *pbas_idx_recontraction,
                           int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<3>();
    int thread_id = item.get_local_id(2);
    int col0 = item.get_group(2) * COL_BLKSIZE;
    int c_bas_id = item.get_group(1);
    int count = item.get_group(0);
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX]>(thread_block);
    #else
    int thread_id = threadIdx.x;
    int col0 = blockIdx.x * COL_BLKSIZE;
    int c_bas_id = blockIdx.y;
    int count = blockIdx.z;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX];
    #endif
    int col1 = min(col0 + COL_BLKSIZE, ncol);
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int di = li * 2 + 1;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    size_t stride = di * ncol;
    double *pgto = out + count * p_nao * ncol;
    if (thread_id < nprim) {
        int p_bas_id = pbas_idx[thread_id];
        p_ao_offsets[thread_id] = p_ao_loc[p_bas_id];
    }
    __syncthreads();

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        size_t c_off = (count * c_nao + c_ao_loc[c_bas_id] + ctr0*di) * ncol;
        for (int i = 0; i < di; ++i) {
            for (int col_id = col0+thread_id; col_id < col1; col_id += THREADS) {
                double *cgto = input + c_off + i * ncol + col_id;
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = cgto[n*stride];
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double *c = coef + ctr0*nprim + ip;
                    double s = cval[thread_id] * c[0];
                    for (int n = 1; n < sub_nctr; ++n) {
                        s += cval[n*THREADS+thread_id] * c[n*nprim];
                    }
                    int p_off = p_ao_offsets[ip] * ncol + col_id;
                    switch (li*li+i) {
                    case 0: { // l=0, m=0
                        pgto[0*ncol+p_off] += 1 * s;
                    } break;
                    case 1: { // l=1, m=0
                        pgto[0*ncol+p_off] += 1 * s;
                    } break;
                    case 2: { // l=1, m=1
                        pgto[1*ncol+p_off] += 1 * s;
                    } break;
                    case 3: { // l=1, m=2
                        pgto[2*ncol+p_off] += 1 * s;
                    } break;
                    case 4: { // l=2, m=0
                        pgto[1*ncol+p_off] += 1.092548430592079070 * s;
                    } break;
                    case 5: { // l=2, m=1
                        pgto[4*ncol+p_off] += 1.092548430592079070 * s;
                    } break;
                    case 6: { // l=2, m=2
                        pgto[0*ncol+p_off] += -0.315391565252520002 * s;
                        pgto[3*ncol+p_off] += -0.315391565252520002 * s;
                        pgto[5*ncol+p_off] += 0.630783130505040012 * s;
                    } break;
                    case 7: { // l=2, m=3
                        pgto[2*ncol+p_off] += 1.092548430592079070 * s;
                    } break;
                    case 8: { // l=2, m=4
                        pgto[0*ncol+p_off] += 0.546274215296039535 * s;
                        pgto[3*ncol+p_off] += -0.546274215296039535 * s;
                    } break;
                    case 9: { // l=3, m=0
                        pgto[1*ncol+p_off] += 1.770130769779930531 * s;
                        pgto[6*ncol+p_off] += -0.590043589926643510 * s;
                    } break;
                    case 10: { // l=3, m=1
                        pgto[4*ncol+p_off] += 2.890611442640554055 * s;
                    } break;
                    case 11: { // l=3, m=2
                        pgto[1*ncol+p_off] += -0.457045799464465739 * s;
                        pgto[6*ncol+p_off] += -0.457045799464465739 * s;
                        pgto[8*ncol+p_off] += 1.828183197857862944 * s;
                    } break;
                    case 12: { // l=3, m=3
                        pgto[2*ncol+p_off] += -1.119528997770346170 * s;
                        pgto[7*ncol+p_off] += -1.119528997770346170 * s;
                        pgto[9*ncol+p_off] += 0.746352665180230782 * s;
                    } break;
                    case 13: { // l=3, m=4
                        pgto[0*ncol+p_off] += -0.457045799464465739 * s;
                        pgto[3*ncol+p_off] += -0.457045799464465739 * s;
                        pgto[5*ncol+p_off] += 1.828183197857862944 * s;
                    } break;
                    case 14: { // l=3, m=5
                        pgto[2*ncol+p_off] += 1.445305721320277020 * s;
                        pgto[7*ncol+p_off] += -1.445305721320277020 * s;
                    } break;
                    case 15: { // l=3, m=6
                        pgto[0*ncol+p_off] += 0.590043589926643510 * s;
                        pgto[3*ncol+p_off] += -1.770130769779930530 * s;
                    } break;
                    case 16: { // l=4, m=0
                        pgto[1*ncol+p_off] += 2.503342941796704538 * s;
                        pgto[6*ncol+p_off] += -2.503342941796704530 * s;
                    } break;
                    case 17: { // l=4, m=1
                        pgto[4*ncol+p_off] += 5.310392309339791593 * s;
                        pgto[11*ncol+p_off] += -1.770130769779930530 * s;
                    } break;
                    case 18: { // l=4, m=2
                        pgto[1*ncol+p_off] += -0.946174695757560014 * s;
                        pgto[6*ncol+p_off] += -0.946174695757560014 * s;
                        pgto[8*ncol+p_off] += 5.677048174545360108 * s;
                    } break;
                    case 19: { // l=4, m=3
                        pgto[4*ncol+p_off] += -2.007139630671867500 * s;
                        pgto[11*ncol+p_off] += -2.007139630671867500 * s;
                        pgto[13*ncol+p_off] += 2.676186174229156671 * s;
                    } break;
                    case 20: { // l=4, m=4
                        pgto[0*ncol+p_off] += 0.317356640745612911 * s;
                        pgto[3*ncol+p_off] += 0.634713281491225822 * s;
                        pgto[5*ncol+p_off] += -2.538853125964903290 * s;
                        pgto[10*ncol+p_off] += 0.317356640745612911 * s;
                        pgto[12*ncol+p_off] += -2.538853125964903290 * s;
                        pgto[14*ncol+p_off] += 0.846284375321634430 * s;
                    } break;
                    case 21: { // l=4, m=5
                        pgto[2*ncol+p_off] += -2.007139630671867500 * s;
                        pgto[7*ncol+p_off] += -2.007139630671867500 * s;
                        pgto[9*ncol+p_off] += 2.676186174229156671 * s;
                    } break;
                    case 22: { // l=4, m=6
                        pgto[0*ncol+p_off] += -0.473087347878780002 * s;
                        pgto[5*ncol+p_off] += 2.838524087272680054 * s;
                        pgto[10*ncol+p_off] += 0.473087347878780009 * s;
                        pgto[12*ncol+p_off] += -2.838524087272680050 * s;
                    } break;
                    case 23: { // l=4, m=7
                        pgto[2*ncol+p_off] += 1.770130769779930531 * s;
                        pgto[7*ncol+p_off] += -5.310392309339791590 * s;
                    } break;
                    case 24: { // l=4, m=8
                        pgto[0*ncol+p_off] += 0.625835735449176134 * s;
                        pgto[3*ncol+p_off] += -3.755014412695056800 * s;
                        pgto[10*ncol+p_off] += 0.625835735449176134 * s;
                    } break;
                    case 25: { // l=5, m=0
                        pgto[1*ncol+p_off] += 3.281910284200850514 * s;
                        pgto[6*ncol+p_off] += -6.563820568401701020 * s;
                        pgto[15*ncol+p_off] += 0.656382056840170102 * s;
                    } break;
                    case 26: { // l=5, m=1
                        pgto[4*ncol+p_off] += 8.302649259524165115 * s;
                        pgto[11*ncol+p_off] += -8.302649259524165110 * s;
                    } break;
                    case 27: { // l=5, m=2
                        pgto[1*ncol+p_off] += -1.467714898305751160 * s;
                        pgto[6*ncol+p_off] += -0.978476598870500779 * s;
                        pgto[8*ncol+p_off] += 11.741719186446009300 * s;
                        pgto[15*ncol+p_off] += 0.489238299435250387 * s;
                        pgto[17*ncol+p_off] += -3.913906395482003100 * s;
                    } break;
                    case 28: { // l=5, m=3
                        pgto[4*ncol+p_off] += -4.793536784973323750 * s;
                        pgto[11*ncol+p_off] += -4.793536784973323750 * s;
                        pgto[13*ncol+p_off] += 9.587073569946647510 * s;
                    } break;
                    case 29: { // l=5, m=4
                        pgto[1*ncol+p_off] += 0.452946651195696921 * s;
                        pgto[6*ncol+p_off] += 0.905893302391393842 * s;
                        pgto[8*ncol+p_off] += -5.435359814348363050 * s;
                        pgto[15*ncol+p_off] += 0.452946651195696921 * s;
                        pgto[17*ncol+p_off] += -5.435359814348363050 * s;
                        pgto[19*ncol+p_off] += 3.623573209565575370 * s;
                    } break;
                    case 30: { // l=5, m=5
                        pgto[2*ncol+p_off] += 1.754254836801353946 * s;
                        pgto[7*ncol+p_off] += 3.508509673602707893 * s;
                        pgto[9*ncol+p_off] += -4.678012898136943850 * s;
                        pgto[16*ncol+p_off] += 1.754254836801353946 * s;
                        pgto[18*ncol+p_off] += -4.678012898136943850 * s;
                        pgto[20*ncol+p_off] += 0.935602579627388771 * s;
                    } break;
                    case 31: { // l=5, m=6
                        pgto[0*ncol+p_off] += 0.452946651195696921 * s;
                        pgto[3*ncol+p_off] += 0.905893302391393842 * s;
                        pgto[5*ncol+p_off] += -5.435359814348363050 * s;
                        pgto[10*ncol+p_off] += 0.452946651195696921 * s;
                        pgto[12*ncol+p_off] += -5.435359814348363050 * s;
                        pgto[14*ncol+p_off] += 3.623573209565575370 * s;
                    } break;
                    case 32: { // l=5, m=7
                        pgto[2*ncol+p_off] += -2.396768392486661870 * s;
                        pgto[9*ncol+p_off] += 4.793536784973323755 * s;
                        pgto[16*ncol+p_off] += 2.396768392486661877 * s;
                        pgto[18*ncol+p_off] += -4.793536784973323750 * s;
                    } break;
                    case 33: { // l=5, m=8
                        pgto[0*ncol+p_off] += -0.489238299435250389 * s;
                        pgto[3*ncol+p_off] += 0.978476598870500775 * s;
                        pgto[5*ncol+p_off] += 3.913906395482003101 * s;
                        pgto[10*ncol+p_off] += 1.467714898305751163 * s;
                        pgto[12*ncol+p_off] += -11.741719186446009300 * s;
                    } break;
                    case 34: { // l=5, m=9
                        pgto[2*ncol+p_off] += 2.075662314881041278 * s;
                        pgto[7*ncol+p_off] += -12.453973889286247600 * s;
                        pgto[16*ncol+p_off] += 2.075662314881041278 * s;
                    } break;
                    case 35: { // l=5, m=10
                        pgto[0*ncol+p_off] += 0.656382056840170102 * s;
                        pgto[3*ncol+p_off] += -6.563820568401701020 * s;
                        pgto[10*ncol+p_off] += 3.281910284200850514 * s;
                    } break;
                    case 36: { // l=6, m=0
                        pgto[1*ncol+p_off] += 4.0991046311514863 * s;
                        pgto[6*ncol+p_off] += -13.6636821038382887 * s;
                        pgto[15*ncol+p_off] += 4.0991046311514863 * s;
                    } break;
                    case 37: { // l=6, m=1
                        pgto[4*ncol+p_off] += 11.8330958111587634 * s;
                        pgto[11*ncol+p_off] += -23.6661916223175268 * s;
                        pgto[22*ncol+p_off] += 2.3666191622317525 * s;
                    } break;
                    case 38: { // l=6, m=2
                        pgto[1*ncol+p_off] += -2.0182596029148963 * s;
                        pgto[8*ncol+p_off] += 20.1825960291489679 * s;
                        pgto[15*ncol+p_off] += 2.0182596029148963 * s;
                        pgto[17*ncol+p_off] += -20.1825960291489679 * s;
                    } break;
                    case 39: { // l=6, m=3
                        pgto[4*ncol+p_off] += -8.2908473356343109 * s;
                        pgto[11*ncol+p_off] += -5.5272315570895412 * s;
                        pgto[13*ncol+p_off] += 22.1089262283581647 * s;
                        pgto[22*ncol+p_off] += 2.7636157785447706 * s;
                        pgto[24*ncol+p_off] += -7.3696420761193888 * s;
                    } break;
                    case 40: { // l=6, m=4
                        pgto[1*ncol+p_off] += 0.9212052595149236 * s;
                        pgto[6*ncol+p_off] += 1.8424105190298472 * s;
                        pgto[8*ncol+p_off] += -14.7392841522387776 * s;
                        pgto[15*ncol+p_off] += 0.9212052595149236 * s;
                        pgto[17*ncol+p_off] += -14.7392841522387776 * s;
                        pgto[19*ncol+p_off] += 14.7392841522387776 * s;
                    } break;
                    case 41: { // l=6, m=5
                        pgto[4*ncol+p_off] += 2.9131068125936568 * s;
                        pgto[11*ncol+p_off] += 5.8262136251873136 * s;
                        pgto[13*ncol+p_off] += -11.6524272503746271 * s;
                        pgto[22*ncol+p_off] += 2.9131068125936568 * s;
                        pgto[24*ncol+p_off] += -11.6524272503746271 * s;
                        pgto[26*ncol+p_off] += 4.6609709001498505 * s;
                    } break;
                    case 42: { // l=6, m=6
                        pgto[0*ncol+p_off] += -0.3178460113381421 * s;
                        pgto[3*ncol+p_off] += -0.9535380340144264 * s;
                        pgto[5*ncol+p_off] += 5.7212282040865583 * s;
                        pgto[10*ncol+p_off] += -0.9535380340144264 * s;
                        pgto[12*ncol+p_off] += 11.4424564081731166 * s;
                        pgto[14*ncol+p_off] += -7.6283042721154111 * s;
                        pgto[21*ncol+p_off] += -0.3178460113381421 * s;
                        pgto[23*ncol+p_off] += 5.7212282040865583 * s;
                        pgto[25*ncol+p_off] += -7.6283042721154111 * s;
                        pgto[27*ncol+p_off] += 1.0171072362820548 * s;
                    } break;
                    case 43: { // l=6, m=7
                        pgto[2*ncol+p_off] += 2.9131068125936568 * s;
                        pgto[7*ncol+p_off] += 5.8262136251873136 * s;
                        pgto[9*ncol+p_off] += -11.6524272503746271 * s;
                        pgto[16*ncol+p_off] += 2.9131068125936568 * s;
                        pgto[18*ncol+p_off] += -11.6524272503746271 * s;
                        pgto[20*ncol+p_off] += 4.6609709001498505 * s;
                    } break;
                    case 44: { // l=6, m=8
                        pgto[0*ncol+p_off] += 0.4606026297574618 * s;
                        pgto[3*ncol+p_off] += 0.4606026297574618 * s;
                        pgto[5*ncol+p_off] += -7.3696420761193888 * s;
                        pgto[10*ncol+p_off] += -0.4606026297574618 * s;
                        pgto[14*ncol+p_off] += 7.3696420761193888 * s;
                        pgto[21*ncol+p_off] += -0.4606026297574618 * s;
                        pgto[23*ncol+p_off] += 7.3696420761193888 * s;
                        pgto[25*ncol+p_off] += -7.3696420761193888 * s;
                    } break;
                    case 45: { // l=6, m=9
                        pgto[2*ncol+p_off] += -2.7636157785447706 * s;
                        pgto[7*ncol+p_off] += 5.5272315570895412 * s;
                        pgto[9*ncol+p_off] += 7.3696420761193888 * s;
                        pgto[16*ncol+p_off] += 8.2908473356343109 * s;
                        pgto[18*ncol+p_off] += -22.1089262283581647 * s;
                    } break;
                    case 46: { // l=6, m=10
                        pgto[0*ncol+p_off] += -0.5045649007287241 * s;
                        pgto[3*ncol+p_off] += 2.5228245036436201 * s;
                        pgto[5*ncol+p_off] += 5.0456490072872420 * s;
                        pgto[10*ncol+p_off] += 2.5228245036436201 * s;
                        pgto[12*ncol+p_off] += -30.2738940437234518 * s;
                        pgto[21*ncol+p_off] += -0.5045649007287241 * s;
                        pgto[23*ncol+p_off] += 5.0456490072872420 * s;
                    } break;
                    case 47: { // l=6, m=11
                        pgto[2*ncol+p_off] += 2.3666191622317525 * s;
                        pgto[7*ncol+p_off] += -23.6661916223175268 * s;
                        pgto[16*ncol+p_off] += 11.8330958111587634 * s;
                    } break;
                    case 48: { // l=6, m=12
                        pgto[0*ncol+p_off] += 0.6831841051919144 * s;
                        pgto[3*ncol+p_off] += -10.2477615778787161 * s;
                        pgto[10*ncol+p_off] += 10.2477615778787161 * s;
                        pgto[21*ncol+p_off] += -0.6831841051919144 * s;
                    } break;
                    }
                }
            }
        }
    }
}

static __global__
void ket_sorted2cart_kernel(double *out, double *input, double *recontract_coef,
                            int *recontract_bas, int *pbas_idx_recontraction,
                            int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int row0 = item.get_group(1) * ROW_BLKSIZE;
    int c_bas_id = item.get_group(0) * TILE_X + tx;
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX*TILE_X] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX*TILE_X]>(thread_block);
    #else
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.x * ROW_BLKSIZE;
    int c_bas_id = blockIdx.y * TILE_X + tx;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX*TILE_X];
    #endif
    int thread_id = ty * TILE_X + tx;
    int row1 = min(row0 + ROW_BLKSIZE, nrow);
    int valid = c_bas_id < nbas;
    if (!valid) {
        c_bas_id = 0;
    }
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    for (int ip = ty; ip < nprim; ip += TILE_Y) {
        int p_bas_id = pbas_idx[ip];
        p_ao_offsets[ip*TILE_X+tx] = p_ao_loc[p_bas_id];
    }
    __syncthreads();
    if (!valid) {
        return;
    }

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        for (int row_id = row0+ty; row_id < row1; row_id += TILE_Y) {
            double *cgto = out   + row_id*c_nao + c_ao_loc[c_bas_id] + ctr0*nfi;
            double *pgto = input + row_id*p_nao;
            for (int i = 0; i < nfi; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = 0;
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double s = pgto[p_ao_offsets[ip*TILE_X+tx]+i];
                    double *c = coef + ctr0*nprim + ip;
                    for (int n = 0; n < sub_nctr; ++n) {
                        cval[n*THREADS+thread_id] += s * c[n*nprim];
                    }
                }
                for (int n = 0; n < sub_nctr; ++n) {
                    cgto[n*nfi+i] = cval[n*THREADS+thread_id];
                }
            }
        }
    }
}

static __global__
void ket_cart2sorted_kernel(double *out, double *input, double *recontract_coef,
                            int *recontract_bas, int *pbas_idx_recontraction,
                            int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int row0 = item.get_group(1) * ROW_BLKSIZE;
    int c_bas_id = item.get_group(0) * TILE_X + tx;
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX*TILE_X] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX*TILE_X]>(thread_block);
    #else
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.x * ROW_BLKSIZE;
    int c_bas_id = blockIdx.y * TILE_X + tx;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX*TILE_X];
    #endif
    int thread_id = ty * TILE_X + tx;
    int row1 = min(row0 + ROW_BLKSIZE, nrow);
    int valid = c_bas_id < nbas;
    if (!valid) {
        c_bas_id = 0;
    }
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    for (int ip = ty; ip < nprim; ip += TILE_Y) {
        int p_bas_id = pbas_idx[ip];
        p_ao_offsets[ip*TILE_X+tx] = p_ao_loc[p_bas_id];
    }
    __syncthreads();
    if (!valid) {
        return;
    }

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        for (int row_id = row0+ty; row_id < row1; row_id += TILE_Y) {
            double *cgto = input + row_id*c_nao + c_ao_loc[c_bas_id] + ctr0*nfi;
            double *pgto = out   + row_id*p_nao;
            for (int i = 0; i < nfi; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = cgto[n*nfi+i];
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double *c = coef + ctr0*nprim + ip;
                    double s = cval[thread_id] * c[0];
                    for (int n = 1; n < sub_nctr; ++n) {
                        s += cval[n*THREADS+thread_id] * c[n*nprim];
                    }
                    pgto[p_ao_offsets[ip*TILE_X+tx]+i] += s;
                }
            }
        }
    }
}

static __global__
void ket_sorted2sph_kernel(double *out, double *input, double *recontract_coef,
                           int *recontract_bas, int *pbas_idx_recontraction,
                           int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int row0 = item.get_group(1) * ROW_BLKSIZE;
    int c_bas_id = item.get_group(0) * TILE_X + tx;
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX*TILE_X] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX*TILE_X]>(thread_block);
    #else
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.x * ROW_BLKSIZE;
    int c_bas_id = blockIdx.y * TILE_X + tx;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX*TILE_X];
    #endif
    int thread_id = ty * TILE_X + tx;
    int row1 = min(row0 + ROW_BLKSIZE, nrow);
    int valid = c_bas_id < nbas;
    if (!valid) {
        c_bas_id = 0;
    }
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int nfi = (li + 1) * (li + 2) / 2;
    int di = li * 2 + 1;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    for (int ip = ty; ip < nprim; ip += TILE_Y) {
        int p_bas_id = pbas_idx[ip];
        p_ao_offsets[ip*TILE_X+tx] = p_ao_loc[p_bas_id];
    }
    __syncthreads();
    if (!valid) {
        return;
    }

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        for (int row_id = row0+ty; row_id < row1; row_id += TILE_Y) {
            size_t c_off = row_id*c_nao + c_ao_loc[c_bas_id] + ctr0*di;
            double *pgto = input + row_id*p_nao;
            for (int i = 0; i < nfi; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = 0;
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double s = pgto[p_ao_offsets[ip*TILE_X+tx]+i];
                    double *c = coef + ctr0*nprim + ip;
                    for (int n = 0; n < sub_nctr; ++n) {
                        cval[n*THREADS+thread_id] += s * c[n*nprim];
                    }
                }
                for (int n = 0; n < n_ctr; ++n) {
                    double *cgto = out + c_off + n * di;
                    switch (i+nfi*li/3) {
                    case 0: { // l=0, i=0
                        cgto[0] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 1: { // l=1, i=0
                        cgto[0] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 2: { // l=1, i=1
                        cgto[1] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 3: { // l=1, i=2
                        cgto[2] += 1 * cval[n*THREADS+thread_id];
                    } break;
                    case 4: { // l=2, i=0
                        cgto[2] += -0.315391565252520002 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.546274215296039535 * cval[n*THREADS+thread_id];
                    } break;
                    case 5: { // l=2, i=1
                        cgto[0] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 6: { // l=2, i=2
                        cgto[3] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 7: { // l=2, i=3
                        cgto[2] += -0.315391565252520002 * cval[n*THREADS+thread_id];
                        cgto[4] += -0.546274215296039535 * cval[n*THREADS+thread_id];
                    } break;
                    case 8: { // l=2, i=4
                        cgto[1] += 1.092548430592079070 * cval[n*THREADS+thread_id];
                    } break;
                    case 9: { // l=2, i=5
                        cgto[2] += 0.630783130505040012 * cval[n*THREADS+thread_id];
                    } break;
                    case 10: { // l=3, i=0
                        cgto[4] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                        cgto[6] += 0.590043589926643510 * cval[n*THREADS+thread_id];
                    } break;
                    case 11: { // l=3, i=1
                        cgto[0] += 1.770130769779930531 * cval[n*THREADS+thread_id];
                        cgto[2] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                    } break;
                    case 12: { // l=3, i=2
                        cgto[3] += -1.119528997770346170 * cval[n*THREADS+thread_id];
                        cgto[5] += 1.445305721320277020 * cval[n*THREADS+thread_id];
                    } break;
                    case 13: { // l=3, i=3
                        cgto[4] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                        cgto[6] += -1.770130769779930530 * cval[n*THREADS+thread_id];
                    } break;
                    case 14: { // l=3, i=4
                        cgto[1] += 2.890611442640554055 * cval[n*THREADS+thread_id];
                    } break;
                    case 15: { // l=3, i=5
                        cgto[4] += 1.828183197857862944 * cval[n*THREADS+thread_id];
                    } break;
                    case 16: { // l=3, i=6
                        cgto[0] += -0.590043589926643510 * cval[n*THREADS+thread_id];
                        cgto[2] += -0.457045799464465739 * cval[n*THREADS+thread_id];
                    } break;
                    case 17: { // l=3, i=7
                        cgto[3] += -1.119528997770346170 * cval[n*THREADS+thread_id];
                        cgto[5] += -1.445305721320277020 * cval[n*THREADS+thread_id];
                    } break;
                    case 18: { // l=3, i=8
                        cgto[2] += 1.828183197857862944 * cval[n*THREADS+thread_id];
                    } break;
                    case 19: { // l=3, i=9
                        cgto[3] += 0.746352665180230782 * cval[n*THREADS+thread_id];
                    } break;
                    case 20: { // l=4, i=0
                        cgto[4] += 0.317356640745612911 * cval[n*THREADS+thread_id];
                        cgto[6] += -0.473087347878780002 * cval[n*THREADS+thread_id];
                        cgto[8] += 0.625835735449176134 * cval[n*THREADS+thread_id];
                    } break;
                    case 21: { // l=4, i=1
                        cgto[0] += 2.503342941796704538 * cval[n*THREADS+thread_id];
                        cgto[2] += -0.946174695757560014 * cval[n*THREADS+thread_id];
                    } break;
                    case 22: { // l=4, i=2
                        cgto[5] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                        cgto[7] += 1.770130769779930531 * cval[n*THREADS+thread_id];
                    } break;
                    case 23: { // l=4, i=3
                        cgto[4] += 0.634713281491225822 * cval[n*THREADS+thread_id];
                        cgto[8] += -3.755014412695056800 * cval[n*THREADS+thread_id];
                    } break;
                    case 24: { // l=4, i=4
                        cgto[1] += 5.310392309339791593 * cval[n*THREADS+thread_id];
                        cgto[3] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                    } break;
                    case 25: { // l=4, i=5
                        cgto[4] += -2.538853125964903290 * cval[n*THREADS+thread_id];
                        cgto[6] += 2.838524087272680054 * cval[n*THREADS+thread_id];
                    } break;
                    case 26: { // l=4, i=6
                        cgto[0] += -2.503342941796704530 * cval[n*THREADS+thread_id];
                        cgto[2] += -0.946174695757560014 * cval[n*THREADS+thread_id];
                    } break;
                    case 27: { // l=4, i=7
                        cgto[5] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                        cgto[7] += -5.310392309339791590 * cval[n*THREADS+thread_id];
                    } break;
                    case 28: { // l=4, i=8
                        cgto[2] += 5.677048174545360108 * cval[n*THREADS+thread_id];
                    } break;
                    case 29: { // l=4, i=9
                        cgto[5] += 2.676186174229156671 * cval[n*THREADS+thread_id];
                    } break;
                    case 30: { // l=4, i=10
                        cgto[4] += 0.317356640745612911 * cval[n*THREADS+thread_id];
                        cgto[6] += 0.473087347878780009 * cval[n*THREADS+thread_id];
                        cgto[8] += 0.625835735449176134 * cval[n*THREADS+thread_id];
                    } break;
                    case 31: { // l=4, i=11
                        cgto[1] += -1.770130769779930530 * cval[n*THREADS+thread_id];
                        cgto[3] += -2.007139630671867500 * cval[n*THREADS+thread_id];
                    } break;
                    case 32: { // l=4, i=12
                        cgto[4] += -2.538853125964903290 * cval[n*THREADS+thread_id];
                        cgto[6] += -2.838524087272680050 * cval[n*THREADS+thread_id];
                    } break;
                    case 33: { // l=4, i=13
                        cgto[3] += 2.676186174229156671 * cval[n*THREADS+thread_id];
                    } break;
                    case 34: { // l=4, i=14
                        cgto[4] += 0.846284375321634430 * cval[n*THREADS+thread_id];
                    } break;
                    case 35: { // l=5, i=0
                        cgto[6] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                        cgto[8] += -0.489238299435250389 * cval[n*THREADS+thread_id];
                        cgto[10] += 0.656382056840170102 * cval[n*THREADS+thread_id];
                    } break;
                    case 36: { // l=5, i=1
                        cgto[0] += 3.281910284200850514 * cval[n*THREADS+thread_id];
                        cgto[2] += -1.467714898305751160 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                    } break;
                    case 37: { // l=5, i=2
                        cgto[5] += 1.754254836801353946 * cval[n*THREADS+thread_id];
                        cgto[7] += -2.396768392486661870 * cval[n*THREADS+thread_id];
                        cgto[9] += 2.075662314881041278 * cval[n*THREADS+thread_id];
                    } break;
                    case 38: { // l=5, i=3
                        cgto[6] += 0.905893302391393842 * cval[n*THREADS+thread_id];
                        cgto[8] += 0.978476598870500775 * cval[n*THREADS+thread_id];
                        cgto[10] += -6.563820568401701020 * cval[n*THREADS+thread_id];
                    } break;
                    case 39: { // l=5, i=4
                        cgto[1] += 8.302649259524165115 * cval[n*THREADS+thread_id];
                        cgto[3] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 40: { // l=5, i=5
                        cgto[6] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                        cgto[8] += 3.913906395482003101 * cval[n*THREADS+thread_id];
                    } break;
                    case 41: { // l=5, i=6
                        cgto[0] += -6.563820568401701020 * cval[n*THREADS+thread_id];
                        cgto[2] += -0.978476598870500779 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.905893302391393842 * cval[n*THREADS+thread_id];
                    } break;
                    case 42: { // l=5, i=7
                        cgto[5] += 3.508509673602707893 * cval[n*THREADS+thread_id];
                        cgto[9] += -12.453973889286247600 * cval[n*THREADS+thread_id];
                    } break;
                    case 43: { // l=5, i=8
                        cgto[2] += 11.741719186446009300 * cval[n*THREADS+thread_id];
                        cgto[4] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                    } break;
                    case 44: { // l=5, i=9
                        cgto[5] += -4.678012898136943850 * cval[n*THREADS+thread_id];
                        cgto[7] += 4.793536784973323755 * cval[n*THREADS+thread_id];
                    } break;
                    case 45: { // l=5, i=10
                        cgto[6] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                        cgto[8] += 1.467714898305751163 * cval[n*THREADS+thread_id];
                        cgto[10] += 3.281910284200850514 * cval[n*THREADS+thread_id];
                    } break;
                    case 46: { // l=5, i=11
                        cgto[1] += -8.302649259524165110 * cval[n*THREADS+thread_id];
                        cgto[3] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 47: { // l=5, i=12
                        cgto[6] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                        cgto[8] += -11.741719186446009300 * cval[n*THREADS+thread_id];
                    } break;
                    case 48: { // l=5, i=13
                        cgto[3] += 9.587073569946647510 * cval[n*THREADS+thread_id];
                    } break;
                    case 49: { // l=5, i=14
                        cgto[6] += 3.623573209565575370 * cval[n*THREADS+thread_id];
                    } break;
                    case 50: { // l=5, i=15
                        cgto[0] += 0.656382056840170102 * cval[n*THREADS+thread_id];
                        cgto[2] += 0.489238299435250387 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.452946651195696921 * cval[n*THREADS+thread_id];
                    } break;
                    case 51: { // l=5, i=16
                        cgto[5] += 1.754254836801353946 * cval[n*THREADS+thread_id];
                        cgto[7] += 2.396768392486661877 * cval[n*THREADS+thread_id];
                        cgto[9] += 2.075662314881041278 * cval[n*THREADS+thread_id];
                    } break;
                    case 52: { // l=5, i=17
                        cgto[2] += -3.913906395482003100 * cval[n*THREADS+thread_id];
                        cgto[4] += -5.435359814348363050 * cval[n*THREADS+thread_id];
                    } break;
                    case 53: { // l=5, i=18
                        cgto[5] += -4.678012898136943850 * cval[n*THREADS+thread_id];
                        cgto[7] += -4.793536784973323750 * cval[n*THREADS+thread_id];
                    } break;
                    case 54: { // l=5, i=19
                        cgto[4] += 3.623573209565575370 * cval[n*THREADS+thread_id];
                    } break;
                    case 55: { // l=5, i=20
                        cgto[5] += 0.935602579627388771 * cval[n*THREADS+thread_id];
                    } break;
                    case 56: { // l=6, i=0
                        cgto[6] += -0.3178460113381421 * cval[n*THREADS+thread_id];
                        cgto[8] += 0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10] += -0.5045649007287241 * cval[n*THREADS+thread_id];
                        cgto[12] += 0.6831841051919144 * cval[n*THREADS+thread_id];
                    } break;
                    case 57: { // l=6, i=1
                        cgto[0] += 4.0991046311514863 * cval[n*THREADS+thread_id];
                        cgto[2] += -2.0182596029148963 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.9212052595149236 * cval[n*THREADS+thread_id];
                    } break;
                    case 58: { // l=6, i=2
                        cgto[7] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                        cgto[9] += -2.7636157785447706 * cval[n*THREADS+thread_id];
                        cgto[11] += 2.3666191622317525 * cval[n*THREADS+thread_id];
                    } break;
                    case 59: { // l=6, i=3
                        cgto[6] += -0.9535380340144264 * cval[n*THREADS+thread_id];
                        cgto[8] += 0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10] += 2.5228245036436201 * cval[n*THREADS+thread_id];
                        cgto[12] += -10.2477615778787161 * cval[n*THREADS+thread_id];
                    } break;
                    case 60: { // l=6, i=4
                        cgto[1] += 11.8330958111587634 * cval[n*THREADS+thread_id];
                        cgto[3] += -8.2908473356343109 * cval[n*THREADS+thread_id];
                        cgto[5] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                    } break;
                    case 61: { // l=6, i=5
                        cgto[6] += 5.7212282040865583 * cval[n*THREADS+thread_id];
                        cgto[8] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[10] += 5.0456490072872420 * cval[n*THREADS+thread_id];
                    } break;
                    case 62: { // l=6, i=6
                        cgto[0] += -13.6636821038382887 * cval[n*THREADS+thread_id];
                        cgto[4] += 1.8424105190298472 * cval[n*THREADS+thread_id];
                    } break;
                    case 63: { // l=6, i=7
                        cgto[7] += 5.8262136251873136 * cval[n*THREADS+thread_id];
                        cgto[9] += 5.5272315570895412 * cval[n*THREADS+thread_id];
                        cgto[11] += -23.6661916223175268 * cval[n*THREADS+thread_id];
                    } break;
                    case 64: { // l=6, i=8
                        cgto[2] += 20.1825960291489679 * cval[n*THREADS+thread_id];
                        cgto[4] += -14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 65: { // l=6, i=9
                        cgto[7] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                        cgto[9] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 66: { // l=6, i=10
                        cgto[6] += -0.9535380340144264 * cval[n*THREADS+thread_id];
                        cgto[8] += -0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10] += 2.5228245036436201 * cval[n*THREADS+thread_id];
                        cgto[12] += 10.2477615778787161 * cval[n*THREADS+thread_id];
                    } break;
                    case 67: { // l=6, i=11
                        cgto[1] += -23.6661916223175268 * cval[n*THREADS+thread_id];
                        cgto[3] += -5.5272315570895412 * cval[n*THREADS+thread_id];
                        cgto[5] += 5.8262136251873136 * cval[n*THREADS+thread_id];
                    } break;
                    case 68: { // l=6, i=12
                        cgto[6] += 11.4424564081731166 * cval[n*THREADS+thread_id];
                        cgto[10] += -30.2738940437234518 * cval[n*THREADS+thread_id];
                    } break;
                    case 69: { // l=6, i=13
                        cgto[3] += 22.1089262283581647 * cval[n*THREADS+thread_id];
                        cgto[5] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                    } break;
                    case 70: { // l=6, i=14
                        cgto[6] += -7.6283042721154111 * cval[n*THREADS+thread_id];
                        cgto[8] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 71: { // l=6, i=15
                        cgto[0] += 4.0991046311514863 * cval[n*THREADS+thread_id];
                        cgto[2] += 2.0182596029148963 * cval[n*THREADS+thread_id];
                        cgto[4] += 0.9212052595149236 * cval[n*THREADS+thread_id];
                    } break;
                    case 72: { // l=6, i=16
                        cgto[7] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                        cgto[9] += 8.2908473356343109 * cval[n*THREADS+thread_id];
                        cgto[11] += 11.8330958111587634 * cval[n*THREADS+thread_id];
                    } break;
                    case 73: { // l=6, i=17
                        cgto[2] += -20.1825960291489679 * cval[n*THREADS+thread_id];
                        cgto[4] += -14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 74: { // l=6, i=18
                        cgto[7] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                        cgto[9] += -22.1089262283581647 * cval[n*THREADS+thread_id];
                    } break;
                    case 75: { // l=6, i=19
                        cgto[4] += 14.7392841522387776 * cval[n*THREADS+thread_id];
                    } break;
                    case 76: { // l=6, i=20
                        cgto[7] += 4.6609709001498505 * cval[n*THREADS+thread_id];
                    } break;
                    case 77: { // l=6, i=21
                        cgto[6] += -0.3178460113381421 * cval[n*THREADS+thread_id];
                        cgto[8] += -0.4606026297574618 * cval[n*THREADS+thread_id];
                        cgto[10] += -0.5045649007287241 * cval[n*THREADS+thread_id];
                        cgto[12] += -0.6831841051919144 * cval[n*THREADS+thread_id];
                    } break;
                    case 78: { // l=6, i=22
                        cgto[1] += 2.3666191622317525 * cval[n*THREADS+thread_id];
                        cgto[3] += 2.7636157785447706 * cval[n*THREADS+thread_id];
                        cgto[5] += 2.9131068125936568 * cval[n*THREADS+thread_id];
                    } break;
                    case 79: { // l=6, i=23
                        cgto[6] += 5.7212282040865583 * cval[n*THREADS+thread_id];
                        cgto[8] += 7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[10] += 5.0456490072872420 * cval[n*THREADS+thread_id];
                    } break;
                    case 80: { // l=6, i=24
                        cgto[3] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                        cgto[5] += -11.6524272503746271 * cval[n*THREADS+thread_id];
                    } break;
                    case 81: { // l=6, i=25
                        cgto[6] += -7.6283042721154111 * cval[n*THREADS+thread_id];
                        cgto[8] += -7.3696420761193888 * cval[n*THREADS+thread_id];
                    } break;
                    case 82: { // l=6, i=26
                        cgto[5] += 4.6609709001498505 * cval[n*THREADS+thread_id];
                    } break;
                    case 83: { // l=6, i=27
                        cgto[6] += 1.0171072362820548 * cval[n*THREADS+thread_id];
                    } break;
                    }
                }
            }
        }
    }
}

static __global__
void ket_sph2sorted_kernel(double *out, double *input, double *recontract_coef,
                           int *recontract_bas, int *pbas_idx_recontraction,
                           int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    int row0 = item.get_group(1) * ROW_BLKSIZE;
    int c_bas_id = item.get_group(0) * TILE_X + tx;
    auto thread_block = item.get_group();
    double (&cval)[THREADS*SHM_BLKSIZE] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[THREADS*SHM_BLKSIZE]>(thread_block);
    int (&p_ao_offsets)[NPRIM_MAX*TILE_X] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[NPRIM_MAX*TILE_X]>(thread_block);
    #else
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.x * ROW_BLKSIZE;
    int c_bas_id = blockIdx.y * TILE_X + tx;
    __shared__ double cval[THREADS*SHM_BLKSIZE];
    __shared__ int p_ao_offsets[NPRIM_MAX*TILE_X];
    #endif
    int thread_id = ty * TILE_X + tx;
    int row1 = min(row0 + ROW_BLKSIZE, nrow);
    int valid = c_bas_id < nbas;
    if (!valid) {
        c_bas_id = 0;
    }
    int li = recontract_bas[c_bas_id*BAS_SLOTS+ANG_OF];
    int di = li * 2 + 1;
    int nprim = recontract_bas[c_bas_id*BAS_SLOTS+NPRIM_OF];
    int n_ctr = recontract_bas[c_bas_id*BAS_SLOTS+NCTR_OF ];
    int *pbas_idx = pbas_idx_recontraction + recontract_bas[c_bas_id*BAS_SLOTS+PTR_PBAS_IDX];
    double *coef = recontract_coef + recontract_bas[c_bas_id*BAS_SLOTS+PTR_COEFF];
    size_t c_nao = c_ao_loc[nbas];
    size_t p_nao = p_ao_loc[npbas];
    for (int ip = ty; ip < nprim; ip += TILE_Y) {
        int p_bas_id = pbas_idx[ip];
        p_ao_offsets[ip*TILE_X+tx] = p_ao_loc[p_bas_id];
    }
    __syncthreads();
    if (!valid) {
        return;
    }

    for (int ctr0 = 0; ctr0 < n_ctr; ctr0 += SHM_BLKSIZE) {
        int sub_nctr = min(n_ctr - ctr0, SHM_BLKSIZE);
        for (int row_id = row0+ty; row_id < row1; row_id += TILE_Y) {
            double *cgto = input + row_id*c_nao + c_ao_loc[c_bas_id] + ctr0*di;
            size_t p_off = row_id*p_nao;
            for (int i = 0; i < di; ++i) {
                for (int n = 0; n < sub_nctr; ++n) {
                    cval[n*THREADS+thread_id] = cgto[n*di+i];
                }
                for (int ip = 0; ip < nprim; ++ip) {
                    double *c = coef + ctr0*nprim + ip;
                    double s = cval[thread_id] * c[0];
                    for (int n = 1; n < sub_nctr; ++n) {
                        s += cval[n*THREADS+thread_id] * c[n*nprim];
                    }
                    double *pgto = out + p_off + p_ao_offsets[ip*TILE_X+tx];
                    switch (li*li+i) {
                    case 0: { // l=0, m=0
                        pgto[0] += 1 * s;
                    } break;
                    case 1: { // l=1, m=0
                        pgto[0] += 1 * s;
                    } break;
                    case 2: { // l=1, m=1
                        pgto[1] += 1 * s;
                    } break;
                    case 3: { // l=1, m=2
                        pgto[2] += 1 * s;
                    } break;
                    case 4: { // l=2, m=0
                        pgto[1] += 1.092548430592079070 * s;
                    } break;
                    case 5: { // l=2, m=1
                        pgto[4] += 1.092548430592079070 * s;
                    } break;
                    case 6: { // l=2, m=2
                        pgto[0] += -0.315391565252520002 * s;
                        pgto[3] += -0.315391565252520002 * s;
                        pgto[5] += 0.630783130505040012 * s;
                    } break;
                    case 7: { // l=2, m=3
                        pgto[2] += 1.092548430592079070 * s;
                    } break;
                    case 8: { // l=2, m=4
                        pgto[0] += 0.546274215296039535 * s;
                        pgto[3] += -0.546274215296039535 * s;
                    } break;
                    case 9: { // l=3, m=0
                        pgto[1] += 1.770130769779930531 * s;
                        pgto[6] += -0.590043589926643510 * s;
                    } break;
                    case 10: { // l=3, m=1
                        pgto[4] += 2.890611442640554055 * s;
                    } break;
                    case 11: { // l=3, m=2
                        pgto[1] += -0.457045799464465739 * s;
                        pgto[6] += -0.457045799464465739 * s;
                        pgto[8] += 1.828183197857862944 * s;
                    } break;
                    case 12: { // l=3, m=3
                        pgto[2] += -1.119528997770346170 * s;
                        pgto[7] += -1.119528997770346170 * s;
                        pgto[9] += 0.746352665180230782 * s;
                    } break;
                    case 13: { // l=3, m=4
                        pgto[0] += -0.457045799464465739 * s;
                        pgto[3] += -0.457045799464465739 * s;
                        pgto[5] += 1.828183197857862944 * s;
                    } break;
                    case 14: { // l=3, m=5
                        pgto[2] += 1.445305721320277020 * s;
                        pgto[7] += -1.445305721320277020 * s;
                    } break;
                    case 15: { // l=3, m=6
                        pgto[0] += 0.590043589926643510 * s;
                        pgto[3] += -1.770130769779930530 * s;
                    } break;
                    case 16: { // l=4, m=0
                        pgto[1] += 2.503342941796704538 * s;
                        pgto[6] += -2.503342941796704530 * s;
                    } break;
                    case 17: { // l=4, m=1
                        pgto[4] += 5.310392309339791593 * s;
                        pgto[11] += -1.770130769779930530 * s;
                    } break;
                    case 18: { // l=4, m=2
                        pgto[1] += -0.946174695757560014 * s;
                        pgto[6] += -0.946174695757560014 * s;
                        pgto[8] += 5.677048174545360108 * s;
                    } break;
                    case 19: { // l=4, m=3
                        pgto[4] += -2.007139630671867500 * s;
                        pgto[11] += -2.007139630671867500 * s;
                        pgto[13] += 2.676186174229156671 * s;
                    } break;
                    case 20: { // l=4, m=4
                        pgto[0] += 0.317356640745612911 * s;
                        pgto[3] += 0.634713281491225822 * s;
                        pgto[5] += -2.538853125964903290 * s;
                        pgto[10] += 0.317356640745612911 * s;
                        pgto[12] += -2.538853125964903290 * s;
                        pgto[14] += 0.846284375321634430 * s;
                    } break;
                    case 21: { // l=4, m=5
                        pgto[2] += -2.007139630671867500 * s;
                        pgto[7] += -2.007139630671867500 * s;
                        pgto[9] += 2.676186174229156671 * s;
                    } break;
                    case 22: { // l=4, m=6
                        pgto[0] += -0.473087347878780002 * s;
                        pgto[5] += 2.838524087272680054 * s;
                        pgto[10] += 0.473087347878780009 * s;
                        pgto[12] += -2.838524087272680050 * s;
                    } break;
                    case 23: { // l=4, m=7
                        pgto[2] += 1.770130769779930531 * s;
                        pgto[7] += -5.310392309339791590 * s;
                    } break;
                    case 24: { // l=4, m=8
                        pgto[0] += 0.625835735449176134 * s;
                        pgto[3] += -3.755014412695056800 * s;
                        pgto[10] += 0.625835735449176134 * s;
                    } break;
                    case 25: { // l=5, m=0
                        pgto[1] += 3.281910284200850514 * s;
                        pgto[6] += -6.563820568401701020 * s;
                        pgto[15] += 0.656382056840170102 * s;
                    } break;
                    case 26: { // l=5, m=1
                        pgto[4] += 8.302649259524165115 * s;
                        pgto[11] += -8.302649259524165110 * s;
                    } break;
                    case 27: { // l=5, m=2
                        pgto[1] += -1.467714898305751160 * s;
                        pgto[6] += -0.978476598870500779 * s;
                        pgto[8] += 11.741719186446009300 * s;
                        pgto[15] += 0.489238299435250387 * s;
                        pgto[17] += -3.913906395482003100 * s;
                    } break;
                    case 28: { // l=5, m=3
                        pgto[4] += -4.793536784973323750 * s;
                        pgto[11] += -4.793536784973323750 * s;
                        pgto[13] += 9.587073569946647510 * s;
                    } break;
                    case 29: { // l=5, m=4
                        pgto[1] += 0.452946651195696921 * s;
                        pgto[6] += 0.905893302391393842 * s;
                        pgto[8] += -5.435359814348363050 * s;
                        pgto[15] += 0.452946651195696921 * s;
                        pgto[17] += -5.435359814348363050 * s;
                        pgto[19] += 3.623573209565575370 * s;
                    } break;
                    case 30: { // l=5, m=5
                        pgto[2] += 1.754254836801353946 * s;
                        pgto[7] += 3.508509673602707893 * s;
                        pgto[9] += -4.678012898136943850 * s;
                        pgto[16] += 1.754254836801353946 * s;
                        pgto[18] += -4.678012898136943850 * s;
                        pgto[20] += 0.935602579627388771 * s;
                    } break;
                    case 31: { // l=5, m=6
                        pgto[0] += 0.452946651195696921 * s;
                        pgto[3] += 0.905893302391393842 * s;
                        pgto[5] += -5.435359814348363050 * s;
                        pgto[10] += 0.452946651195696921 * s;
                        pgto[12] += -5.435359814348363050 * s;
                        pgto[14] += 3.623573209565575370 * s;
                    } break;
                    case 32: { // l=5, m=7
                        pgto[2] += -2.396768392486661870 * s;
                        pgto[9] += 4.793536784973323755 * s;
                        pgto[16] += 2.396768392486661877 * s;
                        pgto[18] += -4.793536784973323750 * s;
                    } break;
                    case 33: { // l=5, m=8
                        pgto[0] += -0.489238299435250389 * s;
                        pgto[3] += 0.978476598870500775 * s;
                        pgto[5] += 3.913906395482003101 * s;
                        pgto[10] += 1.467714898305751163 * s;
                        pgto[12] += -11.741719186446009300 * s;
                    } break;
                    case 34: { // l=5, m=9
                        pgto[2] += 2.075662314881041278 * s;
                        pgto[7] += -12.453973889286247600 * s;
                        pgto[16] += 2.075662314881041278 * s;
                    } break;
                    case 35: { // l=5, m=10
                        pgto[0] += 0.656382056840170102 * s;
                        pgto[3] += -6.563820568401701020 * s;
                        pgto[10] += 3.281910284200850514 * s;
                    } break;
                    case 36: { // l=6, m=0
                        pgto[1] += 4.0991046311514863 * s;
                        pgto[6] += -13.6636821038382887 * s;
                        pgto[15] += 4.0991046311514863 * s;
                    } break;
                    case 37: { // l=6, m=1
                        pgto[4] += 11.8330958111587634 * s;
                        pgto[11] += -23.6661916223175268 * s;
                        pgto[22] += 2.3666191622317525 * s;
                    } break;
                    case 38: { // l=6, m=2
                        pgto[1] += -2.0182596029148963 * s;
                        pgto[8] += 20.1825960291489679 * s;
                        pgto[15] += 2.0182596029148963 * s;
                        pgto[17] += -20.1825960291489679 * s;
                    } break;
                    case 39: { // l=6, m=3
                        pgto[4] += -8.2908473356343109 * s;
                        pgto[11] += -5.5272315570895412 * s;
                        pgto[13] += 22.1089262283581647 * s;
                        pgto[22] += 2.7636157785447706 * s;
                        pgto[24] += -7.3696420761193888 * s;
                    } break;
                    case 40: { // l=6, m=4
                        pgto[1] += 0.9212052595149236 * s;
                        pgto[6] += 1.8424105190298472 * s;
                        pgto[8] += -14.7392841522387776 * s;
                        pgto[15] += 0.9212052595149236 * s;
                        pgto[17] += -14.7392841522387776 * s;
                        pgto[19] += 14.7392841522387776 * s;
                    } break;
                    case 41: { // l=6, m=5
                        pgto[4] += 2.9131068125936568 * s;
                        pgto[11] += 5.8262136251873136 * s;
                        pgto[13] += -11.6524272503746271 * s;
                        pgto[22] += 2.9131068125936568 * s;
                        pgto[24] += -11.6524272503746271 * s;
                        pgto[26] += 4.6609709001498505 * s;
                    } break;
                    case 42: { // l=6, m=6
                        pgto[0] += -0.3178460113381421 * s;
                        pgto[3] += -0.9535380340144264 * s;
                        pgto[5] += 5.7212282040865583 * s;
                        pgto[10] += -0.9535380340144264 * s;
                        pgto[12] += 11.4424564081731166 * s;
                        pgto[14] += -7.6283042721154111 * s;
                        pgto[21] += -0.3178460113381421 * s;
                        pgto[23] += 5.7212282040865583 * s;
                        pgto[25] += -7.6283042721154111 * s;
                        pgto[27] += 1.0171072362820548 * s;
                    } break;
                    case 43: { // l=6, m=7
                        pgto[2] += 2.9131068125936568 * s;
                        pgto[7] += 5.8262136251873136 * s;
                        pgto[9] += -11.6524272503746271 * s;
                        pgto[16] += 2.9131068125936568 * s;
                        pgto[18] += -11.6524272503746271 * s;
                        pgto[20] += 4.6609709001498505 * s;
                    } break;
                    case 44: { // l=6, m=8
                        pgto[0] += 0.4606026297574618 * s;
                        pgto[3] += 0.4606026297574618 * s;
                        pgto[5] += -7.3696420761193888 * s;
                        pgto[10] += -0.4606026297574618 * s;
                        pgto[14] += 7.3696420761193888 * s;
                        pgto[21] += -0.4606026297574618 * s;
                        pgto[23] += 7.3696420761193888 * s;
                        pgto[25] += -7.3696420761193888 * s;
                    } break;
                    case 45: { // l=6, m=9
                        pgto[2] += -2.7636157785447706 * s;
                        pgto[7] += 5.5272315570895412 * s;
                        pgto[9] += 7.3696420761193888 * s;
                        pgto[16] += 8.2908473356343109 * s;
                        pgto[18] += -22.1089262283581647 * s;
                    } break;
                    case 46: { // l=6, m=10
                        pgto[0] += -0.5045649007287241 * s;
                        pgto[3] += 2.5228245036436201 * s;
                        pgto[5] += 5.0456490072872420 * s;
                        pgto[10] += 2.5228245036436201 * s;
                        pgto[12] += -30.2738940437234518 * s;
                        pgto[21] += -0.5045649007287241 * s;
                        pgto[23] += 5.0456490072872420 * s;
                    } break;
                    case 47: { // l=6, m=11
                        pgto[2] += 2.3666191622317525 * s;
                        pgto[7] += -23.6661916223175268 * s;
                        pgto[16] += 11.8330958111587634 * s;
                    } break;
                    case 48: { // l=6, m=12
                        pgto[0] += 0.6831841051919144 * s;
                        pgto[3] += -10.2477615778787161 * s;
                        pgto[10] += 10.2477615778787161 * s;
                        pgto[21] += -0.6831841051919144 * s;
                    } break;
                    }
                }
            }
        }
    }
}

extern "C" {
int bra_sorted2cart(double *out, double *input, double *recontract_coef,
                    int *recontract_bas, int *pbas_idx_recontraction,
                    int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol, int counts)
{
    int nbatch_col = (ncol + COL_BLKSIZE-1) / COL_BLKSIZE;
    #ifdef USE_SYCL
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> blocks(counts, nbas, nbatch_col);
    sycl_get_queue()->parallel_for<class bra_sorted2cart_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      bra_sorted2cart_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    });
    #else
    dim3 blocks(nbatch_col, nbas, counts);
    bra_sorted2cart_kernel<<<blocks, THREADS>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bra_sorted2cart kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bra_cart2sorted(double *out, double *input, double *recontract_coef,
                    int *recontract_bas, int *pbas_idx_recontraction,
                    int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol, int counts)
{
    int nbatch_col = (ncol + COL_BLKSIZE-1) / COL_BLKSIZE;
    #ifdef USE_SYCL
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> blocks(counts, nbas, nbatch_col);
    sycl_get_queue()->parallel_for<class bra_cart2sorted_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      bra_cart2sorted_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    });
    #else
    dim3 blocks(nbatch_col, nbas, counts);
    bra_cart2sorted_kernel<<<blocks, THREADS>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bra_cart2sorted kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bra_sorted2sph(double *out, double *input, double *recontract_coef,
                   int *recontract_bas, int *pbas_idx_recontraction,
                   int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol, int counts)
{
    int nbatch_col = (ncol + COL_BLKSIZE-1) / COL_BLKSIZE;
    #ifdef USE_SYCL
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> blocks(counts, nbas, nbatch_col);
    sycl_get_queue()->parallel_for<class bra_sorted2sph_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      bra_sorted2sph_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    });
    #else
    dim3 blocks(nbatch_col, nbas, counts);
    bra_sorted2sph_kernel<<<blocks, THREADS>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bra_sorted2sph kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bra_sph2sorted(double *out, double *input, double *recontract_coef,
                   int *recontract_bas, int *pbas_idx_recontraction,
                   int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int ncol, int counts)
{
    int nbatch_col = (ncol + COL_BLKSIZE-1) / COL_BLKSIZE;
    #ifdef USE_SYCL
    sycl::range<3> threads(1, 1, THREADS);
    sycl::range<3> blocks(counts, nbas, nbatch_col);
    sycl_get_queue()->parallel_for<class bra_sph2sorted_sycl>(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) {
      bra_sph2sorted_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    });
    #else
    dim3 blocks(nbatch_col, nbas, counts);
    bra_sph2sorted_kernel<<<blocks, THREADS>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, ncol);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bra_sph2sorted kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int ket_sorted2cart(double *out, double *input, double *recontract_coef,
                    int *recontract_bas, int *pbas_idx_recontraction,
                    int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(TILE_Y, TILE_X);
    sycl::range<2> blocks((nbas+TILE_X-1)/TILE_X, (nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE);
    sycl_get_queue()->parallel_for<class ket_sorted2cart_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      ket_sorted2cart_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    });
    #else
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE, (nbas+TILE_X-1)/TILE_X);
    ket_sorted2cart_kernel<<<blocks, threads>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ket_sorted2cart kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int ket_cart2sorted(double *out, double *input, double *recontract_coef,
                    int *recontract_bas, int *pbas_idx_recontraction,
                    int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(TILE_Y, TILE_X);
    sycl::range<2> blocks((nbas+TILE_X-1)/TILE_X, (nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE);
    sycl_get_queue()->parallel_for<class ket_cart2sorted_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      ket_cart2sorted_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    });
    #else
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE, (nbas+TILE_X-1)/TILE_X);
    ket_cart2sorted_kernel<<<blocks, threads>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ket_cart2sorted kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int ket_sorted2sph(double *out, double *input, double *recontract_coef,
                   int *recontract_bas, int *pbas_idx_recontraction,
                   int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(TILE_Y, TILE_X);
    sycl::range<2> blocks((nbas+TILE_X-1)/TILE_X, (nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE);
    sycl_get_queue()->parallel_for<class ket_sorted2sph_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      ket_sorted2sph_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    });
    #else
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE, (nbas+TILE_X-1)/TILE_X);
    ket_sorted2sph_kernel<<<blocks, threads>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ket_sorted2sph kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int ket_sph2sorted(double *out, double *input, double *recontract_coef,
                   int *recontract_bas, int *pbas_idx_recontraction,
                   int *c_ao_loc, int *p_ao_loc, int nbas, int npbas, int nrow)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(TILE_Y, TILE_X);
    sycl::range<2> blocks((nbas+TILE_X-1)/TILE_X, (nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE);
    sycl_get_queue()->parallel_for<class ket_sph2sorted_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      ket_sph2sorted_kernel(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    });
    #else
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((nrow+ROW_BLKSIZE-1)/ROW_BLKSIZE, (nbas+TILE_X-1)/TILE_X);
    ket_sph2sorted_kernel<<<blocks, threads>>>(
            out, input, recontract_coef, recontract_bas, pbas_idx_recontraction,
            c_ao_loc, p_ao_loc, nbas, npbas, nrow);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ket_sph2sorted kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
