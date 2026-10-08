/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include "gint/gint.h"

#define NOT_INITIALIZED -1

/* Contracted addresses encode (i, image, j), with i >= j. Reflecting an
 * AO pair also inverts its image; the auxiliary momentum is zero.
 */
int PBCpair_recontraction_info(int *out_idx, int *out_offsets, double *coef,
                               int *orig_ao_pair_id, int *output_lut,
                               int *cderi_size, uint32_t *bas_ij_idx, int npairs,
                               int *mapping_orig_shell, int *prim_id_within_shell,
                               int *recontract_bas, double *recontract_coef,
                               int *ao_loc, int nbas, int ncells, int nao, int cart)
{
    int cderi_npairs = *cderi_size;
    int bvk_nbas = nbas * ncells;
    int count = 0;
    int inp_offset = 0;
    for (int pair_id = 0; pair_id < npairs; pair_id++) {
        uint32_t bas_ij = bas_ij_idx[pair_id];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int jL = jsh / nbas;
        jsh = jsh - jL * nbas;
        int orig_ish = mapping_orig_shell[ish];
        int orig_jsh = mapping_orig_shell[jsh];
        int ip = prim_id_within_shell[ish];
        int jp = prim_id_within_shell[jsh];
        int li = recontract_bas[orig_ish*BAS_SLOTS+ANG_OF];
        int lj = recontract_bas[orig_jsh*BAS_SLOTS+ANG_OF];
        int di = li * 2 + 1;
        int dj = lj * 2 + 1;
        if (cart) {
            di = (li + 1) * (li + 2) / 2;
            dj = (lj + 1) * (lj + 2) / 2;
        }
        int iprim = recontract_bas[orig_ish*BAS_SLOTS+NPRIM_OF];
        int jprim = recontract_bas[orig_jsh*BAS_SLOTS+NPRIM_OF];
        int ictr = recontract_bas[orig_ish*BAS_SLOTS+NCTR_OF];
        int jctr = recontract_bas[orig_jsh*BAS_SLOTS+NCTR_OF];
        double *ci = recontract_coef + recontract_bas[orig_ish*BAS_SLOTS+PTR_COEFF];
        double *cj = recontract_coef + recontract_bas[orig_jsh*BAS_SLOTS+PTR_COEFF];
        int i0 = ao_loc[orig_ish];
        int j0 = ao_loc[orig_jsh];
        // The evaluator keeps ish >= jsh in the primitive unit-cell basis.
        // Diagonal primitive blocks contain all AO components and image
        // partners, so half-weight them before additive symmetry completion.
        double weight = ish == jsh ? .5 : 1.;
        for (int j_in_shell = 0; j_in_shell < dj; ++j_in_shell) {
        for (int i_in_shell = 0; i_in_shell < di; ++i_in_shell, ++inp_offset) {
            for (int ic = 0; ic < ictr; ++ic) {
                int i = i0 + ic * di + i_in_shell;
                for (int jc = 0; jc < jctr; ++jc) {
                    double cc = weight * ci[ic*iprim+ip] * cj[jc*jprim+jp];
                    if (cc == 0) continue;
                    int j = j0 + jc * dj + j_in_shell;
                    int ij = (i * ncells + jL) * nao + j;
                    if (output_lut[ij] == NOT_INITIALIZED) {
                        orig_ao_pair_id[cderi_npairs] = ij;
                        output_lut[ij] = cderi_npairs++;
                    }
                    out_idx[count] = output_lut[ij];
                    coef[count++] = cc;
                }
            }
            out_offsets[inp_offset+1] = count;
        } }
    }
    *cderi_size = cderi_npairs;
    return count;
}

/* Recontract a host [primitive pair, aux] buffer directly into CDERI.
 */
void PBCrecontract_cderi(double* __restrict__ out,
                         double* __restrict__ prim_cderi,
                         int *out_idx, int *out_offsets, double *coef,
                         int naux, int out_ncol, int inp_ncol)
{
    const int row_block = 8;
#pragma omp parallel for schedule(static)
    for (int row0 = 0; row0 < naux; row0 += row_block) {
        int nrows = naux - row0;
        if (nrows > row_block) nrows = row_block;
        double* __restrict__ dst = out + (size_t)row0 * out_ncol;
        double* __restrict__ src = prim_cderi + row0;
        for (int i = 0; i < inp_ncol; i++) {
            int k0 = out_offsets[i];
            int k1 = out_offsets[i+1];
            for (int k = k0; k < k1; k++) {
                int j = out_idx[k];
                double c = coef[k];
                for (int row = 0; row < nrows; ++row) {
                    dst[row*(size_t)out_ncol+j] += c * src[i*(size_t)naux+row];
                }
            }
        }
    }
}

void PBCzrecontract_cderi(double _Complex* __restrict__ out,
                          double _Complex* __restrict__ prim_cderi,
                          int *out_idx, int *out_offsets, double *coef,
                          int naux, int out_ncol, int inp_ncol)
{
    const int row_block = 8;
#pragma omp parallel for schedule(static)
    for (int row0 = 0; row0 < naux; row0 += row_block) {
        int nrows = naux - row0;
        if (nrows > row_block) nrows = row_block;
        double _Complex* __restrict__ dst = out + (size_t)row0 * out_ncol;
        double _Complex* __restrict__ src = prim_cderi + row0;
        for (int i = 0; i < inp_ncol; i++) {
            int k0 = out_offsets[i];
            int k1 = out_offsets[i+1];
            for (int k = k0; k < k1; k++) {
                int j = out_idx[k];
                double c = coef[k];
                for (int row = 0; row < nrows; ++row) {
                    dst[row*(size_t)out_ncol+j] += c * src[i*(size_t)naux+row];
                }
            }
        }
    }
}
