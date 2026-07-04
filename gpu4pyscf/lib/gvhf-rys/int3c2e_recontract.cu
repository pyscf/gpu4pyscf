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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

#define PTR_PBAS_IDX    4

static __global__
void recontract_kernel(double *out, double *input, int *out_idx, int *inp_idx,
                       double *coef, int naux)
{
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int row_id = blockIdx.x;
    size_t Naux = naux;
    out = out + out_idx[row_id] * Naux;
    input = input + inp_idx[row_id] * Naux;
    double c = coef[row_id];
    for (int i = thread_id; i < naux; i += threads) {
        atomicAdd(out+i, input[i] * c);
    }
}

extern "C" {
int recontract_ao_pair(double *out, double *input, int *out_idx, int *inp_idx,
                       double *coef, int naux, int count)
{
    recontract_kernel<<<count, 256>>>(out, input, out_idx, inp_idx, coef, naux);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr, "recontract_ao_pair error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

#define NOT_INITIALIZED -1

int pair_recontraction_info(int *inp_idx, int *out_idx, double *coef, int *idx_size,
                            int *orig_ao_pair_id, int *output_lut,
                            int *bas_ij_idx, int npairs,
                            int *mapping_orig_shell, int *prim_id_within_shell,
                            int *recontract_bas, double *recontract_coef,
                            int *ao_loc, int nbas_sorted, int nao)
{
    int count = 0;
    int cderi_npairs = 0;
    int inp_offset = 0;
    for (int pair_id = 0; pair_id < npairs; pair_id++) {
        int bas_ij = bas_ij_idx[pair_id];
        int ish = bas_ij / nbas_sorted;
        int jsh = bas_ij - nbas_sorted * ish;
        int orig_ish = mapping_orig_shell[ish];
        int orig_jsh = mapping_orig_shell[jsh];
        int ip = prim_id_within_shell[ish];
        int jp = prim_id_within_shell[jsh];
        int li = recontract_bas[orig_ish*BAS_SLOTS+ANG_OF];
        int lj = recontract_bas[orig_jsh*BAS_SLOTS+ANG_OF];
        int di = li * 2 + 1;
        int dj = lj * 2 + 1;
        int iprim = recontract_bas[orig_ish*BAS_SLOTS+NPRIM_OF];
        int jprim = recontract_bas[orig_jsh*BAS_SLOTS+NPRIM_OF];
        int ictr = recontract_bas[orig_ish*BAS_SLOTS+NCTR_OF];
        int jctr = recontract_bas[orig_jsh*BAS_SLOTS+NCTR_OF];
        double *ci = recontract_coef + recontract_bas[orig_ish*BAS_SLOTS+PTR_COEFF];
        double *cj = recontract_coef + recontract_bas[orig_jsh*BAS_SLOTS+PTR_COEFF];
        int i0 = ao_loc[orig_ish];
        int j0 = ao_loc[orig_jsh];
        if (orig_ish == orig_jsh) {
            for (int j_in_shell = 0; j_in_shell < dj; ++j_in_shell) {
            for (int i_in_shell = 0; i_in_shell < di; ++i_in_shell, ++inp_offset) {
                for (int ic = 0; ic < ictr; ++ic) {
                    int i = i0 + ic * di + i_in_shell;
                    for (int jc = 0; jc < ictr; ++jc) {
                        int j = i0 + jc * di + j_in_shell;
                        int ij = i * nao + j;
                        double cc = ci[ic*iprim+ip] * ci[jc*iprim+jp];
                        if (i >= j) {
                            if (output_lut[ij] == NOT_INITIALIZED) {
                                orig_ao_pair_id[cderi_npairs] = ij;
                                output_lut[ij] = cderi_npairs;
                                cderi_npairs++;
                            }
                            inp_idx[count] = inp_offset;
                            out_idx[count] = output_lut[ij];
                            coef[count] = cc;
                            count++;
                        }
                        if (ish != jsh && i <= j) {
                            // For diagonal blocks of mol, transpose the lower
                            // triangular part of corresponding sorted_mol to
                            // fill the triu part. The triu part of sorted_mol
                            // also contributes to the tril part of mol.
                            ij = j * nao + i;
                            if (output_lut[ij] == NOT_INITIALIZED) {
                                orig_ao_pair_id[cderi_npairs] = ij;
                                output_lut[ij] = cderi_npairs;
                                cderi_npairs++;
                            }
                            inp_idx[count] = inp_offset;
                            out_idx[count] = output_lut[ij];
                            coef[count] = cc;
                            count++;
                        }
                    }
                }
            } }
        } else {
            for (int j_in_shell = 0; j_in_shell < dj; ++j_in_shell) {
            for (int i_in_shell = 0; i_in_shell < di; ++i_in_shell, ++inp_offset) {
                for (int ic = 0; ic < ictr; ++ic) {
                    int i = i0 + ic * di + i_in_shell;
                    for (int jc = 0; jc < jctr; ++jc) {
                        int j = j0 + jc * dj + j_in_shell;
                        int ij = i * nao + j;
                        if (output_lut[ij] == NOT_INITIALIZED) {
                            orig_ao_pair_id[cderi_npairs] = ij;
                            output_lut[ij] = cderi_npairs;
                            cderi_npairs++;
                        }
                        inp_idx[count] = inp_offset;
                        out_idx[count] = output_lut[ij];
                        coef[count] = ci[ic*iprim+ip] * cj[jc*jprim+jp];
                        count++;
                    }
                }
            } }
        }
    }
    *idx_size = count;
    return cderi_npairs;
}
}
