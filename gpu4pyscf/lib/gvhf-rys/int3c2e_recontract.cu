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
#define WARP_SIZE       32

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

// Gather a destination's primitive contributions in [pair, aux] layout.
// Each block owns an output pair; its warps loop over auxiliary components.
static __global__
void recontract_gather_kernel(double *out, double *input, int *inp_idx,
                              double *coef, int *offsets, size_t naux)
{
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    int warps = blockDim.x / warpSize;
    int row_id = blockIdx.x;
    int start = offsets[row_id] + lane;
    int stop = offsets[row_id+1];
    unsigned mask = __activemask();
    for (size_t col = warp; col < naux; col += warps) {
        double real = 0.;
        double imag = 0.;
        for (int k = start; k < stop; k += warpSize) {
            size_t address = inp_idx[k] * naux + col;
            double c = coef[k];
            real += c * input[address];
        }
        for (int d = warpSize / 2; d > 0; d /= 2) {
            real += __shfl_down_sync(mask, real, d, warpSize);
        }
        if (lane == 0) {
            size_t address = row_id * naux + col;
            out[address] = real;
        }
    }
}

extern "C" {
int recontract_ao_pair_gather(cudaStream_t stream, double *out, double *input,
                             int *inp_idx, double *coef, int *offsets,
                             int naux, int npair)
{
    if (naux == 0 || npair == 0) {
        return 0;
    }
    int threads = 16 * WARP_SIZE;
    // One block per output pair; the kernel loops over all auxiliary columns.
    recontract_gather_kernel<<<npair, threads, 0, stream>>>(
        out, input, inp_idx, coef, offsets, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "recontract_ao_pair_gather error %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

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
                            int *orig_ao_pair_id, uint32_t *bas_ij_idx, int npairs,
                            int *mapping_orig_shell, int *prim_id_within_shell,
                            int *recontract_bas, double *recontract_coef,
                            int *ao_loc, int nbas_sorted, int nao, int cart)
{
    int *output_lut = (int *)malloc(sizeof(int) * nao * nao);
    for (int n = 0; n < nao * nao; ++n) output_lut[n] = NOT_INITIALIZED;

    int count = 0;
    int cderi_npairs = 0;
    int inp_offset = 0;
    for (int pair_id = 0; pair_id < npairs; pair_id++) {
        uint32_t bas_ij = bas_ij_idx[pair_id];
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
        if (orig_ish == orig_jsh) {
            for (int j_in_shell = 0; j_in_shell < dj; ++j_in_shell) {
            for (int i_in_shell = 0; i_in_shell < di; ++i_in_shell, ++inp_offset) {
                for (int ic = 0; ic < ictr; ++ic) {
                    int i = i0 + ic * di + i_in_shell;
                    for (int jc = 0; jc < ictr; ++jc) {
                        double cc = ci[ic*iprim+ip] * ci[jc*iprim+jp];
                        if (cc == 0) {
                            continue;
                        }
                        int j = i0 + jc * di + j_in_shell;
                        int ij = i * nao + j;
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
                        double cc = ci[ic*iprim+ip] * cj[jc*jprim+jp];
                        if (cc == 0) {
                            continue;
                        }
                        int j = j0 + jc * dj + j_in_shell;
                        int ij = i * nao + j;
                        // Ensure writing to the tril part of the output matrix.
                        if (i < j) {
                            // The off-block of the original mol can be both triu
                            // and tril blocks. When one general contraction shell
                            // of mol is decontracted to two types of primitive
                            // shells, such as
                            // C  S                   C  S
                            // 9.0  0.7  0.0          9.0  0.7
                            // 2.5  0.5  0.0    =>    2.5  0.5
                            // 0.5  0.4  1.0          C  S
                            //                        0.5  1.0
                            // the two primitives of different atoms are collected
                            // into two groups. In the bas_ij_idx of sorted_mol,
                            // the cross-group block in the tril part can
                            // contribute to the triu block of the original mol.
                            ij = j * nao + i;
                        }
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
            } }
        }
    }
    *idx_size = count;
    free(output_lut);
    return cderi_npairs;
}

// Generate the same mapping as pair_recontraction_info, grouped by output pair
// for recontract_gather_kernel.
int pair_recontraction_gather_info(int *inp_idx, int *offsets, double *coef,
                                   int *idx_size, int *orig_ao_pair_id,
                                   uint32_t *bas_ij_idx, int npairs,
                                   int *mapping_orig_shell, int *prim_id_within_shell,
                                   int *recontract_bas, double *recontract_coef,
                                   int *ao_loc, int nbas_sorted, int nao, int cart)
{
    *idx_size = 0;
    offsets[0] = 0;
    size_t capacity = 0;
    for (int pair_id = 0; pair_id < npairs; ++pair_id) {
        uint32_t bas_ij = bas_ij_idx[pair_id];
        int ish = bas_ij / nbas_sorted;
        int jsh = bas_ij - nbas_sorted * ish;
        int orig_ish = mapping_orig_shell[ish];
        int orig_jsh = mapping_orig_shell[jsh];
        int li = recontract_bas[orig_ish*BAS_SLOTS+ANG_OF];
        int lj = recontract_bas[orig_jsh*BAS_SLOTS+ANG_OF];
        int di = li * 2 + 1;
        int dj = lj * 2 + 1;
        if (cart) {
            di = (li + 1) * (li + 2) / 2;
            dj = (lj + 1) * (lj + 2) / 2;
        }
        int ictr = recontract_bas[orig_ish*BAS_SLOTS+NCTR_OF];
        int jctr = recontract_bas[orig_jsh*BAS_SLOTS+NCTR_OF];
        int ij_count = di * dj * ictr * jctr;
        if (orig_ish == orig_jsh) ij_count *= 2;
        // Both primitive orientations can contribute to a diagonal element.
        capacity += ij_count;
    }
    if (capacity == 0) {
        return 0;
    }
    int *scatter_inp = (int *)malloc(capacity * sizeof(int));
    int *scatter_out = (int *)malloc(capacity * sizeof(int));
    double *scatter_coef = (double *)malloc(capacity * sizeof(double));

    int inp_count = 0;
    int cderi_npairs = pair_recontraction_info(
            scatter_inp, scatter_out, scatter_coef, &inp_count,
            orig_ao_pair_id, bas_ij_idx, npairs,
            mapping_orig_shell, prim_id_within_shell,
            recontract_bas, recontract_coef, ao_loc, nbas_sorted, nao, cart);

    int *counts = (int *)malloc(cderi_npairs * sizeof(int));
    for (int row = 0; row < cderi_npairs; ++row) {
        counts[row] = 0;
    }
    for (int k = 0; k < inp_count; ++k) {
        ++counts[scatter_out[k]];
    }
    int off = 0;
    for (int row = 0; row < cderi_npairs; ++row) {
        off += counts[row];
        offsets[row+1] = off;
    }
    int *cursor = counts;
    for (int row = 0; row < cderi_npairs; ++row) {
        cursor[row] = offsets[row];
    }
    // Use the row starts as insertion cursors, preserving contribution order
    // within each row.
    for (int k = 0; k < inp_count; ++k) {
        int dst = cursor[scatter_out[k]]++;
        inp_idx[dst] = scatter_inp[k];
        coef[dst] = scatter_coef[k];
    }
    *idx_size = inp_count;
    free(counts);
    free(scatter_inp);
    free(scatter_out);
    free(scatter_coef);
    return cderi_npairs;
}
}
