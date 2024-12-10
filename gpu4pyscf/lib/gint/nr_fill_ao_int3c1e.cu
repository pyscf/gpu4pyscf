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
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g1e.cu"
#include "g1e_root_123.cu"
#include "g3c1e.cu"

// 1 roots upto (p|s)  6    = 3*1*(2*1)
// 2 roots upto (d|p)  36   = 3*2*(3*2)
// 3 roots upto (f|d)  108  = 3*3*(4*3)
// 4 roots upto (g|f)  240  = 3*4*(5*4)
// 5 roots upto (h|g)  450  = 3*5*(6*5)

#define GSIZE1_INT3C_1E 6
#define GSIZE2_INT3C_1E 36
#define GSIZE3_INT3C_1E 108
#define GSIZE4_INT3C_1E 240
#define GSIZE5_INT3C_1E 450

static int GINTfill_int3c1e_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                  const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                  const double omega, const double* grid_points, const double* charge_exponents, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l << 2) | j_l;
        switch (type_ijkl) {
        case (0<<2)|0: GINTfill_int3c1e_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        case (1<<2)|0: GINTfill_int3c1e_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: GINTfill_int3c1e_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 3: GINTfill_int3c1e_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 4: GINTfill_int3c1e_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 5: GINTfill_int3c1e_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_charge_contracted_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                    const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                    const double omega, const double* grid_points, const double* charge_exponents,
                                                    const int n_charge_sum_per_thread, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = (offsets.ntasks_kl + n_charge_sum_per_thread - 1) / n_charge_sum_per_thread;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l << 2) | j_l;
        switch (type_ijkl) {
        case (0<<2)|0: GINTfill_int3c1e_charge_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        case (1<<2)|0: GINTfill_int3c1e_charge_contracted_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: GINTfill_int3c1e_charge_contracted_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 3: GINTfill_int3c1e_charge_contracted_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 4: GINTfill_int3c1e_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 5: GINTfill_int3c1e_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_density_contracted_tasks(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                     const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                     const double omega, const double* grid_points, const double* charge_exponents,
                                                     const int n_pair_sum_per_thread, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int ntasks_ij = (offsets.ntasks_ij + n_pair_sum_per_thread - 1) / n_pair_sum_per_thread;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l << 2) | j_l;
        switch (type_ijkl) {
        case (0<<2)|0: GINTfill_int3c1e_density_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
        case (1<<2)|0: GINTfill_int3c1e_density_contracted_kernel10<<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: GINTfill_int3c1e_density_contracted_kernel_general<2> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, i_l, j_l, nprim_ij, omega, grid_points, charge_exponents); break;
    case 3: GINTfill_int3c1e_density_contracted_kernel_general<3> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, i_l, j_l, nprim_ij, omega, grid_points, charge_exponents); break;
    case 4: GINTfill_int3c1e_density_contracted_kernel_general<4> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, i_l, j_l, nprim_ij, omega, grid_points, charge_exponents); break;
    case 5: GINTfill_int3c1e_density_contracted_kernel_general<5> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, i_l, j_l, nprim_ij, omega, grid_points, charge_exponents); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int3c1e(const cudaStream_t stream, const BasisProdCache* bpcache,
                     const double* grid_points, const double* charge_exponents, const int ngrids,
                     double* integrals, const int nao,
                     const int* strides, const int* ao_offsets,
                     const int* bins_locs_ij, int nbins,
                     const int cp_ij_id, const double omega)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > 5) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c1e_tasks(integrals, offsets, i_l, j_l, nprim_ij,
                                               strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                               omega, grid_points, charge_exponents, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_charge_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                       const double* grid_points, const double* charge_exponents, const int ngrids,
                                       double* integral_charge_contracted, const int nao,
                                       const int* strides, const int* ao_offsets,
                                       const int* bins_locs_ij, int nbins,
                                       const int cp_ij_id, const double omega, const int n_charge_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > 5) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c1e_charge_contracted_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                                 strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                                 omega, grid_points, charge_exponents, n_charge_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_density_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                        const double* grid_points, const double* charge_exponents, const int ngrids,
                                        const double* dm_pair_ordered, const int* density_offset,
                                        double* integral_density_contracted,
                                        const int* bins_locs_ij, int nbins,
                                        const int cp_ij_id, const double omega, const int n_pair_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > 5) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        HermiteDensityOffsets hermite_density_offsets;
        hermite_density_offsets.density_offset_of_angular_pair = density_offset[cp_ij_id];
        hermite_density_offsets.pair_offset_of_angular_pair = bas_pairs_locs[cp_ij_id];
        hermite_density_offsets.n_pair_of_angular_pair = bas_pairs_locs[cp_ij_id + 1] - bas_pairs_locs[cp_ij_id];

        const int err = GINTfill_int3c1e_density_contracted_tasks(integral_density_contracted, dm_pair_ordered, hermite_density_offsets,
                                                                  offsets, i_l, j_l, nprim_ij,
                                                                  omega, grid_points, charge_exponents, n_pair_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
