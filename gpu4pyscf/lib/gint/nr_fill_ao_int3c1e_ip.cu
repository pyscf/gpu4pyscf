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
#include "gint1e.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g1e.cu"
#include "g1e_ip_root_1.cu"
#include "g3c1e_ip.cu"

static int GINTfill_int3c1e_ip_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                     const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                     const double omega, const double* grid_points, const double* charge_exponents, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l + 1) * 10 + j_l;
        switch (type_ijkl) {
        case 10: GINTfill_int3c1e_ip_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: GINTfill_int3c1e_ip_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 3: GINTfill_int3c1e_ip_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 4: GINTfill_int3c1e_ip_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 5: GINTfill_int3c1e_ip_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
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

static int GINTfill_int3c1e_ip1_charge_contracted_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                        const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                        const double omega, const double* grid_points, const double* charge_exponents,
                                                        const int n_charge_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = (offsets.ntasks_kl + n_charge_sum_per_thread - 1) / n_charge_sum_per_thread;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: GINTfill_int3c1e_ip1_charge_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 01: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 02: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 03: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 04: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 4> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 10: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 11: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 12: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 13: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 20: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 21: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 22: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 30: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<3, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 31: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<3, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 40: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<4, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 4: GINTfill_int3c1e_ip1_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        case 5: GINTfill_int3c1e_ip1_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "type_ij = %d, nrys_roots = %d out of range\n", type_ij, nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_ip1_density_contracted_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                         const double* density, const int* aoslice, const int nao,
                                                         const double omega, const double* grid_points, const double* charge_exponents,
                                                         const cudaStream_t stream)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: GINTfill_int3c1e_ip1_density_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 01: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 02: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 03: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, shell, nao, omega, grid_points, charge_exponents); break;
    // case 04: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 4> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 10: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 11: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 12: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 13: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 20: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 21: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 22: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 30: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<3, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 31: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<3, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    // case 40: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<4, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 1: GINTfill_int3c1e_ip1_density_contracted_kernel_general<1, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
        case 2: GINTfill_int3c1e_ip1_density_contracted_kernel_general<2, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
        case 3: GINTfill_int3c1e_ip1_density_contracted_kernel_general<3, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
        case 4: GINTfill_int3c1e_ip1_density_contracted_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
        case 5: GINTfill_int3c1e_ip1_density_contracted_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "type_ij = %d, nrys_roots = %d out of range\n", type_ij, nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_ip2_density_contracted_tasks(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                         const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                         const double omega, const double* grid_points, const double* charge_exponents,
                                                         const int n_pair_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = (offsets.ntasks_ij + n_pair_sum_per_thread - 1) / n_pair_sum_per_thread;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    switch (i_l + j_l) {
    case  0: GINTfill_int3c1e_ip2_density_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  1: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 1> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  2: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 2> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  3: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 3> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  4: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 4> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  5: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 5> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  6: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 6> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  7: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 7> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  8: GINTfill_int3c1e_ip2_density_contracted_kernel_general< 8> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    // Up to g + g = 8 now
    default:
        fprintf(stderr, "i_l + j_l = %d out of range\n", i_l + j_l);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_ip2_charge_contracted_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                        const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                        const int* gridslice,
                                                        const double omega, const double* grid_points, const double* charge_exponents,
                                                        const cudaStream_t stream)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: GINTfill_int3c1e_ip2_charge_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 01: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 02: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 03: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 04: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 4> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 10: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 11: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 12: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 13: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 3> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 20: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 21: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 22: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 30: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<3, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 31: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<3, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    // case 40: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<4, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 1: GINTfill_int3c1e_ip2_charge_contracted_kernel_general<1, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
        case 2: GINTfill_int3c1e_ip2_charge_contracted_kernel_general<2, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
        case 3: GINTfill_int3c1e_ip2_charge_contracted_kernel_general<3, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
        case 4: GINTfill_int3c1e_ip2_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
        case 5: GINTfill_int3c1e_ip2_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "type_ij = %d, nrys_roots = %d out of range\n", type_ij, nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int3c1e_ip(const cudaStream_t stream, const BasisProdCache* bpcache,
                        const double* grid_points, const double* charge_exponents, const int ngrids,
                        double* integrals,
                        const int* strides, const int* ao_offsets,
                        const int* bins_locs_ij, const int nbins,
                        const int cp_ij_id, const double omega)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
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

        const int err = GINTfill_int3c1e_ip_tasks(integrals, offsets, i_l, j_l, nprim_ij,
                                                  strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                  omega, grid_points, charge_exponents, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_ip1_density_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                            const double* grid_points, const double* charge_exponents, const int ngrids,
                                            double* integral_charge_contracted,
                                            const int* bins_locs_ij, const int nbins,
                                            const int cp_ij_id,
                                            const double* density, const int* aoslice, const int nao,
                                            const double omega)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
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

        const int err = GINTfill_int3c1e_ip1_density_contracted_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                                      density, aoslice, nao,
                                                                      omega, grid_points, charge_exponents, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}


int GINTfill_int3c1e_ip1_charge_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                           const double* grid_points, const double* charge_exponents, const int ngrids,
                                           double* integral_charge_contracted,
                                           const int* strides, const int* ao_offsets,
                                           const int* bins_locs_ij, const int nbins,
                                           const int cp_ij_id, const double omega, const int n_charge_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
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

        const int err = GINTfill_int3c1e_ip1_charge_contracted_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                                     strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                                     omega, grid_points, charge_exponents, n_charge_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_ip2_density_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                            const double* grid_points, const double* charge_exponents, const int ngrids,
                                            const double* dm_pair_ordered, const int* density_offset,
                                            double* integral_density_contracted,
                                            const int* bins_locs_ij, const int nbins,
                                            const int cp_ij_id, const double omega, const int n_pair_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
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

        const int err = GINTfill_int3c1e_ip2_density_contracted_tasks(integral_density_contracted, dm_pair_ordered, hermite_density_offsets,
                                                                      offsets, i_l, j_l, nprim_ij,
                                                                      omega, grid_points, charge_exponents, n_pair_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_ip2_charge_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                           const double* grid_points, const double* charge_exponents, const int ngrids,
                                           double* integral_charge_contracted,
                                           const int* strides, const int* ao_offsets,
                                           const int* bins_locs_ij, const int nbins,
                                           const int cp_ij_id,
                                           const int* gridslice,
                                           const double omega)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
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

        const int err = GINTfill_int3c1e_ip2_charge_contracted_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                                     strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                                     gridslice, omega, grid_points, charge_exponents, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
