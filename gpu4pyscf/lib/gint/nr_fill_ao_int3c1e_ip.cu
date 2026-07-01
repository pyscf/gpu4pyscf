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


#include "gint.h"
#include "gint1e.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g1e.cu"
#include "g1e_ip_root_1.cu"
#include "g3c1e_ip.cu"

// Abstracts 2D thread/block config (THREADSX/Y are swapped between SYCL and CUDA).
// Used 3x in this file.
#ifdef USE_SYCL
#define LAUNCH_CONFIG() \
    sycl::range<2> threads(THREADSY, THREADSX); \
    sycl::range<2> blocks((ngrids+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
#else
#define LAUNCH_CONFIG() \
    const dim3 threads(THREADSX, THREADSY); \
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
#endif

// Abstracts 2D kernel launch syntax. blocks/threads must be in scope.
// TAG:    unique SYCL class name (ignored on CUDA)
// KERNEL: kernel function with template args if needed
// ...:    kernel arguments
#ifdef USE_SYCL
#define LAUNCH_KERNEL(TAG, KERNEL, ...) \
    stream.parallel_for<class TAG>(     \
        sycl::nd_range<2>(blocks * threads, threads), \
        [=](auto item) [[intel::kernel_args_restrict]] { KERNEL(__VA_ARGS__); });
#else
#define LAUNCH_KERNEL(TAG, KERNEL, ...) \
    KERNEL<<<blocks, threads, 0, stream>>>(__VA_ARGS__);
#endif

static int GINTfill_int3c1e_ip_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                     const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                     const double omega, const double* grid_points, const double* charge_exponents, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    LAUNCH_CONFIG();
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l + 1) * 10 + j_l;
        switch (type_ijkl) {
        case 10: LAUNCH_KERNEL(GINTfill_int3c1e_ip_kernel00_sycl, GINTfill_int3c1e_ip_kernel00, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: LAUNCH_KERNEL(GINTfill_int3c1e_ip_kernel_general_2_sycl, GINTfill_int3c1e_ip_kernel_general<2, GSIZE2_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 3: LAUNCH_KERNEL(GINTfill_int3c1e_ip_kernel_general_3_sycl, GINTfill_int3c1e_ip_kernel_general<3, GSIZE3_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 4: LAUNCH_KERNEL(GINTfill_int3c1e_ip_kernel_general_4_sycl, GINTfill_int3c1e_ip_kernel_general<4, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 5: LAUNCH_KERNEL(GINTfill_int3c1e_ip_kernel_general_5_sycl, GINTfill_int3c1e_ip_kernel_general<5, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
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

    LAUNCH_CONFIG();
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel00_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel00, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 01: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_01_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 1>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 02: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_02_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 2>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 03: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_03_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 3>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 04: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_04_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<0, 4>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 10: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_10_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 0>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 11: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_11_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 1>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 12: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_12_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 2>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 13: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_13_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<1, 3>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 20: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_20_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 0>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 21: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_21_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 1>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 22: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_22_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<2, 2>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 30: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_30_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<3, 0>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 31: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_31_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<3, 1>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    case 40: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded_40_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded<4, 0>, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 4: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_general_4_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
        case 5: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_charge_contracted_kernel_general_5_sycl, GINTfill_int3c1e_ip1_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents) break;
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

    LAUNCH_CONFIG();
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel00_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel00, output, offsets, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
    // case 01: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 1>
    // case 02: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 2>
    // case 03: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 3>
    // case 04: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<0, 4>
    // case 10: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 0>
    // case 11: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 1>
    // case 12: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 2>
    // case 13: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<1, 3>
    // case 20: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 0>
    // case 21: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 1>
    // case 22: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<2, 2>
    // case 30: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<3, 0>
    // case 31: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<3, 1>
    // case 40: GINTfill_int3c1e_ip1_density_contracted_kernel_expanded<4, 0>
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 1: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel_general_1_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel_general<1, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
        case 2: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel_general_2_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel_general<2, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
        case 3: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel_general_3_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel_general<3, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
        case 4: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel_general_4_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel_general<4, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
        case 5: LAUNCH_KERNEL(GINTfill_int3c1e_ip1_density_contracted_kernel_general_5_sycl, GINTfill_int3c1e_ip1_density_contracted_kernel_general<5, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, density, aoslice, nao, omega, grid_points, charge_exponents) break;
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

    LAUNCH_CONFIG();
    switch (i_l + j_l) {
    case  0: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel00_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel00, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  1: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_1_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 1>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  2: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_2_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 2>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  3: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_3_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 3>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  4: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_4_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 4>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  5: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_5_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 5>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  6: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_6_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 6>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  7: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_7_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 7>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
    case  8: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_density_contracted_kernel_general_8_sycl, GINTfill_int3c1e_ip2_density_contracted_kernel_general< 8>, output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents) break;
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

    LAUNCH_CONFIG();
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel00_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel00, output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
    // case 01: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 1>
    // case 02: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 2>
    // case 03: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 3>
    // case 04: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<0, 4>
    // case 10: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 0>
    // case 11: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 1>
    // case 12: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 2>
    // case 13: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<1, 3>
    // case 20: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 0>
    // case 21: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 1>
    // case 22: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<2, 2>
    // case 30: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<3, 0>
    // case 31: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<3, 1>
    // case 40: GINTfill_int3c1e_ip2_charge_contracted_kernel_expanded<4, 0>
    default:
        const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
        switch (nrys_roots) {
        case 1: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel_general_1_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel_general<1, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
        case 2: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel_general_2_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel_general<2, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
        case 3: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel_general_3_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel_general<3, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
        case 4: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel_general_4_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
        case 5: LAUNCH_KERNEL(GINTfill_int3c1e_ip2_charge_contracted_kernel_general_5_sycl, GINTfill_int3c1e_ip2_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E>, output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, gridslice, omega, grid_points, charge_exponents) break;
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

#ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
#else // USE_SYCL
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
#endif // USE_SYCL

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

#ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
#else // USE_SYCL
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
#endif

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

#ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
#else // USE_SYCL
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
#endif

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

#ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
#else // USE_SYCL
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
#endif

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

#ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
#else // USE_SYCL
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
#endif

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

#undef LAUNCH_CONFIG
#undef LAUNCH_KERNEL
