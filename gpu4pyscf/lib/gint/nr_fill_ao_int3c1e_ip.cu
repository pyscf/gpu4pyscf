/* Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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
// #include "g1e_root_123.cu"
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
    // case 1:
    //     type_ijkl = (i_l << 2) | j_l;
    //     switch (type_ijkl) {
    //     case (0<<2)|0: GINTfill_int3c1e_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    //     case (1<<2)|0: GINTfill_int3c1e_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    //     default:
    //         fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
    //     }
    //     break;
    case 1: GINTfill_int3c1e_ip_kernel_general<1, GSIZE1_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
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

extern "C" {
int GINTfill_int3c1e_ip(const cudaStream_t stream, const BasisProdCache* bpcache,
                        const double* grid_points, const double* charge_exponents, const int ngrids,
                        double* integrals, const int nao,
                        const int* strides, const int* ao_offsets,
                        const int* bins_locs_ij, int nbins,
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
}
