/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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


static void CINTcart_comp(int16_t *nx, int16_t *ny, int16_t *nz, int lmax)
{
    int inc = 0;
    for (int16_t lx = lmax; lx >= 0; lx--) {
        for (int16_t ly = lmax - lx; ly >= 0; ly--) {
            int16_t lz = lmax - lx - ly;
            nx[inc] = lx;
            ny[inc] = ly;
            nz[inc] = lz;
            inc++;
        }
    }
}

static int GINTfill_int3c1e_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                  const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                  const double omega, const double* grid_points, const cudaStream_t stream)
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
        case (0<<2)|0: GINTfill_int3c1e_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
        case (1<<2)|0: GINTfill_int3c1e_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    // case 2:
    //     type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
    //     switch (type_ijkl) {
    //     case (0<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel0020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (0<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel0030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel1010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel1020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel1100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel1110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel2000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel2010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel2100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel3000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     default:
    //         GINTfill_int3c2e_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     }
    //     break;
    // case 3:
    //     type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
    //     switch (type_ijkl) {
    //     case (1<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel1030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(1<<4)|(2<<2)|0: GINTfill_int2e_kernel1120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (1<<6)|(1<<4)|(3<<2)|0: GINTfill_int2e_kernel1130<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel2020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel2030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel2110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(1<<4)|(2<<2)|0: GINTfill_int2e_kernel2120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(2<<4)|(0<<2)|0: GINTfill_int2e_kernel2200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (2<<6)|(2<<4)|(1<<2)|0: GINTfill_int2e_kernel2210<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel3010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel3020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel3100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel3110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     case (3<<6)|(2<<4)|(0<<2)|0: GINTfill_int2e_kernel3200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     default:
    //         GINTfill_int3c2e_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    //     }
    //     break;
    case 2: GINTfill_int3c1e_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
    case 3: GINTfill_int3c1e_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
    case 4: GINTfill_int3c1e_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
    case 5: GINTfill_int3c1e_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int3c1e(const cudaStream_t stream, const BasisProdCache* bpcache,
                     const double* grid_points, const int ngrids,
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

    if (nrys_roots > 9) {
        return 2;
    }

    if (nrys_roots > 1) {
        int16_t cart_component[GPU_CART_MAX * 6] {0};
        CINTcart_comp(cart_component + 0 * GPU_CART_MAX, cart_component + 1 * GPU_CART_MAX, cart_component + 2 * GPU_CART_MAX, i_l);
        CINTcart_comp(cart_component + 3 * GPU_CART_MAX, cart_component + 4 * GPU_CART_MAX, cart_component + 5 * GPU_CART_MAX, j_l);

        checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, cart_component, sizeof(int16_t) * GPU_CART_MAX * 6));
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
                                               omega, grid_points, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
