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
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g2e.cu"
#include "gout3c2e.cu"
#include "g3c2e_ipip2.cu"

__host__
static int GINTfill_int3c2e_ipip2(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    int li = envs->i_l;
    int lj = envs->j_l;
    int lk = envs->k_l;
    int type_ijk = li * 100 + lj * 10 + lk;
    
    switch (type_ijk) {
        // li+lj+lk=0
        case 0: GINTfill_int3c2e_ipip2_kernel000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=1
        case 1: GINTfill_int3c2e_ipip2_kernel<0,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 10: GINTfill_int3c2e_ipip2_kernel<0,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 100: GINTfill_int3c2e_ipip2_kernel<1,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=2
        case 2: GINTfill_int3c2e_ipip2_kernel<0,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 11: GINTfill_int3c2e_ipip2_kernel<0,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 20: GINTfill_int3c2e_ipip2_kernel<0,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 101: GINTfill_int3c2e_ipip2_kernel<1,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 110: GINTfill_int3c2e_ipip2_kernel<1,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 200: GINTfill_int3c2e_ipip2_kernel<2,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=3
        case 3: GINTfill_int3c2e_ipip2_kernel<0,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 12: GINTfill_int3c2e_ipip2_kernel<0,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 21: GINTfill_int3c2e_ipip2_kernel<0,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 30: GINTfill_int3c2e_ipip2_kernel<0,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 102: GINTfill_int3c2e_ipip2_kernel<1,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 111: GINTfill_int3c2e_ipip2_kernel<1,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 120: GINTfill_int3c2e_ipip2_kernel<1,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 201: GINTfill_int3c2e_ipip2_kernel<2,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 210: GINTfill_int3c2e_ipip2_kernel<2,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 300: GINTfill_int3c2e_ipip2_kernel<3,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
#ifdef UNROLL_INT3C2E
        // li+lj+lk=4
        case 4: GINTfill_int3c2e_ipip2_kernel<0,0,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 13: GINTfill_int3c2e_ipip2_kernel<0,1,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 22: GINTfill_int3c2e_ipip2_kernel<0,2,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 31: GINTfill_int3c2e_ipip2_kernel<0,3,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 40: GINTfill_int3c2e_ipip2_kernel<0,4,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 103: GINTfill_int3c2e_ipip2_kernel<1,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 112: GINTfill_int3c2e_ipip2_kernel<1,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 121: GINTfill_int3c2e_ipip2_kernel<1,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 130: GINTfill_int3c2e_ipip2_kernel<1,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 202: GINTfill_int3c2e_ipip2_kernel<2,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 211: GINTfill_int3c2e_ipip2_kernel<2,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 220: GINTfill_int3c2e_ipip2_kernel<2,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 301: GINTfill_int3c2e_ipip2_kernel<3,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 310: GINTfill_int3c2e_ipip2_kernel<3,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 400: GINTfill_int3c2e_ipip2_kernel<4,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=5
        case 5: GINTfill_int3c2e_ipip2_kernel<0,0,5><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 14: GINTfill_int3c2e_ipip2_kernel<0,1,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 23: GINTfill_int3c2e_ipip2_kernel<0,2,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 32: GINTfill_int3c2e_ipip2_kernel<0,3,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 41: GINTfill_int3c2e_ipip2_kernel<0,4,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 50: GINTfill_int3c2e_ipip2_kernel<0,5,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 104: GINTfill_int3c2e_ipip2_kernel<1,0,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 113: GINTfill_int3c2e_ipip2_kernel<1,1,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 122: GINTfill_int3c2e_ipip2_kernel<1,2,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 131: GINTfill_int3c2e_ipip2_kernel<1,3,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 140: GINTfill_int3c2e_ipip2_kernel<1,4,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 203: GINTfill_int3c2e_ipip2_kernel<2,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 212: GINTfill_int3c2e_ipip2_kernel<2,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 221: GINTfill_int3c2e_ipip2_kernel<2,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 230: GINTfill_int3c2e_ipip2_kernel<2,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 302: GINTfill_int3c2e_ipip2_kernel<3,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 311: GINTfill_int3c2e_ipip2_kernel<3,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 320: GINTfill_int3c2e_ipip2_kernel<3,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 401: GINTfill_int3c2e_ipip2_kernel<4,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 410: GINTfill_int3c2e_ipip2_kernel<4,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 500: GINTfill_int3c2e_ipip2_kernel<5,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
#endif
        default: switch (nrys_roots) {
            case 2: GINTfill_int3c2e_ipip2_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 3: GINTfill_int3c2e_ipip2_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 4: GINTfill_int3c2e_ipip2_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 5: GINTfill_int3c2e_ipip2_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 6: GINTfill_int3c2e_ipip2_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 7: GINTfill_int3c2e_ipip2_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 8: GINTfill_int3c2e_ipip2_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            case 9: GINTfill_int3c2e_ipip2_kernel<9, GSIZE9_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            default: fprintf(stderr, "rys roots %d\n", nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ipip2_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


extern "C" {

int GINTfill_int3c2e_ipip2(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;

    int ng[4] = {0,0,2,0};

    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 9) {
        return 2;
    }

    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    ERITensor eritensor;
    eritensor.stride_j = strides[1];
    eritensor.stride_k = strides[2];
    eritensor.stride_l = strides[3];
    eritensor.ao_offsets_i = ao_offsets[0];
    eritensor.ao_offsets_j = ao_offsets[1];
    eritensor.ao_offsets_k = ao_offsets[2];
    eritensor.ao_offsets_l = ao_offsets[3];
    eritensor.nao = nao;
    eritensor.data = eri;
    BasisProdOffsets offsets;

    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int kl_bin = 0; kl_bin < nbins; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin+1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }
        // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
        int ij_bin1 = nbins - kl_bin;
        int bas_ij0 = bins_locs_ij[0];
        int bas_ij1 = bins_locs_ij[ij_bin1];
        int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ntasks_kl;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * envs.nprim_ij;
        offsets.primitive_kl = primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;

        int err = GINTfill_int3c2e_ipip2(&eritensor, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

}
