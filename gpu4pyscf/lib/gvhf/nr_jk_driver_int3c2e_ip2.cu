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

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/cuda_alloc.cuh"
#include "gint/g2e.h"
#include "gint/cint2e.cuh"

#include "contract_jk.cu"
#include "gint/rys_roots.cu"
#include "gint/g2e.cu"
#include "g3c2e.cuh"
#include "g3c2e_ip2.cu"

__host__
static int GINTrun_tasks_int3c2e_ip2_jk(JKMatrix *jk, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
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
        case   0: GINTint3c2e_ip2_jk_kernel001<<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        // li+lj+lk=1
        case 1: GINTint3c2e_ip2_jk_kernel<0,0,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 10: GINTint3c2e_ip2_jk_kernel<0,1,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 100: GINTint3c2e_ip2_jk_kernel<1,0,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        // li+lj+lk=2
        case 2: GINTint3c2e_ip2_jk_kernel<0,0,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 11: GINTint3c2e_ip2_jk_kernel<0,1,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 20: GINTint3c2e_ip2_jk_kernel<0,2,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 101: GINTint3c2e_ip2_jk_kernel<1,0,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 110: GINTint3c2e_ip2_jk_kernel<1,1,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 200: GINTint3c2e_ip2_jk_kernel<2,0,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        // li+lj+lk=3
        case 3: GINTint3c2e_ip2_jk_kernel<0,0,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 12: GINTint3c2e_ip2_jk_kernel<0,1,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 21: GINTint3c2e_ip2_jk_kernel<0,2,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 30: GINTint3c2e_ip2_jk_kernel<0,3,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 102: GINTint3c2e_ip2_jk_kernel<1,0,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 111: GINTint3c2e_ip2_jk_kernel<1,1,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 120: GINTint3c2e_ip2_jk_kernel<1,2,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 201: GINTint3c2e_ip2_jk_kernel<2,0,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 210: GINTint3c2e_ip2_jk_kernel<2,1,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 300: GINTint3c2e_ip2_jk_kernel<3,0,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
#ifdef UNROLL_INT3C2E
        // li+lj+lk=4
        case 4: GINTint3c2e_ip2_jk_kernel<0,0,4><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 13: GINTint3c2e_ip2_jk_kernel<0,1,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 22: GINTint3c2e_ip2_jk_kernel<0,2,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 31: GINTint3c2e_ip2_jk_kernel<0,3,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 40: GINTint3c2e_ip2_jk_kernel<0,4,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 103: GINTint3c2e_ip2_jk_kernel<1,0,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 112: GINTint3c2e_ip2_jk_kernel<1,1,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 121: GINTint3c2e_ip2_jk_kernel<1,2,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 130: GINTint3c2e_ip2_jk_kernel<1,3,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 202: GINTint3c2e_ip2_jk_kernel<2,0,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 211: GINTint3c2e_ip2_jk_kernel<2,1,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 220: GINTint3c2e_ip2_jk_kernel<2,2,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 301: GINTint3c2e_ip2_jk_kernel<3,0,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 310: GINTint3c2e_ip2_jk_kernel<3,1,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 400: GINTint3c2e_ip2_jk_kernel<4,0,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        // li+lj+lk=5
        case 5: GINTint3c2e_ip2_jk_kernel<0,0,5><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 14: GINTint3c2e_ip2_jk_kernel<0,1,4><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 23: GINTint3c2e_ip2_jk_kernel<0,2,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 32: GINTint3c2e_ip2_jk_kernel<0,3,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 41: GINTint3c2e_ip2_jk_kernel<0,4,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 50: GINTint3c2e_ip2_jk_kernel<0,5,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 104: GINTint3c2e_ip2_jk_kernel<1,0,4><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 113: GINTint3c2e_ip2_jk_kernel<1,1,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 122: GINTint3c2e_ip2_jk_kernel<1,2,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 131: GINTint3c2e_ip2_jk_kernel<1,3,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 140: GINTint3c2e_ip2_jk_kernel<1,4,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 203: GINTint3c2e_ip2_jk_kernel<2,0,3><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 212: GINTint3c2e_ip2_jk_kernel<2,1,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 221: GINTint3c2e_ip2_jk_kernel<2,2,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 230: GINTint3c2e_ip2_jk_kernel<2,3,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 302: GINTint3c2e_ip2_jk_kernel<3,0,2><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 311: GINTint3c2e_ip2_jk_kernel<3,1,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 320: GINTint3c2e_ip2_jk_kernel<3,2,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 401: GINTint3c2e_ip2_jk_kernel<4,0,1><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 410: GINTint3c2e_ip2_jk_kernel<4,1,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 500: GINTint3c2e_ip2_jk_kernel<5,0,0><<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
#endif
        default: switch (nrys_roots) {
            case 2: GINTint3c2e_ip2_jk_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 3: GINTint3c2e_ip2_jk_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 4: GINTint3c2e_ip2_jk_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 5: GINTint3c2e_ip2_jk_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 6: GINTint3c2e_ip2_jk_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 7: GINTint3c2e_ip2_jk_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 8: GINTint3c2e_ip2_jk_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            case 9: GINTint3c2e_ip2_jk_kernel<9, GSIZE9_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
            default: fprintf(stderr, "rys roots %d\n", nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ip2_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


extern "C" { __host__
int GINTbuild_int3c2e_ip2_jk(cudaStream_t stream, BasisProdCache *bpcache,
                 double *vj, double *vk, double *dm, double *rhoj, double *rhok,
                 int *ao_offsets, int nao, int naux, int n_dm,
                 int *bins_locs_ij, int ntasks_kl, int ncp_ij, int cp_kl_id, double omega)
{
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;

    int ng[4] = {0,0,1,0};

    // move bpcache to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    JKMatrix jk;
    jk.n_dm = n_dm;
    jk.nao = nao;
    jk.naux = naux;
    jk.dm = dm;
    jk.vj = vj;
    jk.vk = vk;
    jk.rhoj = rhoj;
    jk.rhok = rhok;
    jk.ao_offsets_i = ao_offsets[0];
    jk.ao_offsets_j = ao_offsets[1];
    jk.ao_offsets_k = ao_offsets[2];
    jk.ao_offsets_l = ao_offsets[3];

    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int cp_ij_id = 0; cp_ij_id < ncp_ij; cp_ij_id++){
        GINTEnvVars envs;
        ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
        GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
        envs.omega = omega;
        if (envs.nrys_roots > 9) {
            return 2;
        }

        int ntasks_ij = bins_locs_ij[cp_ij_id+1] - bins_locs_ij[cp_ij_id];
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ntasks_kl;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id];
        offsets.bas_kl = bas_pairs_locs[cp_kl_id];
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id];
        offsets.primitive_kl = primitive_pairs_locs[cp_kl_id];

        int err = GINTrun_tasks_int3c2e_ip2_jk(&jk, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

}
