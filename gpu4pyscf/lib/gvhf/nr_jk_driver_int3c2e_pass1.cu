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

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/cuda_alloc.cuh"
#include "gint/g2e.h"
#include "gint/cint2e.cuh"

#include "contract_jk.cu"
#include "gint/rys_roots.cu"
#include "gint/g2e.cu"
#include "g3c2e.cuh"
#include "g3c2e_pass1.cu"

__host__
static int GINTrun_tasks_int3c2e_pass1_j(JKMatrix *jk, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);

    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (envs->nrys_roots) {
        case 1:
            type_ijkl = (envs->i_l << 3) | (envs->j_l << 2) | (envs->k_l << 1) | envs->l_l;
            switch (type_ijkl) {
                case 0b0000: GINTint3c2e_pass1_j_kernel0000<<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
                case 0b0010: GINTint3c2e_pass1_j_kernel0010<<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
                case 0b1000: GINTint3c2e_pass1_j_kernel1000<<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
                default: fprintf(stderr, "rys roots 1 type_ijkl %d\n", type_ijkl);
                return 1;
                }
            break;
        case 2: GINTint3c2e_pass1_j_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 3: GINTint3c2e_pass1_j_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 4: GINTint3c2e_pass1_j_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 5: GINTint3c2e_pass1_j_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 6: GINTint3c2e_pass1_j_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 7: GINTint3c2e_pass1_j_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 8: GINTint3c2e_pass1_j_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        case 9: GINTint3c2e_pass1_j_kernel<9, GSIZE9_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *jk, *offsets); break;
        default: fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTint2e_jk_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


extern "C" { __host__
int GINTbuild_j_int3c2e_pass1(cudaStream_t stream, BasisProdCache *bpcache,
                 double *dm, double *rhoj,
                 int nao, int naux, int n_dm,
                 int *bins_locs_ij, int *bins_locs_kl,
                 int ncp_ij, int ncp_kl)
{
    // move bpcache to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    int ng[4] = {0,0,0,0};

    JKMatrix jk;
    jk.n_dm = n_dm;
    jk.nao = nao;
    jk.naux = naux;
    jk.dm = dm;
    jk.rhoj = rhoj;
    jk.ao_offsets_i = 0;
    jk.ao_offsets_j = 0;
    jk.ao_offsets_k = nao + 1;
    jk.ao_offsets_l = nao;

    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    /*
    int *idx = (int *)malloc(sizeof(int) * TOT_NF * 3);
    int *l_locs = (int *)malloc(sizeof(int) * (GPU_LMAX + 2));
    GINTinit_index1d_xyz(idx, l_locs);
    
    for (int i = 0; i < 3*TOT_NF; i++){
        printf("%d, ", idx[i]);
    }
    printf("\n");
    checkCudaErrors(cudaMemcpyToSymbol(c_idx, idx, sizeof(int) * TOT_NF*3));
    checkCudaErrors(cudaMemcpyToSymbol(c_l_locs, l_locs, sizeof(int) * (GPU_LMAX + 2)));
    free(idx);
    free(l_locs);
    */
    for (int cp_ij_id = 0; cp_ij_id < ncp_ij; cp_ij_id++){
        for (int k = 0; k < ncp_kl; k++){
            int cp_kl_id = k + ncp_ij;
            ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
            ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
            GINTEnvVars envs;

            GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
            envs.omega = 0.0;
            if (envs.nrys_roots > 9) {
                return 2;
            }

            int ntasks_ij = bins_locs_ij[cp_ij_id+1] - bins_locs_ij[cp_ij_id];
            int ntasks_kl = bins_locs_kl[k+1] - bins_locs_kl[k];
            if (ntasks_kl <= 0) continue;
            if (ntasks_ij <= 0) continue;

            BasisProdOffsets offsets;
            offsets.ntasks_ij = ntasks_ij;
            offsets.ntasks_kl = ntasks_kl;
            offsets.bas_ij = bas_pairs_locs[cp_ij_id];
            offsets.bas_kl = bas_pairs_locs[cp_kl_id];
            offsets.primitive_ij = primitive_pairs_locs[cp_ij_id];
            offsets.primitive_kl = primitive_pairs_locs[cp_kl_id];
            int err = GINTrun_tasks_int3c2e_pass1_j(&jk, &offsets, &envs, stream);
            if (err != 0) {
                return err;
            }
        }
    }
    return 0;
}


}
