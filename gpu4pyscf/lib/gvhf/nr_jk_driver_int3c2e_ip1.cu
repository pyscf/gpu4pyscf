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
#include "gint/g2e.cu"

#include "contract_jk.cu"
#include "gint/rys_roots.cu"

#include "g3c2e.cuh"
#include "g3c2e_ip1.cu"

__host__
static int GINTrun_tasks_int3c2e_ip1_jk(JKMatrix *jk, BasisProdOffsets *offsets, GINTEnvVars *envs)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);

    switch (envs->nrys_roots) {
        case 1: GINTrun_int3c2e_ip1_jk_kernel1000<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 2: GINTint3c2e_ip1_jk_kernel<2, GSIZE2_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 3: GINTint3c2e_ip1_jk_kernel<3, GSIZE3_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 4: GINTint3c2e_ip1_jk_kernel<4, GSIZE4_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 5: GINTint3c2e_ip1_jk_kernel<5, GSIZE5_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 6: GINTint3c2e_ip1_jk_kernel<6, GSIZE6_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 7: GINTint3c2e_ip1_jk_kernel<7, GSIZE7_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case 8: GINTint3c2e_ip1_jk_kernel<8, GSIZE8_INT3C> <<<blocks, threads>>>(*envs, *jk, *offsets); break;
        default:
            fprintf(stderr, "rys roots %d\n", nrys_roots);
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
int GINTbuild_int3c2e_ip1_jk(BasisProdCache *bpcache,
                 double *vj, double *vk, double *dm, double *rhoj, double *rhok, 
                 int *ao_offsets, int nao, int naux, int n_dm,
                 int *bins_locs_ij, int *bins_locs_kl, int nbins,
                 int cp_ij_id, int cp_kl_id, int ip_type, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {1,0,0,0};
    
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 8) {
        return 2;
    }
    
    // TODO: improve the efficiency by unrolling
    if (envs.nrys_roots > 1) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        GINTg2e_index_xyz(idx4c, &envs);
        checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
        free(idx4c);
    }
    
    int kl_bin, ij_bin1;
    
    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
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
    BasisProdOffsets offsets;
    
    int *bas_pairs_locs = bpcache->bas_pairs_locs;
    int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (kl_bin = 0; kl_bin < nbins; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin+1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }
        // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
        ij_bin1 = nbins - kl_bin;
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

        int err = GINTrun_tasks_int3c2e_ip1_jk(&jk, &offsets, &envs);
        if (err != 0) {
            return err;
        }
    }
    
    return 0;
}


}
