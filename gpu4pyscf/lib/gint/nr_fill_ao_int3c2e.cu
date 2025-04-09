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
#include "config.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"
#include "g2e.h"

#include "rys_roots.cu"
#include "g2e.cu"
#include "gout3c2e.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"
#include "g3c2e.cu"

static int GINTfill_int3c2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
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
    const int type_ijkl = li * 100 + lj * 10 + lk;
    switch (type_ijkl) {
        // nroots = 1
        case 0: GINTfill_int3c2e_kernel0000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 1: GINTfill_int3c2e_kernel0010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 10: GINTfill_int3c2e_kernel0100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 100: GINTfill_int3c2e_kernel1000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // nroots = 2
        case 2: GINTfill_int2e_kernel0020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 3: GINTfill_int2e_kernel0030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 101: GINTfill_int2e_kernel1010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 102: GINTfill_int2e_kernel1020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 110: GINTfill_int2e_kernel1100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 111: GINTfill_int2e_kernel1110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 200: GINTfill_int2e_kernel2000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 201: GINTfill_int2e_kernel2010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 210: GINTfill_int2e_kernel2100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 300: GINTfill_int2e_kernel3000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // nroots = 3
        case 103: GINTfill_int2e_kernel1030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 112: GINTfill_int2e_kernel1120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 113: GINTfill_int2e_kernel1130<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 202: GINTfill_int2e_kernel2020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 203: GINTfill_int2e_kernel2030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 211: GINTfill_int2e_kernel2110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 212: GINTfill_int2e_kernel2120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 220: GINTfill_int2e_kernel2200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 221: GINTfill_int2e_kernel2210<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 301: GINTfill_int2e_kernel3010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 302: GINTfill_int2e_kernel3020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 310: GINTfill_int2e_kernel3100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 311: GINTfill_int2e_kernel3110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 320: GINTfill_int2e_kernel3200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default: {
            dim3 threads(THREADSX*THREADSY);
            dim3 blocks(ntasks_ij, ntasks_kl);
            const int gsize = 3*nrys_roots*(li+1)*(lj+1)*(lk+1);
            cudaError_t err = cudaFuncSetAttribute(
                GINTfill_int3c2e_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (gsize+16)*sizeof(double));
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error of GINTfill_int3c2e_kernel: %s\n", cudaGetErrorString(err));
                return 1;
            }
            const int shm_size = gsize*sizeof(double);
            GINTfill_int3c2e_kernel<<<blocks, threads, shm_size, stream>>>(*envs, *eri, *offsets);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int3c2e(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {0,0,0,0};

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

        int err = -1;
        err = GINTfill_int3c2e_tasks(&eritensor, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
