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
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"

#include "rys_roots.cu"
#include "g2e.cu"
#include "cint2e.cuh"
#include "gout2e.cuh"

#include "fill_ints.cu"
#include "g2e_root1.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"
#include "g2e_root_n.cu"

// Abstracts 2D kernel launch syntax.  dev_envs/dev_eri/dev_offsets are value
// copies hoisted unconditionally so both branches use identical argument names.
// TAG:    unique SYCL class name (ignored on CUDA)
// KERNEL: kernel function (with template args if needed)
// ...:    kernel arguments
#define GINT_CAT_(a, b) a##b
#define GINT_CAT(a, b)  GINT_CAT_(a, b)
#ifdef USE_SYCL
// Kernel-id (with any template args) is the trailing __VA_ARGS__ so its commas
// survive macro expansion. SYCL kernel name is generated inline per source line.
// dev_envs/dev_eri/dev_offsets are on-host value copies made just before launch
// for lambda capture.
#define LAUNCH_KERNEL(...) { \
    auto dev_envs = *envs; auto dev_eri = *eri; auto dev_offsets = *offsets; \
    stream.parallel_for<class GINT_CAT(gint_kernel_L, __LINE__)>( \
        sycl::nd_range<2>(blocks * threads, threads), \
        [=](auto item) [[intel::kernel_args_restrict]] { \
            __VA_ARGS__(dev_envs, dev_eri, dev_offsets); }); }
#else
// CUDA passes the dereferenced structs by value at launch, like master.
#define LAUNCH_KERNEL(...) \
    __VA_ARGS__ <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets);
#endif

__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    int type_ijkl;
    #ifdef USE_SYCL
    sycl::range<2> threads(THREADSY, THREADSX);
    sycl::range<2> blocks((ntasks_kl+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
    #else
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    #endif
    switch (nrys_roots) {
    case 1:
        type_ijkl = (envs->i_l << 3) | (envs->j_l << 2) | (envs->k_l << 1) | envs->l_l;
        switch (type_ijkl) {
        case 0b0000: LAUNCH_KERNEL(GINTfill_int2e_kernel0000) break;
        case 0b0010: LAUNCH_KERNEL(GINTfill_int2e_kernel0010) break;
        case 0b1000: LAUNCH_KERNEL(GINTfill_int2e_kernel1000) break;
        default:
            //GINTfill_int2e_kernel<1, GOUTSIZE1> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (0<<6)|(0<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel0011) break;
        case (0<<6)|(0<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel0020) break;
        case (0<<6)|(0<<4)|(2<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel0021) break;
        case (0<<6)|(0<<4)|(3<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel0030) break;
        case (1<<6)|(0<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1010) break;
        case (1<<6)|(0<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel1011) break;
        case (1<<6)|(0<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1020) break;
        case (1<<6)|(1<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1100) break;
        case (1<<6)|(1<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1110) break;
        case (2<<6)|(0<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2000) break;
        case (2<<6)|(0<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2010) break;
        case (2<<6)|(1<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2100) break;
        case (3<<6)|(0<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3000) break;
        default: LAUNCH_KERNEL(GINTfill_int2e_kernel<2, GOUTSIZE2>) break;
        }
        break;
    case 3:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (0<<6)|(0<<4)|(2<<2)|2: LAUNCH_KERNEL(GINTfill_int2e_kernel0022) break;
        case (0<<6)|(0<<4)|(3<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel0031) break;
        case (0<<6)|(0<<4)|(3<<2)|2: LAUNCH_KERNEL(GINTfill_int2e_kernel0032) break;
        case (1<<6)|(0<<4)|(2<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel1021) break;
        case (1<<6)|(0<<4)|(2<<2)|2: LAUNCH_KERNEL(GINTfill_int2e_kernel1022) break;
        case (1<<6)|(0<<4)|(3<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1030) break;
        case (1<<6)|(0<<4)|(3<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel1031) break;
        case (1<<6)|(1<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel1111) break;
        case (1<<6)|(1<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1120) break;
        case (1<<6)|(1<<4)|(2<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel1121) break;
        case (1<<6)|(1<<4)|(3<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel1130) break;
        case (2<<6)|(0<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel2011) break;
        case (2<<6)|(0<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2020) break;
        case (2<<6)|(0<<4)|(2<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel2021) break;
        case (2<<6)|(0<<4)|(3<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2030) break;
        case (2<<6)|(1<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2110) break;
        case (2<<6)|(1<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel2111) break;
        case (2<<6)|(1<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2120) break;
        case (2<<6)|(2<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2200) break;
        case (2<<6)|(2<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel2210) break;
        case (3<<6)|(0<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3010) break;
        case (3<<6)|(0<<4)|(1<<2)|1: LAUNCH_KERNEL(GINTfill_int2e_kernel3011) break;
        case (3<<6)|(0<<4)|(2<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3020) break;
        case (3<<6)|(1<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3100) break;
        case (3<<6)|(1<<4)|(1<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3110) break;
        case (3<<6)|(2<<4)|(0<<2)|0: LAUNCH_KERNEL(GINTfill_int2e_kernel3200) break;
        default: LAUNCH_KERNEL(GINTfill_int2e_kernel<3, GOUTSIZE3>) break;
        }
        break;

    case 4: LAUNCH_KERNEL(GINTfill_int2e_kernel<4, GOUTSIZE4>) break;
    case 5: LAUNCH_KERNEL(GINTfill_int2e_kernel<5, GOUTSIZE5>) break;
    case 6: LAUNCH_KERNEL(GINTfill_int2e_kernel<6, GOUTSIZE6>) break;
    case 7: LAUNCH_KERNEL(GINTfill_int2e_kernel<7, GOUTSIZE7>) break;
    case 8: LAUNCH_KERNEL(GINTfill_int2e_kernel<8, GOUTSIZE8>) break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int2e_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int2e(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl,
                   double *bins_floor_ij, double *bins_floor_kl,
                   int nbins_ij, int nbins_kl,
                   int cp_ij_id, int cp_kl_id, double log_cutoff, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {0,0,0,0};
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > POLYFIT_ORDER) {
        fprintf(stderr, "GINTfill_int2e: unsupported rys order %d\n", envs.nrys_roots);
        return 2;
    }

    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    #ifdef USE_SYCL
    stream.memcpy(s_bpcache, bpcache, sizeof(BasisProdCache)).wait();
    #else
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    #endif
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
    for (int kl_bin = 0; kl_bin < nbins_kl; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin+1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }

        // ij_bin1 is the index of first bin out of cutoff
        int ij_bin1 = 0;
        double log_q_kl_bin, log_q_ij_bin;
        log_q_kl_bin = bins_floor_kl[kl_bin];
        for(int ij_bin = 0; ij_bin < nbins_ij; ij_bin++){
            log_q_ij_bin = bins_floor_ij[ij_bin];
            if (log_q_ij_bin + log_q_kl_bin < log_cutoff){
                break;
            }
            ij_bin1++;
        }

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

        int err = GINTfill_int2e_tasks(&eritensor, &offsets, &envs, stream);
        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}

#undef LAUNCH_KERNEL
