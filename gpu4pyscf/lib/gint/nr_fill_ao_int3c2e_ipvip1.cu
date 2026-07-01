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
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g2e.cu"
#include "gout3c2e.cu"
#include "g3c2e_ipvip1.cu"

// Abstracts 2D thread/block config (THREADSX/Y swapped between SYCL and CUDA).
// Used 1x in this file.
#ifdef USE_SYCL
#define LAUNCH_CONFIG() \
    sycl::range<2> threads(THREADSY, THREADSX); \
    sycl::range<2> blocks((ntasks_kl+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
#else
#define LAUNCH_CONFIG() \
    dim3 threads(THREADSX, THREADSY); \
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
#endif

// Abstracts 2D kernel launch syntax. dev_envs/dev_eri/dev_offsets are value copies
// hoisted unconditionally so both branches use identical argument names.
// TAG:    unique SYCL class name (ignored on CUDA)
// KERNEL: kernel function (with template args if needed)
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

__host__
static int GINTfill_int3c2e_ipvip1(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    auto dev_envs = *envs;
    auto dev_eri = *eri;
    auto dev_offsets = *offsets;
    LAUNCH_CONFIG();
    int li = envs->i_l;
    int lj = envs->j_l;
    int lk = envs->k_l;
    int type_ijk = li * 100 + lj * 10 + lk;

    switch (type_ijk) {
        // li+lj+lk=0
        case 0: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel000dev, GINTfill_int3c2e_ipvip1_kernel000, dev_envs, dev_eri, dev_offsets) break;
        // li+lj+lk=1
        case 1: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel001, GINTfill_int3c2e_ipvip1_kernel<0,0,1>, dev_envs, dev_eri, dev_offsets) break;
        case 10: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel010, GINTfill_int3c2e_ipvip1_kernel<0,1,0>, dev_envs, dev_eri, dev_offsets) break;
        case 100: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel100, GINTfill_int3c2e_ipvip1_kernel<1,0,0>, dev_envs, dev_eri, dev_offsets) break;
        // li+lj+lk=2
        case 2: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel002, GINTfill_int3c2e_ipvip1_kernel<0,0,2>, dev_envs, dev_eri, dev_offsets) break;
        case 11: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel011, GINTfill_int3c2e_ipvip1_kernel<0,1,1>, dev_envs, dev_eri, dev_offsets) break;
        case 20: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel020, GINTfill_int3c2e_ipvip1_kernel<0,2,0>, dev_envs, dev_eri, dev_offsets) break;
        case 101: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel101, GINTfill_int3c2e_ipvip1_kernel<1,0,1>, dev_envs, dev_eri, dev_offsets) break;
        case 110: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel110, GINTfill_int3c2e_ipvip1_kernel<1,1,0>, dev_envs, dev_eri, dev_offsets) break;
        case 200: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel200, GINTfill_int3c2e_ipvip1_kernel<2,0,0>, dev_envs, dev_eri, dev_offsets) break;
        // li+lj+lk=3
        case 3: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel003, GINTfill_int3c2e_ipvip1_kernel<0,0,3>, dev_envs, dev_eri, dev_offsets) break;
        case 12: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel012, GINTfill_int3c2e_ipvip1_kernel<0,1,2>, dev_envs, dev_eri, dev_offsets) break;
        case 21: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel021, GINTfill_int3c2e_ipvip1_kernel<0,2,1>, dev_envs, dev_eri, dev_offsets) break;
        case 30: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel030, GINTfill_int3c2e_ipvip1_kernel<0,3,0>, dev_envs, dev_eri, dev_offsets) break;
        case 102: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel102, GINTfill_int3c2e_ipvip1_kernel<1,0,2>, dev_envs, dev_eri, dev_offsets) break;
        case 111: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel111, GINTfill_int3c2e_ipvip1_kernel<1,1,1>, dev_envs, dev_eri, dev_offsets) break;
        case 120: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel120, GINTfill_int3c2e_ipvip1_kernel<1,2,0>, dev_envs, dev_eri, dev_offsets) break;
        case 201: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel201, GINTfill_int3c2e_ipvip1_kernel<2,0,1>, dev_envs, dev_eri, dev_offsets) break;
        case 210: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel210, GINTfill_int3c2e_ipvip1_kernel<2,1,0>, dev_envs, dev_eri, dev_offsets) break;
        case 300: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel300, GINTfill_int3c2e_ipvip1_kernel<3,0,0>, dev_envs, dev_eri, dev_offsets) break;
        // li+lj+lk=4
        case 4: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel004, GINTfill_int3c2e_ipvip1_kernel<0,0,4>, dev_envs, dev_eri, dev_offsets) break;
        case 13: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel013, GINTfill_int3c2e_ipvip1_kernel<0,1,3>, dev_envs, dev_eri, dev_offsets) break;
        case 22: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel022, GINTfill_int3c2e_ipvip1_kernel<0,2,2>, dev_envs, dev_eri, dev_offsets) break;
        case 31: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel031, GINTfill_int3c2e_ipvip1_kernel<0,3,1>, dev_envs, dev_eri, dev_offsets) break;
        case 40: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel040, GINTfill_int3c2e_ipvip1_kernel<0,4,0>, dev_envs, dev_eri, dev_offsets) break;
        case 103: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel103, GINTfill_int3c2e_ipvip1_kernel<1,0,3>, dev_envs, dev_eri, dev_offsets) break;
        case 112: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel112, GINTfill_int3c2e_ipvip1_kernel<1,1,2>, dev_envs, dev_eri, dev_offsets) break;
        case 121: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel121, GINTfill_int3c2e_ipvip1_kernel<1,2,1>, dev_envs, dev_eri, dev_offsets) break;
        case 130: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel130, GINTfill_int3c2e_ipvip1_kernel<1,3,0>, dev_envs, dev_eri, dev_offsets) break;
        case 202: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel202, GINTfill_int3c2e_ipvip1_kernel<2,0,2>, dev_envs, dev_eri, dev_offsets) break;
        case 211: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel211, GINTfill_int3c2e_ipvip1_kernel<2,1,1>, dev_envs, dev_eri, dev_offsets) break;
        case 220: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel220, GINTfill_int3c2e_ipvip1_kernel<2,2,0>, dev_envs, dev_eri, dev_offsets) break;
        case 301: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel301, GINTfill_int3c2e_ipvip1_kernel<3,0,1>, dev_envs, dev_eri, dev_offsets) break;
        case 310: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel310, GINTfill_int3c2e_ipvip1_kernel<3,1,0>, dev_envs, dev_eri, dev_offsets) break;
        case 400: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel400, GINTfill_int3c2e_ipvip1_kernel<4,0,0>, dev_envs, dev_eri, dev_offsets) break;
        // li+lj+lk=5
        //case 5: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel005, GINTfill_int3c2e_ipvip1_kernel<0,0,5>, dev_envs, dev_eri, dev_offsets) break;
        case 14: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel014, GINTfill_int3c2e_ipvip1_kernel<0,1,4>, dev_envs, dev_eri, dev_offsets) break;
        case 23: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel023, GINTfill_int3c2e_ipvip1_kernel<0,2,3>, dev_envs, dev_eri, dev_offsets) break;
        case 32: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel032, GINTfill_int3c2e_ipvip1_kernel<0,3,2>, dev_envs, dev_eri, dev_offsets) break;
        case 41: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel041, GINTfill_int3c2e_ipvip1_kernel<0,4,1>, dev_envs, dev_eri, dev_offsets) break;
        //case 50: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel050, GINTfill_int3c2e_ipvip1_kernel<0,5,0>, dev_envs, dev_eri, dev_offsets) break;
        case 104: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel104, GINTfill_int3c2e_ipvip1_kernel<1,0,4>, dev_envs, dev_eri, dev_offsets) break;
        case 113: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel113, GINTfill_int3c2e_ipvip1_kernel<1,1,3>, dev_envs, dev_eri, dev_offsets) break;
        case 122: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel122, GINTfill_int3c2e_ipvip1_kernel<1,2,2>, dev_envs, dev_eri, dev_offsets) break;
        case 131: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel131, GINTfill_int3c2e_ipvip1_kernel<1,3,1>, dev_envs, dev_eri, dev_offsets) break;
        case 140: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel140, GINTfill_int3c2e_ipvip1_kernel<1,4,0>, dev_envs, dev_eri, dev_offsets) break;
        case 203: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel203, GINTfill_int3c2e_ipvip1_kernel<2,0,3>, dev_envs, dev_eri, dev_offsets) break;
        case 212: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel212, GINTfill_int3c2e_ipvip1_kernel<2,1,2>, dev_envs, dev_eri, dev_offsets) break;
        case 221: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel221, GINTfill_int3c2e_ipvip1_kernel<2,2,1>, dev_envs, dev_eri, dev_offsets) break;
        case 230: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel230, GINTfill_int3c2e_ipvip1_kernel<2,3,0>, dev_envs, dev_eri, dev_offsets) break;
        case 302: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel302, GINTfill_int3c2e_ipvip1_kernel<3,0,2>, dev_envs, dev_eri, dev_offsets) break;
        case 311: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel311, GINTfill_int3c2e_ipvip1_kernel<3,1,1>, dev_envs, dev_eri, dev_offsets) break;
        case 320: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel320, GINTfill_int3c2e_ipvip1_kernel<3,2,0>, dev_envs, dev_eri, dev_offsets) break;
        case 401: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel401, GINTfill_int3c2e_ipvip1_kernel<4,0,1>, dev_envs, dev_eri, dev_offsets) break;
        case 410: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel410, GINTfill_int3c2e_ipvip1_kernel<4,1,0>, dev_envs, dev_eri, dev_offsets) break;
        //case 500: LAUNCH_KERNEL(GINTfill_int3c2e_ipvip1_kernel500, GINTfill_int3c2e_ipvip1_kernel<5,0,0>, dev_envs, dev_eri, dev_offsets) break;
#ifdef UNROLL_INT3C2E
#endif
        default: {
            const int li_ceil = li + 1;
            const int lj_ceil = lj + 1;
            const int gsize = 3*nrys_roots*(li_ceil+1)*(lj_ceil+1)*(lk+1);
	    #ifdef USE_SYCL
            sycl::range<2> threads(1, THREADSX*THREADSY);
            sycl::range<2> blocks(ntasks_kl, ntasks_ij);
	    stream.submit([&](sycl::handler &cgh) {
		sycl::local_accessor<double, 1> local_acc(sycl::range<1>(gsize+16), cgh);
		cgh.parallel_for<class GINTfill_int3c2e_ipvip1_general_kernel_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) [[intel::kernel_args_restrict]] {
                  GINTfill_int3c2e_ipvip1_general_kernel(dev_envs, dev_eri, dev_offsets, item,
			GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
		}); });
            #else
            dim3 threads(THREADSX*THREADSY);
            dim3 blocks(ntasks_ij, ntasks_kl);
            cudaError_t err = cudaFuncSetAttribute(
                GINTfill_int3c2e_ipvip1_general_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (gsize+16)*sizeof(double));
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ipvip1_kernel: %s\n", cudaGetErrorString(err));
                return 1;
            }
            const int shm_size = gsize*sizeof(double);
            GINTfill_int3c2e_ipvip1_general_kernel<<<blocks, threads, shm_size, stream>>>(*envs, *eri, *offsets);
            #endif
        }
    }

    #ifndef USE_SYCL
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ipvip1_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}


extern "C" {

int GINTfill_int3c2e_ipvip1(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;

    int ng[4] = {1,1,0,0};

    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 9) {
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

        int err = GINTfill_int3c2e_ipvip1(&eritensor, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

}

#undef LAUNCH_CONFIG
#undef LAUNCH_KERNEL
