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

#ifndef USE_SYCL
#include <cuda_runtime.h>
#endif
#include "cuda_alloc.cuh"
#include "gint.h"
#include "config.h"
#include "g2e.h"

#include "rys_roots.cu"
#include "g2e.cu"
#include "cint2e.cuh"
#include "gout3c2e.cu"
#include "g3c2e_ip2.cu"

__host__
static int GINTfill_int3c2e_ip2_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    #ifdef USE_SYCL
    auto dev_envs = *envs;
    auto dev_eri = *eri;
    auto dev_offsets = *offsets;

    sycl::range<2> threads(THREADSY, THREADSX);
    sycl::range<2> blocks((ntasks_kl+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
    #else
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    #endif
    int li = envs->i_l;
    int lj = envs->j_l;
    int lk = envs->k_l;
    int type_ijk = li * 100 + lj * 10 + lk;

    switch (type_ijk) {
#ifdef USE_SYCL
        case   0: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_000>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel000(dev_envs, dev_eri, dev_offsets); }); break;
        // li+lj+lk=1
        case 1: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_001>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,0,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 10: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_010>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,1,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 100: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_100>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,0,0>(dev_envs, dev_eri, dev_offsets); }); break;
        // li+lj+lk=2
        case 2: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_002>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,0,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 11: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_011>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,1,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 20: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_020>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,2,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 101: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_101>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,0,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 110: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_110>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,1,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 200: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_200>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,0,0>(dev_envs, dev_eri, dev_offsets); }); break;
        // li+lj+lk=3
        case 3: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_003>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,0,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 12: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_012>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,1,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 21: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_021>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,2,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 30: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_030>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,3,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 102: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_102>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,0,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 111: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_111>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,1,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 120: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_120>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,2,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 201: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_201>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,0,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 210: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_210>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,1,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 300: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_300>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,0,0>(dev_envs, dev_eri, dev_offsets); }); break;
        // li+lj+lk=4
        case 4: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_004>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,0,4>(dev_envs, dev_eri, dev_offsets); }); break;
        case 13: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_013>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,1,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 22: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_022>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,2,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 31: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_031>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,3,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 40: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_040>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,4,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 103: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_103>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,0,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 112: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_112>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,1,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 121: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_121>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,2,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 130: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_130>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,3,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 202: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_202>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,0,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 211: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_211>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,1,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 220: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_220>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,2,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 301: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_301>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,0,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 310: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_310>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,1,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 400: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_400>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<4,0,0>(dev_envs, dev_eri, dev_offsets); }); break;
        // li+lj+lk=5
        //case 5: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_005>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,0,5>(dev_envs, dev_eri, dev_offsets); }); break;
        case 14: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_014>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,1,4>(dev_envs, dev_eri, dev_offsets); }); break;
        case 23: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_023>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,2,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 32: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_032>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,3,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 41: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_041>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,4,1>(dev_envs, dev_eri, dev_offsets); }); break;
        //case 50: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_050>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<0,5,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 104: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_104>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,0,4>(dev_envs, dev_eri, dev_offsets); }); break;
        case 113: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_113>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,1,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 122: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_122>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,2,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 131: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_131>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,3,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 140: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_140>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<1,4,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 203: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_203>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,0,3>(dev_envs, dev_eri, dev_offsets); }); break;
        case 212: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_212>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,1,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 221: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_221>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,2,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 230: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_230>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<2,3,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 302: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_302>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,0,2>(dev_envs, dev_eri, dev_offsets); }); break;
        case 311: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_311>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,1,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 320: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_320>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<3,2,0>(dev_envs, dev_eri, dev_offsets); }); break;
        case 401: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_401>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<4,0,1>(dev_envs, dev_eri, dev_offsets); }); break;
        case 410: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_410>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<4,1,0>(dev_envs, dev_eri, dev_offsets); }); break;
        //case 500: stream.parallel_for<class GINTfill_int3c2e_ip2_kernel_500>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ip2_kernel<5,0,0>(dev_envs, dev_eri, dev_offsets); }); break;
#ifdef UNROLL_INT3C2E
#endif
        default: {
            sycl::range<2> threads(1, THREADSX*THREADSY);
            sycl::range<2> blocks(ntasks_kl, ntasks_ij);
            const int lk_ceil = lk + 1;
            const int gsize = 3*nrys_roots*(li+1)*(lj+1)*(lk_ceil+1);
	    stream.submit([&](sycl::handler &cgh) {
		sycl::local_accessor<double, 1> local_acc(sycl::range<1>(gsize+16), cgh);
		cgh.parallel_for<class GINTfill_int3c2e_ip2_general_devkernel>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
                  GINTfill_int3c2e_ip2_general_kernel(*envs, *eri, *offsets, item,
			GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
		}); });
        }
#else // USE_SYCL
        case   0: GINTfill_int3c2e_ip2_kernel000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=1
        case 1: GINTfill_int3c2e_ip2_kernel<0,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 10: GINTfill_int3c2e_ip2_kernel<0,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 100: GINTfill_int3c2e_ip2_kernel<1,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=2
        case 2: GINTfill_int3c2e_ip2_kernel<0,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 11: GINTfill_int3c2e_ip2_kernel<0,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 20: GINTfill_int3c2e_ip2_kernel<0,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 101: GINTfill_int3c2e_ip2_kernel<1,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 110: GINTfill_int3c2e_ip2_kernel<1,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 200: GINTfill_int3c2e_ip2_kernel<2,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=3
        case 3: GINTfill_int3c2e_ip2_kernel<0,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 12: GINTfill_int3c2e_ip2_kernel<0,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 21: GINTfill_int3c2e_ip2_kernel<0,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 30: GINTfill_int3c2e_ip2_kernel<0,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 102: GINTfill_int3c2e_ip2_kernel<1,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 111: GINTfill_int3c2e_ip2_kernel<1,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 120: GINTfill_int3c2e_ip2_kernel<1,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 201: GINTfill_int3c2e_ip2_kernel<2,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 210: GINTfill_int3c2e_ip2_kernel<2,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 300: GINTfill_int3c2e_ip2_kernel<3,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=4
        case 4: GINTfill_int3c2e_ip2_kernel<0,0,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 13: GINTfill_int3c2e_ip2_kernel<0,1,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 22: GINTfill_int3c2e_ip2_kernel<0,2,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 31: GINTfill_int3c2e_ip2_kernel<0,3,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 40: GINTfill_int3c2e_ip2_kernel<0,4,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 103: GINTfill_int3c2e_ip2_kernel<1,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 112: GINTfill_int3c2e_ip2_kernel<1,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 121: GINTfill_int3c2e_ip2_kernel<1,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 130: GINTfill_int3c2e_ip2_kernel<1,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 202: GINTfill_int3c2e_ip2_kernel<2,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 211: GINTfill_int3c2e_ip2_kernel<2,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 220: GINTfill_int3c2e_ip2_kernel<2,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 301: GINTfill_int3c2e_ip2_kernel<3,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 310: GINTfill_int3c2e_ip2_kernel<3,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 400: GINTfill_int3c2e_ip2_kernel<4,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        // li+lj+lk=5
        //case 5: GINTfill_int3c2e_ip2_kernel<0,0,5><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 14: GINTfill_int3c2e_ip2_kernel<0,1,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 23: GINTfill_int3c2e_ip2_kernel<0,2,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 32: GINTfill_int3c2e_ip2_kernel<0,3,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 41: GINTfill_int3c2e_ip2_kernel<0,4,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        //case 50: GINTfill_int3c2e_ip2_kernel<0,5,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 104: GINTfill_int3c2e_ip2_kernel<1,0,4><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 113: GINTfill_int3c2e_ip2_kernel<1,1,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 122: GINTfill_int3c2e_ip2_kernel<1,2,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 131: GINTfill_int3c2e_ip2_kernel<1,3,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 140: GINTfill_int3c2e_ip2_kernel<1,4,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 203: GINTfill_int3c2e_ip2_kernel<2,0,3><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 212: GINTfill_int3c2e_ip2_kernel<2,1,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 221: GINTfill_int3c2e_ip2_kernel<2,2,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 230: GINTfill_int3c2e_ip2_kernel<2,3,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 302: GINTfill_int3c2e_ip2_kernel<3,0,2><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 311: GINTfill_int3c2e_ip2_kernel<3,1,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 320: GINTfill_int3c2e_ip2_kernel<3,2,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 401: GINTfill_int3c2e_ip2_kernel<4,0,1><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 410: GINTfill_int3c2e_ip2_kernel<4,1,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        //case 500: GINTfill_int3c2e_ip2_kernel<5,0,0><<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
#ifdef UNROLL_INT3C2E
#endif
        default: {
            dim3 threads(THREADSX*THREADSY);
            dim3 blocks(ntasks_ij, ntasks_kl);
            const int lk_ceil = lk + 1;
            const int gsize = 3*nrys_roots*(li+1)*(lj+1)*(lk_ceil+1);
            cudaError_t err = cudaFuncSetAttribute(
                GINTfill_int3c2e_ip2_general_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (gsize+16)*sizeof(double));
            const int shm_size = gsize*sizeof(double);
            GINTfill_int3c2e_ip2_general_kernel<<<blocks, threads, shm_size, stream>>>(*envs, *eri, *offsets);
        }
#endif // USE_SYCL
    }

    #ifndef USE_SYCL
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ip2_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

extern "C" {
int GINTfill_int3c2e_ip2(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {0,0,1,0};

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

        int err = GINTfill_int3c2e_ip2_tasks(&eritensor, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
