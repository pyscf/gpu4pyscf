/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
#include "ecp.h"
#include "bessel.cu"
#include "cart2sph.cu"
#include "gauss_chebyshev.cu"
#include "common.cu"
#include "type1_ang_nuc.cu"
#include "type2_ang_nuc.cu"
#include "ecp_type1.cu"
#include "ecp_type2.cu"
#include "ecp_type1_ip.cu"
#include "ecp_type2_ip.cu"
#include "ecp_type1_ipip.cu"
#include "ecp_type2_ipip.cu"

#define ECP_ARGS  gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env

#ifdef USE_SYCL
#define ECP_LAUNCH1(TAG, KPREFIX, LI, LJ) \
    stream.parallel_for<class TAG>( \
        sycl::nd_range<1>(blocks * threads, threads), \
        [=](auto item) [[intel::kernel_args_restrict]] { \
            KPREFIX<LI,LJ>(ECP_ARGS); \
        })
#else
#define ECP_LAUNCH1(TAG, KPREFIX, LI, LJ) \
    KPREFIX<LI,LJ><<<blocks, threads>>>(ECP_ARGS)
#endif

#ifdef USE_SYCL
#define ECP_LAUNCH2(TAG, KPREFIX, LI, LJ, LC) \
    stream.parallel_for<class TAG>( \
        sycl::nd_range<1>(blocks * threads, threads), \
        [=](auto item) [[intel::kernel_args_restrict]] { \
            KPREFIX<LI,LJ,LC>(ECP_ARGS); \
        })
#else
#define ECP_LAUNCH2(TAG, KPREFIX, LI, LJ, LC) \
    KPREFIX<LI,LJ,LC><<<blocks, threads>>>(ECP_ARGS)
#endif

#ifdef USE_SYCL
#define ECP_LAUNCH_GENERAL(TAG, SMEM, KFUNC, ...) \
    stream.submit([&](sycl::handler &cgh) { \
        sycl::local_accessor<double, 1> local_acc(sycl::range<1>(SMEM), cgh); \
        cgh.parallel_for<class TAG>( \
            sycl::nd_range<1>(blocks * threads, threads), \
            [=](auto item) [[intel::kernel_args_restrict]] { \
                KFUNC(__VA_ARGS__, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); \
            }); \
    })
#else
#define ECP_LAUNCH_GENERAL(TAG, SMEM, KFUNC, ...) do { \
    cudaError_t _e = cudaFuncSetAttribute( \
        KFUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, (SMEM)*sizeof(double)); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s: %s\n", \
                __func__, cudaGetErrorString(_e)); \
        return 1; \
    } \
    KFUNC<<<blocks, threads, (SMEM)*sizeof(double)>>>(__VA_ARGS__); \
} while(0)
#endif

extern "C" {
int ECP_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
#ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(ntasks);
    sycl::queue& stream = *sycl_get_queue();
#else
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
#endif
    if (lc >= 0){
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        case 0:   ECP_LAUNCH2(type2_cart_000, type2_cart, 0,0,0); break;
        case 1:   ECP_LAUNCH2(type2_cart_001, type2_cart, 0,0,1); break;
        case 2:   ECP_LAUNCH2(type2_cart_002, type2_cart, 0,0,2); break;
        case 3:   ECP_LAUNCH2(type2_cart_003, type2_cart, 0,0,3); break;
        case 10:  ECP_LAUNCH2(type2_cart_010, type2_cart, 0,1,0); break;
        case 11:  ECP_LAUNCH2(type2_cart_011, type2_cart, 0,1,1); break;
        case 12:  ECP_LAUNCH2(type2_cart_012, type2_cart, 0,1,2); break;
        case 110: ECP_LAUNCH2(type2_cart_110, type2_cart, 1,1,0); break;
        case 111: ECP_LAUNCH2(type2_cart_111, type2_cart, 1,1,1); break;
        case 112: ECP_LAUNCH2(type2_cart_112, type2_cart, 1,1,2); break;
        case 20:  ECP_LAUNCH2(type2_cart_020, type2_cart, 0,2,0); break;
        case 21:  ECP_LAUNCH2(type2_cart_021, type2_cart, 0,2,1); break;
        case 30:  ECP_LAUNCH2(type2_cart_030, type2_cart, 0,3,0); break;
        case 120: ECP_LAUNCH2(type2_cart_120, type2_cart, 1,2,0); break;
        // General kernel
        default: {
            const int li1 = li+1;
            const int lj1 = lj+1;
            const int nfi = (li+1)*(li+2)/2;
            const int nfj = (lj+1)*(lj+2)/2;
            const int lic1 = li+lc+1;
            const int ljc1 = lj+lc+1;
            const int lcc1 = 2*lc+1;
            const int blki = (lic1+1)/2 * lcc1;
            const int blkj = (ljc1+1)/2 * lcc1;

            int smem_size0 = (li+lj+1) * lic1 * ljc1; // rad_all
            int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
            int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
            int smem_size3 = li1*nfi*lic1; // angi
            int smem_size4 = lj1*nfj*ljc1; // angj
            int smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;

            ECP_LAUNCH_GENERAL(type2_cart_sycl, smem_size, type2_cart,
                               gctr, li, lj, lc, ao_loc, nao,
                               tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        }}
    } else {
        int task_type = li * 10 + lj;
        switch (task_type)
        {
        case 0:  ECP_LAUNCH1(type1_cart_00, type1_cart, 0,0); break;
        case 1:  ECP_LAUNCH1(type1_cart_01, type1_cart, 0,1); break;
        case 11: ECP_LAUNCH1(type1_cart_11, type1_cart, 1,1); break;
        case 2:  ECP_LAUNCH1(type1_cart_02, type1_cart, 0,2); break;
        case 3:  ECP_LAUNCH1(type1_cart_03, type1_cart, 0,3); break;
        case 12: ECP_LAUNCH1(type1_cart_12, type1_cart, 1,2); break;
        case 4:  ECP_LAUNCH1(type1_cart_04, type1_cart, 0,4); break;
        case 13: ECP_LAUNCH1(type1_cart_13, type1_cart, 1,3); break;
        case 22: ECP_LAUNCH1(type1_cart_22, type1_cart, 2,2); break;
        default: {
            const int lij1 = li+lj+1;
            const int lij3 = lij1*lij1*lij1;
            int smem_size = lij3 + lij1*lij1;

            ECP_LAUNCH_GENERAL(type1_cart_kernel, smem_size, type1_cart,
                               gctr, li, lj, ao_loc, nao,
                               tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ip_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
#ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(ntasks);
    sycl::queue& stream = *sycl_get_queue();
#else
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
#endif
    if (lc < 0){
        int task_type = li * 10 + lj;
        switch (task_type) {
        case 0:  ECP_LAUNCH1(type1_cart_ip1_00, type1_cart_ip1, 0,0); break;
        case 1:  ECP_LAUNCH1(type1_cart_ip1_01, type1_cart_ip1, 0,1); break;
        case 11: ECP_LAUNCH1(type1_cart_ip1_11, type1_cart_ip1, 1,1); break;
        case 2:  ECP_LAUNCH1(type1_cart_ip1_02, type1_cart_ip1, 0,2); break;
        case 3:  ECP_LAUNCH1(type1_cart_ip1_03, type1_cart_ip1, 0,3); break;
        case 12: ECP_LAUNCH1(type1_cart_ip1_12, type1_cart_ip1, 1,2); break;
        case 4:  ECP_LAUNCH1(type1_cart_ip1_04, type1_cart_ip1, 0,4); break;
        case 13: ECP_LAUNCH1(type1_cart_ip1_13, type1_cart_ip1, 1,3); break;
        case 22: ECP_LAUNCH1(type1_cart_ip1_22, type1_cart_ip1, 2,2); break;
        default: {
            const int lij1 = li+lj+2;
            const int lij3 = lij1*lij1*lij1;
            int smem_size = lij3 + lij1*lij1;

            ECP_LAUNCH_GENERAL(type1_cart_ip1_general_kernel, smem_size,
                               type1_cart_ip1_general,
                               gctr, li, lj, ao_loc, nao,
                               tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        }}
    } else {
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        case 0:   ECP_LAUNCH2(type2_cart_ip1_000, type2_cart_ip1, 0,0,0); break;
        case 1:   ECP_LAUNCH2(type2_cart_ip1_001, type2_cart_ip1, 0,0,1); break;
        case 2:   ECP_LAUNCH2(type2_cart_ip1_002, type2_cart_ip1, 0,0,2); break;
        case 3:   ECP_LAUNCH2(type2_cart_ip1_003, type2_cart_ip1, 0,0,3); break;
        case 10:  ECP_LAUNCH2(type2_cart_ip1_010, type2_cart_ip1, 0,1,0); break;
        case 11:  ECP_LAUNCH2(type2_cart_ip1_011, type2_cart_ip1, 0,1,1); break;
        case 12:  ECP_LAUNCH2(type2_cart_ip1_012, type2_cart_ip1, 0,1,2); break;
        case 110: ECP_LAUNCH2(type2_cart_ip1_110, type2_cart_ip1, 1,1,0); break;
        case 111: ECP_LAUNCH2(type2_cart_ip1_111, type2_cart_ip1, 1,1,1); break;
        case 112: ECP_LAUNCH2(type2_cart_ip1_112, type2_cart_ip1, 1,1,2); break;
        case 20:  ECP_LAUNCH2(type2_cart_ip1_020, type2_cart_ip1, 0,2,0); break;
        case 21:  ECP_LAUNCH2(type2_cart_ip1_021, type2_cart_ip1, 0,2,1); break;
        case 30:  ECP_LAUNCH2(type2_cart_ip1_030, type2_cart_ip1, 0,3,0); break;
        case 120: ECP_LAUNCH2(type2_cart_ip1_120, type2_cart_ip1, 1,2,0); break;
        // General kernel
        default: {
            const int li1 = li+2;
            const int lj1 = lj+1;
            const int lij1 = (li+1)+lj+1;
            const int nfi = (li+2)*(li+3)/2;
            const int nfj = (lj+1)*(lj+2)/2;
            const int lic1 = li1+lc+1;
            const int ljc1 = lj1+lc+1;
            const int lcc1 = 2*lc+1;
            const int blki = (lic1+1)/2 * lcc1;
            const int blkj = (ljc1+1)/2 * lcc1;

            int smem_size0 = lij1 * lic1 * ljc1; // rad_all
            int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
            int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
            int smem_size3 = li1*lic1*nfi; // angi
            int smem_size4 = lj1*ljc1*nfj; // angj
            int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;

            ECP_LAUNCH_GENERAL(type2_cart_ip1_general_kernel, dynamic_smem_size,
                               type2_cart_ip1_general,
                               gctr, li, lj, lc, ao_loc, nao,
                               tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        }}
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ipipv_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
#ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(ntasks);
    sycl::queue& stream = *sycl_get_queue();
#else
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
#endif

    if (lc < 0){
        const int lij1 = li+lj+3;
        const int lij3 = lij1*lij1*lij1;

        int smem_size = lij3 + lij1*lij1;

        ECP_LAUNCH_GENERAL(type1_cart_ipipv_kernel, smem_size, type1_cart_ipipv,
                           gctr, li, lj, ao_loc, nao,
                           tasks, ntasks, ecpbas, ecploc, atm, bas, env);
    } else {
        const int li1 = li+3;
        const int lj1 = lj+1;
        const int lij1 = li1+lj;
        const int nfi =  li1*(li1+1)/2;
        const int nfj = lj1*(lj1+1)/2;
        const int lic1 = li1+lc;
        const int ljc1 = lj1+lc;
        const int lcc1 = 2*lc+1;
        const int blki = (lic1+1)/2 * lcc1;
        const int blkj = (ljc1+1)/2 * lcc1;

        int smem_size0 = lij1 * lic1 * ljc1; // rad_all
        int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
        int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
        int smem_size3 = li1*lic1*nfi; // angi
        int smem_size4 = lj1*ljc1*nfj; // angj

        int NF1_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
        int NF0_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF1_MAX*NF0_MAX);

        ECP_LAUNCH_GENERAL(type2_cart_ipipv_kernel, dynamic_smem_size, type2_cart_ipipv,
                           gctr, li, lj, lc, ao_loc, nao,
                           tasks, ntasks, ecpbas, ecploc, atm, bas, env);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ipvip_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
#ifdef USE_SYCL
    sycl::range<1> threads(THREADS);
    sycl::range<1> blocks(ntasks);
    sycl::queue& stream = *sycl_get_queue();
#else
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
#endif

    if (lc < 0){
        const int lij1 = li+lj+3;
        const int lij3 = lij1*lij1*lij1;

        int smem_size = lij3 + lij1*lij1;

        ECP_LAUNCH_GENERAL(type1_cart_ipvip_kernel, smem_size, type1_cart_ipvip,
                           gctr, li, lj, ao_loc, nao,
                           tasks, ntasks, ecpbas, ecploc, atm, bas, env);
    } else {
        const int li1 = li+2;
        const int lj1 = lj+2;
        const int lij1 = li1+lj1-1;
        const int nfi = li1*(li1+1)/2;
        const int nfj = lj1*(lj1+1)/2;
        const int lic1 = li1+lc;
        const int ljc1 = lj1+lc;
        const int lcc1 = 2*lc+1;
        const int blki = (lic1+1)/2 * lcc1;
        const int blkj = (ljc1+1)/2 * lcc1;

        int smem_size0 = lij1 * lic1 * ljc1; // rad_all
        int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
        int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
        int smem_size3 = li1*lic1*nfi; // angi
        int smem_size4 = lj1*ljc1*nfj; // angj

        int NF1_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
        int NF0_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF0_MAX*NF1_MAX);

        ECP_LAUNCH_GENERAL(type2_cart_ipvip_kernel, dynamic_smem_size, type2_cart_ipvip,
                           gctr, li, lj, lc, ao_loc, nao,
                           tasks, ntasks, ecpbas, ecploc, atm, bas, env);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}

#undef ECP_ARGS
#undef ECP_LAUNCH1
#undef ECP_LAUNCH2
#undef ECP_LAUNCH_GENERAL
