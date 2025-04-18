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

extern "C" {
int ECP_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    if (lc >= 0){
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        case 0:  type2_cart<0,0,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 1:  type2_cart<0,0,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 2:  type2_cart<0,0,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 3:  type2_cart<0,0,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 10:  type2_cart<0,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 11:  type2_cart<0,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 12:  type2_cart<0,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 110: type2_cart<1,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 111: type2_cart<1,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 112: type2_cart<1,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 20:  type2_cart<0,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 21:  type2_cart<0,2,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 30:  type2_cart<0,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 120: type2_cart<1,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

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

            type2_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr,
                li, lj, lc,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    } else {
        int task_type = li * 10 + lj;
        switch (task_type)
        {
        case 0:  type1_cart<0,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 1:  type1_cart<0,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 11: type1_cart<1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 2:  type1_cart<0,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 3:  type1_cart<0,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 12: type1_cart<1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 4:  type1_cart<0,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 13: type1_cart<1,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 22: type1_cart<2,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        default: {
            const int lij1 = li+lj+1;
            const int lij3 = lij1*lij1*lij1;

            int smem_size = 0;
            smem_size += lij3;      // rad_ang
            smem_size += lij1*lij1; // rad_all
            type1_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr, li, lj,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
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
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    if (lc < 0){
        int task_type = li * 10 + lj;
        switch (task_type) {
        case 0:  type1_cart_ip1<0,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 1:  type1_cart_ip1<0,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 11: type1_cart_ip1<1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 2:  type1_cart_ip1<0,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 3:  type1_cart_ip1<0,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 12: type1_cart_ip1<1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 4:  type1_cart_ip1<0,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 13: type1_cart_ip1<1,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 22: type1_cart_ip1<2,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        default: {
            const int lij1 = li+lj+2;
            const int lij3 = lij1*lij1*lij1;

            int smem_size = 0;
            smem_size += lij3;      // rad_ang
            smem_size += lij1*lij1; // rad_all
            type1_cart_ip1_general<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr, li, lj,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    } else {
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        case 0:  type2_cart_ip1<0,0,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 1:  type2_cart_ip1<0,0,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 2:  type2_cart_ip1<0,0,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 3:  type2_cart_ip1<0,0,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 10:  type2_cart_ip1<0,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 11:  type2_cart_ip1<0,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 12:  type2_cart_ip1<0,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 110: type2_cart_ip1<1,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 111: type2_cart_ip1<1,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 112: type2_cart_ip1<1,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 20:  type2_cart_ip1<0,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 21:  type2_cart_ip1<0,2,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 30:  type2_cart_ip1<0,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 120: type2_cart_ip1<1,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

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
            cudaError_t err = cudaFuncSetAttribute(
                type2_cart_ip1_general,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_smem_size*sizeof(double));

            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s: %s\n", __func__, cudaGetErrorString(err));
                return 1;
            }

            type2_cart_ip1_general<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
                gctr, li, lj, lc,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
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
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    
    if (lc < 0){
        const int lij1 = li+lj+3; //
        const int lij3 = lij1*lij1*lij1;

        int smem_size = 0;
        smem_size += lij3;      // rad_ang
        smem_size += lij1*lij1; // rad_all
        type1_cart_ipipv<<<blocks, threads, smem_size*sizeof(double)>>>(
            gctr, li, lj,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);

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

        //int NF2_MAX = (AO_LMAX+3)*(AO_LMAX+4)/2;
        int NF1_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
        int NF0_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
        //int static_smem_size = NF2_MAX*NF0_MAX;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF1_MAX*NF0_MAX);
        //int total_smem_size = static_smem_size + dynamic_smem_size;

        cudaError_t err = cudaFuncSetAttribute(type2_cart_ipipv,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         dynamic_smem_size*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s: %s\n", __func__, cudaGetErrorString(err));
            return 1;
        }

        type2_cart_ipipv<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
            gctr, li, lj, lc,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);

    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
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
    dim3 threads(THREADS);
    dim3 blocks(ntasks);

    if (lc < 0){
        const int lij1 = li+lj+3; //
        const int lij3 = lij1*lij1*lij1;

        int smem_size = 0;
        smem_size += lij3;      // rad_ang
        smem_size += lij1*lij1; // rad_all
        type1_cart_ipvip<<<blocks, threads, smem_size*sizeof(double)>>>(
            gctr, li, lj,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);
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
        //int static_smem_size = NF1_MAX*NF1_MAX;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF0_MAX*NF1_MAX);

        //int total_smem_size = static_smem_size + dynamic_smem_size;
        cudaError_t err = cudaFuncSetAttribute(
            type2_cart_ipvip,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_smem_size*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s: %s\n", __func__, cudaGetErrorString(err));
            return 1;
        }

        type2_cart_ipvip<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
            gctr, li, lj, lc,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}
