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
#include "ecp_type1.cu"
#include "ecp_type2.cu"

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
        switch (task_type)
        {
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
            
            int smem_size = 0; 
            smem_size += (li+lj+1) * lic1 * ljc1; // rad_all
            smem_size += li1*(li1+1)*(li1+2)/6 * blki; // omegai
            smem_size += lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
            smem_size += li1*nfi*lic1; // angi
            smem_size += lj1*nfj*ljc1; // angj

            type2_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr,
                li, lj, lc,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }
        }
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
            type1_cart_general<0,0><<<blocks, threads, smem_size*sizeof(double)>>>(
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

int ECP_ip1_cart(double *gctr, 
            const int *ao_loc, const int nao, 
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc, 
            const int *atm, const int *bas, const double *env, 
            const int li, const int lj, const int lc){
    // one task per thread block
    dim3 threads(THREADS);
    dim3 blocks(ntasks);

    if (lc < 0){
        int task_type = li * 100 + lj * 10 + lc;
        type1_cart_general<1,0><<<blocks, threads>>>(gctr, li+1, lj, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        type1_cart_general<0,0><<<blocks, threads>>>(gctr, li-1, lj, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env);
        }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}

