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
            int li, int lj, int lk){
    // one task per thread block
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    if (lk >= 0){
        int task_type = li * 100 + lj * 10 + lk;
        switch (task_type)
        {
        case 0:  type2_cart<0,0,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 1:  type2_cart<0,0,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 2:  type2_cart<0,0,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 3:  type2_cart<0,0,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 4:  type2_cart<0,0,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 10:  type2_cart<0,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 11:  type2_cart<0,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 12:  type2_cart<0,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 13:  type2_cart<0,1,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 14:  type2_cart<0,1,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 110: type2_cart<1,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 111: type2_cart<1,1,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 112: type2_cart<1,1,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 113: type2_cart<1,1,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 114: type2_cart<1,1,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        
        case 20:  type2_cart<0,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 21:  type2_cart<0,2,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 22:  type2_cart<0,2,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 23:  type2_cart<0,2,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 24:  type2_cart<0,2,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 30:  type2_cart<0,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 31:  type2_cart<0,3,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 32:  type2_cart<0,3,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 33:  type2_cart<0,3,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 34:  type2_cart<0,3,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        
        case 120: type2_cart<1,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 121: type2_cart<1,2,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 122: type2_cart<1,2,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 123: type2_cart<1,2,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 124: type2_cart<1,2,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 40:  type2_cart<0,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 41:  type2_cart<0,4,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 42:  type2_cart<0,4,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 43:  type2_cart<0,4,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 44:  type2_cart<0,4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
           
        case 130: type2_cart<1,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 131: type2_cart<1,3,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 132: type2_cart<1,3,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 133: type2_cart<1,3,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 134: type2_cart<1,3,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        
        case 220: type2_cart<2,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 221: type2_cart<2,2,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 222: type2_cart<2,2,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 223: type2_cart<2,2,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 224: type2_cart<2,2,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 140: type2_cart<1,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 141: type2_cart<1,4,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 142: type2_cart<1,4,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 143: type2_cart<1,4,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 144: type2_cart<1,4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        
        case 230: type2_cart<2,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 231: type2_cart<2,3,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 232: type2_cart<2,3,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 233: type2_cart<2,3,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 234: type2_cart<2,3,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 240: type2_cart<2,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 241: type2_cart<2,4,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 242: type2_cart<2,4,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 243: type2_cart<2,4,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 244: type2_cart<2,4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        
        case 330: type2_cart<3,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 331: type2_cart<3,3,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 332: type2_cart<3,3,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 333: type2_cart<3,3,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 334: type2_cart<3,3,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 340: type2_cart<3,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 341: type2_cart<3,4,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 342: type2_cart<3,4,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 343: type2_cart<3,4,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 344: type2_cart<3,4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 440: type2_cart<4,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 441: type2_cart<4,4,1><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 442: type2_cart<4,4,2><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 443: type2_cart<4,4,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 444: type2_cart<4,4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        default: fprintf(stderr, "(%d,%d,%d) is not supported in ECP.\n", li, lj, lk); break;
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

        case 14: type1_cart<1,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 23: type1_cart<2,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 24: type1_cart<2,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 33: type1_cart<3,3><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 34: type1_cart<3,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 44: type1_cart<4,4><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        default: fprintf(stderr, "(%d,%d) is not supported in ECP.\n", li, lj); break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}
