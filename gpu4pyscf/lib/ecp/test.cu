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
#include "ecp.cu"
/*
template <int LMAX> __global__
void _ang_nuc_part(double *omega, double *x, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n){
        return;
    }
    int offset = idx * (2*LMAX+1);
    ang_nuc_part<LMAX>(omega+offset, x[3*idx], x[3*idx+1], x[3*idx+2]);
}

template <int LMAX> __global__
void _type1_rad_part(double *rad_all, double k, double aij, double *ur, int n){
    int idx = blockIdx.x;
    if (idx >= n){
        return;
    }
    type1_rad_part<LMAX>(rad_all, k, aij, ur[threadIdx.x]);
}

template <int LMAX> __global__
void _type1_rad_ang(double *rad_ang, double *r, double *rad_all, double fac, int n){
    int idx = blockIdx.x;
    if (idx >= n){
        return;
    }
    constexpr int offset = (LMAX+1)*(LMAX+1)*(LMAX+1);
    type1_rad_ang<LMAX>(rad_ang+offset*idx, r+3*idx, rad_all, fac);
}

__global__
void _rad_part(int ish, int *ecpbas, double *env, double *rs, double *ws, double *ur, int nr, int n){
    int idx = blockIdx.x;
    if (idx >= n){
        return;
    }
    ur[threadIdx.x] = rad_part(ish, ecpbas, env);
}

template <int LI, int LC>
__global__
void _type2_facs_ang(double *facs, double *r){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1){
        return;
    }
    type2_facs_ang<LI, LC>(facs, r);
}

template <int LI, int LC>
__global__
void _type2_facs_rad(double *facs, int np, double rca, double *ci, double *ai){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1){
        return;
    }
    type2_facs_rad<LI, LC>(facs, np, rca, ci, ai);
}
*/

extern "C" {
/*
int ECPsph_ine(double *out, int order, double *zs, int n)
{
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    _ine_kernel<<<blocks, threads>>>(out, order, zs, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int ECPang_nuc_part(double *omega, double *x, int n, const int l){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    switch (l){
    case 0: _ang_nuc_part<0><<<blocks, threads>>>(omega, x, n); break;
    case 1: _ang_nuc_part<1><<<blocks, threads>>>(omega, x, n); break;
    case 2: _ang_nuc_part<2><<<blocks, threads>>>(omega, x, n); break;
    case 3: _ang_nuc_part<3><<<blocks, threads>>>(omega, x, n); break;
    case 4: _ang_nuc_part<4><<<blocks, threads>>>(omega, x, n); break;
    case 5: _ang_nuc_part<5><<<blocks, threads>>>(omega, x, n); break;
    case 6: _ang_nuc_part<6><<<blocks, threads>>>(omega, x, n); break;
    case 7: _ang_nuc_part<7><<<blocks, threads>>>(omega, x, n); break;
    case 8: _ang_nuc_part<8><<<blocks, threads>>>(omega, x, n); break;
    default:
        fprintf(stderr, "l > 8 is not supported\n");
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int ECPrad_part(int ish, int *ecpbas, double *env, double *rs, double *ws, double *ur, int nr){
    int n = 1;
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    _rad_part<<<blocks, threads>>>(ish, ecpbas, env, rs, ws, ur, nr, n);
    return 0;
}

int ECPtype1_rad_part(double *rad_all, int l, double k, double aij, double *ur, int n){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    switch (l){
    case 0: _type1_rad_part<0><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 1: _type1_rad_part<1><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 2: _type1_rad_part<2><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 3: _type1_rad_part<3><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 4: _type1_rad_part<4><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 5: _type1_rad_part<5><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 6: _type1_rad_part<6><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 7: _type1_rad_part<7><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;
    case 8: _type1_rad_part<8><<<blocks, threads>>>(rad_all, k, aij, ur, n); break;

    default:
        fprintf(stderr, "l > 8 is not supported\n");
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int ECPtype1_rad_ang(double *rad_ang, int l, int n, double *r, double fac, double *rad_all){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    switch (l){
    case 0: _type1_rad_ang<0><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 1: _type1_rad_ang<1><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 2: _type1_rad_ang<2><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 3: _type1_rad_ang<3><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 4: _type1_rad_ang<4><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 5: _type1_rad_ang<5><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 6: _type1_rad_ang<6><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 7: _type1_rad_ang<7><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    case 8: _type1_rad_ang<8><<<blocks, threads>>>(rad_ang, r, rad_all, fac, n); break;
    default:
        fprintf(stderr, "l > 8 is not supported\n");
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
*/
/*
int ECPtype1_cart(double *gctr, const int *tasks, const int ntasks,
                    const int *ecpbas, const int *ecploc, const int *atm,
                    const int *bas, const double *env, int li, int lj){
    int ntile = (ntasks + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);

    int task_type = li * 10 + lj;
    switch (task_type)
    {
    case 0:  type1_cart<0,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 1:  type1_cart<0,1><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 11: type1_cart<1,1><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 2:  type1_cart<0,2><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 3:  type1_cart<0,3><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 12: type1_cart<1,2><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 4:  type1_cart<0,4><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 13: type1_cart<1,3><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 22: type1_cart<2,2><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 5:  type1_cart<0,5><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 14: type1_cart<1,4><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 23: type1_cart<2,3><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 24: type1_cart<2,4><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 33: type1_cart<3,3><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 34: type1_cart<3,4><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 44: type1_cart<4,4><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    default: fprintf(stderr, "(%d,%d) is not supported in ECP.\n", li, lj); break;
    }

    return 0;
    }
*/
    /*
int ECPtype2_facs_rad(double *facs, int l, int lc, int np, double rca, double *ci, double *ai){
    int n = 1;
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    int task_type = l * 10 + lc;
    switch (task_type) {
    case 0: _type2_facs_rad<0,0><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 1: _type2_facs_rad<1,1><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 2: _type2_facs_rad<2,2><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 3: _type2_facs_rad<3,3><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 4: _type2_facs_rad<4,4><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 5: _type2_facs_rad<5,0><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 6: _type2_facs_rad<6,1><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 7: _type2_facs_rad<7,2><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    case 8: _type2_facs_rad<8,3><<<blocks, threads>>>(facs, np, rca, ci, ai); break;
    default:
        fprintf(stderr, "l > 8 is not supported\n");
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
    }
*/
/*
int ECPtype2_facs_ang(double *facs, double *r, int l, int lc){
    int n = 1;
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    int task_type = l * 10 + lc;
    switch (task_type) {
    case 0:  _type2_facs_ang<0,0><<<blocks, threads>>>(facs, r); break;

    case 1:  _type2_facs_ang<0,1><<<blocks, threads>>>(facs, r); break;
    case 10: _type2_facs_ang<1,0><<<blocks, threads>>>(facs, r); break;

    case 2:  _type2_facs_ang<0,2><<<blocks, threads>>>(facs, r); break;
    case 11: _type2_facs_ang<1,1><<<blocks, threads>>>(facs, r); break;
    case 20: _type2_facs_ang<2,0><<<blocks, threads>>>(facs, r); break;

    case 3:  _type2_facs_ang<0,3><<<blocks, threads>>>(facs, r); break;
    case 12: _type2_facs_ang<1,2><<<blocks, threads>>>(facs, r); break;
    case 21: _type2_facs_ang<2,1><<<blocks, threads>>>(facs, r); break;
    case 30: _type2_facs_ang<3,0><<<blocks, threads>>>(facs, r); break;

    case 4:  _type2_facs_ang<0,4><<<blocks, threads>>>(facs, r); break;
    case 13: _type2_facs_ang<1,3><<<blocks, threads>>>(facs, r); break;
    case 22: _type2_facs_ang<2,2><<<blocks, threads>>>(facs, r); break;
    case 31: _type2_facs_ang<3,1><<<blocks, threads>>>(facs, r); break;
    case 40: _type2_facs_ang<4,0><<<blocks, threads>>>(facs, r); break;

    case 14: _type2_facs_ang<1,4><<<blocks, threads>>>(facs, r); break;
    case 23: _type2_facs_ang<2,3><<<blocks, threads>>>(facs, r); break;
    case 32: _type2_facs_ang<3,2><<<blocks, threads>>>(facs, r); break;
    case 41: _type2_facs_ang<4,1><<<blocks, threads>>>(facs, r); break;

    case 24: _type2_facs_ang<2,4><<<blocks, threads>>>(facs, r); break;
    case 33: _type2_facs_ang<3,3><<<blocks, threads>>>(facs, r); break;
    case 42: _type2_facs_ang<4,2><<<blocks, threads>>>(facs, r); break;

    case 34: _type2_facs_ang<3,4><<<blocks, threads>>>(facs, r); break;
    case 43: _type2_facs_ang<4,3><<<blocks, threads>>>(facs, r); break;

    case 44: _type2_facs_ang<4,4><<<blocks, threads>>>(facs, r); break;

    default:
        fprintf(stderr, "l > 8 is not supported\n");
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
    }
*/
/*
int ECPtype2_cart(double *gctr, const int *tasks, const int ntasks,
        const int *ecpbas, const int *ecploc, const int *atm,
        const int *bas, const double *env, int li, int lj){
    int ntile = (ntasks + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);

    int task_type = li * 10 + lj;
    switch (task_type)
    {
    case 0:  type2_cart<0,0,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 1:  type2_cart<0,1,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 11: type2_cart<1,1,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 2:  type2_cart<0,2,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 3:  type2_cart<0,3,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 12: type2_cart<1,2,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 4:  type2_cart<0,4,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 13: type2_cart<1,3,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 22: type2_cart<2,2,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 5:  type2_cart<0,5,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 14: type2_cart<1,4,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 23: type2_cart<2,3,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 24: type2_cart<2,4,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    case 33: type2_cart<3,3,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 34: type2_cart<3,4,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

    case 44: type2_cart<4,4,0><<<blocks, threads>>>(gctr, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
    default: fprintf(stderr, "(%d,%d) is not supported in ECP.\n", li, lj); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
*/
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

        case 10:  type2_cart<0,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 110: type2_cart<1,1,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 20:  type2_cart<0,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 30:  type2_cart<0,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 120: type2_cart<1,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 40:  type2_cart<0,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 130: type2_cart<1,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 220: type2_cart<2,2,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 140: type2_cart<1,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 230: type2_cart<2,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 240: type2_cart<2,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        case 330: type2_cart<3,3,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 340: type2_cart<3,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;

        case 440: type2_cart<4,4,0><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); break;
        default: fprintf(stderr, "(%d,%d) is not supported in ECP.\n", li, lj); break;
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
