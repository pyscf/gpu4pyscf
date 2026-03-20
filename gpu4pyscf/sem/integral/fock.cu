/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_70 fock.cu -o libfock.so

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__
void build_jk_2c2e_kernel(
    const double* w_1d,
    const double* P,
    double* J,
    double* K,
    const int* pair_i,
    const int* pair_j,
    const int* kr_offsets,
    const int* aoslice,
    const int* natorb,
    const int* loc_row,
    const int* loc_col,
    int npairs,
    int nao) 
{
    // Each block processes exactly one pair of interacting atoms (Atom A and Atom B)
    int p = blockIdx.x;
    if (p >= npairs) return;
    
    int A = pair_i[p];
    int B = pair_j[p];
    
    // aoslice[A, 0] and aoslice[B, 0]
    int offset_A = aoslice[A * 2]; 
    int offset_B = aoslice[B * 2];
    
    int nA = natorb[A];
    int nB = natorb[B];
    
    // Calculate the length of the lower triangular combinations for both atoms
    int limij = nA * (nA + 1) / 2;
    int limkl = nB * (nB + 1) / 2;
    int total_elements = limij * limkl;
    
    int start = kr_offsets[p];
    
    // up to 2025 integrals within the block
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        double val = w_1d[start + idx];
        
        // skip strictly zero elements to save massive atomicAdd overhead
        if (fabs(val) < 1e-14) continue;
        
        int IJ = idx / limkl;
        int KL = idx % limkl;
        
        int mu_loc = loc_row[IJ];
        int nu_loc = loc_col[IJ];
        int lam_loc = loc_row[KL];
        int sig_loc = loc_col[KL];
        
        int mu = offset_A + mu_loc;
        int nu = offset_A + nu_loc;
        int lam = offset_B + lam_loc;
        int sig = offset_B + sig_loc;
        
        //  J_mu,nu += V * P_lam,sig  AND  J_lam,sig += V * P_mu,nu
        double P_ls_sum = P[lam * nao + sig];
        if (lam != sig) P_ls_sum += P[sig * nao + lam];
        
        atomicAdd(&J[mu * nao + nu], val * P_ls_sum);
        if (mu != nu) {
            atomicAdd(&J[nu * nao + mu], val * P_ls_sum);
        }
        
        double P_mn_sum = P[mu * nao + nu];
        if (mu != nu) P_mn_sum += P[nu * nao + mu];
        
        atomicAdd(&J[lam * nao + sig], val * P_mn_sum);
        if (lam != sig) {
            atomicAdd(&J[sig * nao + lam], val * P_mn_sum);
        }
        
        //  K_mu,lam += V * P_nu,sig (and all 8-fold permutations)
        atomicAdd(&K[mu * nao + lam], val * P[nu * nao + sig]);
        atomicAdd(&K[lam * nao + mu], val * P[sig * nao + nu]);
        
        if (mu != nu) {
            atomicAdd(&K[nu * nao + lam], val * P[mu * nao + sig]);
            atomicAdd(&K[lam * nao + nu], val * P[sig * nao + mu]);
        }
        
        if (lam != sig) {
            atomicAdd(&K[mu * nao + sig], val * P[nu * nao + lam]);
            atomicAdd(&K[sig * nao + mu], val * P[lam * nao + nu]);
        }
        
        if (mu != nu && lam != sig) {
            atomicAdd(&K[nu * nao + sig], val * P[mu * nao + lam]);
            atomicAdd(&K[sig * nao + nu], val * P[lam * nao + mu]);
        }
    }
}

extern "C" {
    void launch_build_jk_2c2e(
        const double* w_1d,
        const double* P,
        double* J,
        double* K,
        const int* pair_i,
        const int* pair_j,
        const int* kr_offsets,
        const int* aoslice,
        const int* natorb,
        const int* loc_row,
        const int* loc_col,
        int npairs,
        int nao)
    {
        if (npairs <= 0) return;

        // 1 block per atom pair (npairs blocks in total)
        // 256 threads per block to cooperatively process the elements (up to 2025)
        int blocks = npairs;
        int threads = 256;
        
        build_jk_2c2e_kernel<<<blocks, threads>>>(
            w_1d, P, J, K, 
            pair_i, pair_j, kr_offsets, 
            aoslice, natorb, loc_row, loc_col, 
            npairs, nao
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
}