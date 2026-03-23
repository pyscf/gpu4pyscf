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


// TODO: THis function uses many atomicAdd, which can be slow.
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


// Resolves 8-fold symmetry dynamically and removes duplicate indices
// to completely prevent double-counting on diagonal blocks (e.g., mu=nu=lam=sig).
// ! Explicitly handle the 8-fold symmetry.
// TODO: THis function uses many atomicAdd, which can be slow.
__device__ void apply_eri_1c2e(int mu, int nu, int lam, int sig, double val, 
                               const double* s_P, double* s_J, double* s_K) 
{
    if (fabs(val) < 1e-14) return;
    
    int t[8][4];
    int num = 0;
    
    int c[8][4] = {
        {mu, nu, lam, sig},
        {nu, mu, lam, sig},
        {mu, nu, sig, lam},
        {nu, mu, sig, lam},
        {lam, sig, mu, nu},
        {lam, sig, nu, mu},
        {sig, lam, mu, nu},
        {sig, lam, nu, mu}
    };
    
    for(int i = 0; i < 8; ++i) {
        bool dup = false;
        for(int j = 0; j < num; ++j) {
            if(t[j][0] == c[i][0] && t[j][1] == c[i][1] && 
               t[j][2] == c[i][2] && t[j][3] == c[i][3]) {
                dup = true; 
                break;
            }
        }
        if(!dup) {
            t[num][0] = c[i][0]; t[num][1] = c[i][1];
            t[num][2] = c[i][2]; t[num][3] = c[i][3];
            num++;
        }
    }
    
    // J_mn += val * P_ls
    // K_ml += val * P_ns
    for(int i = 0; i < num; ++i) {
        int m = t[i][0], n = t[i][1], l = t[i][2], s = t[i][3];
        atomicAdd(&s_J[m * 9 + n], val * s_P[l * 9 + s]);
        atomicAdd(&s_K[m * 9 + l], val * s_P[n * 9 + s]);
    }
}

__global__
void build_jk_1c2e_kernel(
    const double* P,
    double* J,
    double* K,
    const double* gss,
    const double* gsp,
    const double* hsp,
    const double* gpp,
    const double* gp2,
    const double* repd,
    const int* intij,
    const int* intkl,
    const int* intrep,
    const int* aoslice,
    const int* natorb,
    const int* loc_row,
    const int* loc_col,
    int natm,
    int nao,
    int num_d_pairs) 
{
    // Grid handles 1 atom per block
    int A = blockIdx.x;
    if (A >= natm) return;
    
    int offset = aoslice[A * 2]; 
    int nao_A = natorb[A];
    
    __shared__ double s_P[81];
    __shared__ double s_J[81];
    __shared__ double s_K[81];
    
    for (int i = threadIdx.x; i < 81; i += blockDim.x) {
        s_J[i] = 0.0;
        s_K[i] = 0.0;
        int row = i / 9;
        int col = i % 9;
        if (row < nao_A && col < nao_A) {
            s_P[i] = P[(offset + row) * nao + (offset + col)];
        } else {
            s_P[i] = 0.0;
        }
    }
    __syncthreads();
    
    // Thread 0 handles the small number of s and p orbital integrals
    if (threadIdx.x == 0) {
        // s-orbital
        apply_eri_1c2e(0, 0, 0, 0, gss[A], s_P, s_J, s_K);
        
        // p-orbitals
        if (nao_A >= 4) {
            double v_gsp = gsp[A];
            double v_gpp = gpp[A];
            double v_gp2 = gp2[A];
            double v_hsp = hsp[A];
            double v_hpp = 0.5 * (v_gpp - v_gp2);
            
            for (int i = 1; i < 4; ++i) {
                apply_eri_1c2e(0, 0, i, i, v_gsp, s_P, s_J, s_K);
                apply_eri_1c2e(i, i, i, i, v_gpp, s_P, s_J, s_K);
                apply_eri_1c2e(0, i, 0, i, v_hsp, s_P, s_J, s_K);
                
                for (int j = 1; j < i; ++j) {
                    apply_eri_1c2e(i, i, j, j, v_gp2, s_P, s_J, s_K);
                    apply_eri_1c2e(i, j, i, j, v_hpp, s_P, s_J, s_K);
                }
            }
        }
    }
    
    // All threads cooperatively handle d-orbital combinations
    if (nao_A == 9 && num_d_pairs > 0) {
        for (int idx = threadIdx.x; idx < num_d_pairs; idx += blockDim.x) {
            int IJ = intij[idx];
            int KL = intkl[idx];
            int rp = intrep[idx];

            // The intij contains the symmetric part, in apply_eri_1c2e
            // we use the atomic add, thus half should be left.
            if (IJ < KL) continue;
            
            // repd has shape (52, natm)
            double val = repd[rp * natm + A];
            
            int mu = loc_row[IJ];
            int nu = loc_col[IJ];
            int lam = loc_row[KL];
            int sig = loc_col[KL];
            
            apply_eri_1c2e(mu, nu, lam, sig, val, s_P, s_J, s_K);
        }
    }
    
    __syncthreads();
    
    for (int i = threadIdx.x; i < 81; i += blockDim.x) {
        int row = i / 9;
        int col = i % 9;
        if (row < nao_A && col < nao_A) {
            if (fabs(s_J[i]) > 1e-15) {
                atomicAdd(&J[(offset + row) * nao + (offset + col)], s_J[i]);
            }
            if (fabs(s_K[i]) > 1e-15) {
                atomicAdd(&K[(offset + row) * nao + (offset + col)], s_K[i]);
            }
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

    void launch_build_jk_1c2e(
        const double* P,
        double* J,
        double* K,
        const double* gss,
        const double* gsp,
        const double* hsp,
        const double* gpp,
        const double* gp2,
        const double* repd,
        const int* intij,
        const int* intkl,
        const int* intrep,
        const int* aoslice,
        const int* natorb,
        const int* loc_row,
        const int* loc_col,
        int natm,
        int nao,
        int num_d_pairs)
    {
        if (natm <= 0) return;

        // 1 block per atom
        int blocks = natm;
        // 64 threads per block is sufficient since max d-orbital combinations is 243
        int threads = 64; 
        
        build_jk_1c2e_kernel<<<blocks, threads>>>(
            P, J, K, 
            gss, gsp, hsp, gpp, gp2, repd, 
            intij, intkl, intrep,
            aoslice, natorb, loc_row, loc_col,
            natm, nao, num_d_pairs
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel 1c2e failed: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
}