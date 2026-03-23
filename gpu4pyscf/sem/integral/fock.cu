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
    // Each block processes one pair of interacting atoms (Atom A and Atom B)
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

    // Allocate shared memory. 
    // In PM6, the maximum number of orbitals per atom is 9 (s, p, d).
    __shared__ double s_PAA[81];
    __shared__ double s_PBB[81];
    __shared__ double s_PAB[81];
    __shared__ double s_PBA[81];
    
    __shared__ double s_JAA[81];
    __shared__ double s_JBB[81];
    __shared__ double s_KAB[81];
    __shared__ double s_KBA[81];
    
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Initialize shared memory to zero
    for (int i = tid; i < 81; i += bdim) {
        s_JAA[i] = 0.0; s_JBB[i] = 0.0; s_KAB[i] = 0.0; s_KBA[i] = 0.0;
        s_PAA[i] = 0.0; s_PBB[i] = 0.0; s_PAB[i] = 0.0; s_PBA[i] = 0.0;
    }
    __syncthreads();

    // Cooperatively load global density matrix P into shared memory
    for (int i = tid; i < nA * nA; i += bdim) {
        int r = i / nA; int c = i % nA;
        s_PAA[i] = P[(offset_A + r) * nao + (offset_A + c)];
    }
    for (int i = tid; i < nB * nB; i += bdim) {
        int r = i / nB; int c = i % nB;
        s_PBB[i] = P[(offset_B + r) * nao + (offset_B + c)];
    }
    for (int i = tid; i < nA * nB; i += bdim) {
        int r = i / nB; int c = i % nB; // r in A, c in B
        s_PAB[i] = P[(offset_A + r) * nao + (offset_B + c)];
    }
    for (int i = tid; i < nB * nA; i += bdim) {
        int r = i / nA; int c = i % nA; // r in B, c in A
        s_PBA[i] = P[(offset_B + r) * nao + (offset_A + c)];
    }
    __syncthreads();
    
    for (int idx = tid; idx < total_elements; idx += bdim) {
        double val = w_1d[start + idx];
        
        if (fabs(val) < 1e-14) continue;
        
        int IJ = idx / limkl;
        int KL = idx % limkl;
        
        int mu = loc_row[IJ];
        int nu = loc_col[IJ];
        int lam = loc_row[KL];
        int sig = loc_col[KL];
        
        // J calculations
        double P_ls_sum = s_PBB[lam * nB + sig];
        if (lam != sig) P_ls_sum += s_PBB[sig * nB + lam];
        
        atomicAdd(&s_JAA[mu * nA + nu], val * P_ls_sum);
        if (mu != nu) atomicAdd(&s_JAA[nu * nA + mu], val * P_ls_sum);
        
        double P_mn_sum = s_PAA[mu * nA + nu];
        if (mu != nu) P_mn_sum += s_PAA[nu * nA + mu];
        
        atomicAdd(&s_JBB[lam * nB + sig], val * P_mn_sum);
        if (lam != sig) atomicAdd(&s_JBB[sig * nB + lam], val * P_mn_sum);
        
        // K calculations
        atomicAdd(&s_KAB[mu * nB + lam], val * s_PAB[nu * nB + sig]);
        atomicAdd(&s_KBA[lam * nA + mu], val * s_PBA[sig * nA + nu]);
        
        if (mu != nu) {
            atomicAdd(&s_KAB[nu * nB + lam], val * s_PAB[mu * nB + sig]);
            atomicAdd(&s_KBA[lam * nA + nu], val * s_PBA[sig * nA + mu]);
        }
        
        if (lam != sig) {
            atomicAdd(&s_KAB[mu * nB + sig], val * s_PAB[nu * nB + lam]);
            atomicAdd(&s_KBA[sig * nA + mu], val * s_PBA[lam * nA + nu]);
        }
        
        if (mu != nu && lam != sig) {
            atomicAdd(&s_KAB[nu * nB + sig], val * s_PAB[mu * nB + lam]);
            atomicAdd(&s_KBA[sig * nA + nu], val * s_PBA[lam * nA + mu]);
        }
    }
    __syncthreads();

    // After the loop, write the accumulated Shared Memory results back to Global Memory J/K
    for (int i = tid; i < nA * nA; i += bdim) {
        if (fabs(s_JAA[i]) > 1e-15) {
            int r = i / nA; int c = i % nA;
            atomicAdd(&J[(offset_A + r) * nao + (offset_A + c)], s_JAA[i]);
        }
    }
    for (int i = tid; i < nB * nB; i += bdim) {
        if (fabs(s_JBB[i]) > 1e-15) {
            int r = i / nB; int c = i % nB;
            atomicAdd(&J[(offset_B + r) * nao + (offset_B + c)], s_JBB[i]);
        }
    }
    for (int i = tid; i < nA * nB; i += bdim) {
        if (fabs(s_KAB[i]) > 1e-15) {
            int r = i / nB; int c = i % nB;
            atomicAdd(&K[(offset_A + r) * nao + (offset_B + c)], s_KAB[i]);
        }
    }
    for (int i = tid; i < nB * nA; i += bdim) {
        if (fabs(s_KBA[i]) > 1e-15) {
            int r = i / nA; int c = i % nA;
            atomicAdd(&K[(offset_B + r) * nao + (offset_A + c)], s_KBA[i]);
        }
    }
}


__device__ __forceinline__ void add_sJK(int m, int n, int l, int s, double val, 
                                        const double* s_P, double* s_J, double* s_K) {
    atomicAdd(&s_J[m * 9 + n], val * s_P[l * 9 + s]);
    atomicAdd(&s_K[m * 9 + l], val * s_P[n * 9 + s]);
}


// This resolves the 8-fold symmetry dynamically and prevents double-counting.
__device__ void apply_eri_1c2e(int mu, int nu, int lam, int sig, double val, 
                               const double* s_P, double* s_J, double* s_K) 
{
    if (fabs(val) < 1e-14) return;
    
    // Intra-pair permutations for (mu, nu | lam, sig)
    add_sJK(mu, nu, lam, sig, val, s_P, s_J, s_K);
    if (mu != nu) add_sJK(nu, mu, lam, sig, val, s_P, s_J, s_K);
    
    if (lam != sig) {
        add_sJK(mu, nu, sig, lam, val, s_P, s_J, s_K);
        if (mu != nu) add_sJK(nu, mu, sig, lam, val, s_P, s_J, s_K);
    }
    
    // If the two orbital pairs are equivalent (e.g., 0 0 | 0 0 or 0 1 | 1 0), 
    // skip cross-pair combinations to avoid duplicate additions.
    bool distinct_pairs = !((mu == lam && nu == sig) || (mu == sig && nu == lam));
    
    if (distinct_pairs) {
        // Inter-pair permutations for (lam, sig | mu, nu)
        add_sJK(lam, sig, mu, nu, val, s_P, s_J, s_K);
        if (lam != sig) add_sJK(sig, lam, mu, nu, val, s_P, s_J, s_K);
        
        if (mu != nu) {
            add_sJK(lam, sig, nu, mu, val, s_P, s_J, s_K);
            if (lam != sig) add_sJK(sig, lam, nu, mu, val, s_P, s_J, s_K);
        }
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