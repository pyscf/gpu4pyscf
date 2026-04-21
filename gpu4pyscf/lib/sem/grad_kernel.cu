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
#include <math.h>
#include <cuda_runtime.h>

__global__ void unpack_w_kernel(
    const double* __restrict__ w_1d, 
    const int* __restrict__ pair_i, 
    const int* __restrict__ pair_j, 
    const int* __restrict__ natorb, 
    const int* __restrict__ kr_offsets,
    const int* __restrict__ local_row_idx, 
    const int* __restrict__ local_col_idx,
    double* __restrict__ eri_4d_AB, 
    int n_pairs
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    
    int A = pair_i[p];
    int B = pair_j[p];
    int n_i = natorb[A];
    int n_j = natorb[B];
    int limij = n_i * (n_i + 1) / 2;
    int limkl = n_j * (n_j + 1) / 2;
    
    int start = kr_offsets[p];
    // 9 * 9 * 9 * 9 = 6561
    double* out = eri_4d_AB + p * 6561;
    
    for (int IJ = 0; IJ < limij; IJ++) {
        int mu_loc = local_row_idx[IJ];
        int nu_loc = local_col_idx[IJ];
        for (int KL = 0; KL < limkl; KL++) {
            int lam_loc = local_row_idx[KL];
            int sig_loc = local_col_idx[KL];
            
            double val = w_1d[start + IJ * limkl + KL];
            if (val == 0.0) continue;
            
            out[mu_loc * 729 + nu_loc * 81 + lam_loc * 9 + sig_loc] = val;
            out[nu_loc * 729 + mu_loc * 81 + lam_loc * 9 + sig_loc] = val;
            out[mu_loc * 729 + nu_loc * 81 + sig_loc * 9 + lam_loc] = val;
            out[nu_loc * 729 + mu_loc * 81 + sig_loc * 9 + lam_loc] = val;
        }
    }
}

extern "C" {
int launch_unpack_w_kernel_c(
    const double* w_1d, 
    const int* pair_i, 
    const int* pair_j, 
    const int* natorb, 
    const int* kr_offsets,
    const int* local_row_idx, 
    const int* local_col_idx,
    double* eri_4d_AB, 
    int n_pairs
) {
    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    
    unpack_w_kernel<<<blocks, threads>>>(
        w_1d, pair_i, pair_j, natorb, kr_offsets, 
        local_row_idx, local_col_idx, eri_4d_AB, n_pairs
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}