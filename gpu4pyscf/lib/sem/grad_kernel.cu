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


__global__ void calc_pair_e2e_kernel(
    const double* __restrict__ w_1d,
    const double* __restrict__ P_AA,   // (n_pairs, 9, 9)
    const double* __restrict__ P_BB,   // (n_pairs, 9, 9)
    const double* __restrict__ P_AB, // (n_pairs, 9, 9)
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ natorb,
    const int* __restrict__ kr_offsets,
    double* __restrict__ E_2e_out,     // (n_pairs,)
    int n_pairs
) {
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int p = item.get_global_id(0);
#else
    int p = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (p >= n_pairs) return;
    
    int A = pair_i[p];
    int B = pair_j[p];
    int n_i = natorb[A];
    int n_j = natorb[B];
    int start = kr_offsets[p];
    int limkl = n_j * (n_j + 1) / 2;
    
    const double* paa = P_AA + p * 81;
    const double* pbb = P_BB + p * 81;
    const double* pab = P_AB + p * 81;
    
    double e_j = 0.0;
    double e_k = 0.0;
    
    // Direct reduction over local mu, nu, lam, sig
    for (int mu = 0; mu < n_i; mu++) {
        for (int nu = 0; nu < n_i; nu++) {
            int mu_max = mu > nu ? mu : nu;
            int mu_min = mu > nu ? nu : mu;
            int IJ = mu_max * (mu_max + 1) / 2 + mu_min;
            
            for (int lam = 0; lam < n_j; lam++) {
                for (int sig = 0; sig < n_j; sig++) {
                    int lam_max = lam > sig ? lam : sig;
                    int lam_min = lam > sig ? sig : lam;
                    int KL = lam_max * (lam_max + 1) / 2 + lam_min;
                    
                    double w_val = w_1d[start + IJ * limkl + KL];
                    if (w_val == 0.0) continue;
                    
                    e_j += paa[mu * 9 + nu] * pbb[lam * 9 + sig] * w_val;
                    e_k += pab[mu * 9 + lam] * pab[nu * 9 + sig] * w_val;
                }
            }
        }
    }
    
    E_2e_out[p] = 1.0 * e_j - 0.5 * e_k;
}

extern "C" {
int launch_calc_pair_e2e_c(
    const double* w_1d,
    const double* P_AA,
    const double* P_BB,
    const double* P_AB,
    const int* pair_i,
    const int* pair_j,
    const int* natorb,
    const int* kr_offsets,
    double* E_2e_out,
    int n_pairs
) {
    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
#ifdef USE_SYCL
    sycl_get_queue()->parallel_for<class calc_pair_e2e_sycl>(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) [[intel::kernel_args_restrict]] {
        calc_pair_e2e_kernel(w_1d, P_AA, P_BB, P_AB,
                             pair_i, pair_j, natorb, kr_offsets, E_2e_out, n_pairs);
    });
#else
    calc_pair_e2e_kernel<<<blocks, threads>>>(
        w_1d, P_AA, P_BB, P_AB,
        pair_i, pair_j, natorb, kr_offsets, E_2e_out, n_pairs
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1;
#endif
    return 0;
}

}
