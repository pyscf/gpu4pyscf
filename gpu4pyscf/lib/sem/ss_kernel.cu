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

// nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC -arch=sm_70 ss_kernel.cu -o libss_kernel.so

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BINOM_DIM 13
#define IDX2(r, c) ((r) * (BINOM_DIM) + (c))


__global__ void afn_kernel(
    const int n_data,
    const double* __restrict__ p_vec,
    double* __restrict__ af_out // Shape: (n_data, 20)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int out_id = tid * 20;

    if (tid >= n_data) return;
    double p = p_vec[tid] + 1e-16;
    double inv_p = 1.0 / p;
    double term0 = inv_p * exp(-p);
    af_out[out_id] = term0;
    double val_prev = term0;

    for (int i = 1; i < 20; ++i) {
        double val_curr = (i * inv_p * val_prev) + term0;
        af_out[out_id + i] = val_curr;
        val_prev = val_curr;
    }
}


__global__ void bfn_kernel(
    const int n_data,
    const double* __restrict__ x,
    const double* __restrict__ taylor_coeffs, // Flattened (13 * 16) transposed taylor coeffs
    double* __restrict__ bf_out // Shape: (n_data, 13)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_data) return;

    double x_val = x[idx];
    double absx = fabs(x_val);
    double* out_ptr = &bf_out[idx * 13];

    if (absx <= 1.0e-6) {
        for (int i = 0; i < 13; ++i) {
            if (i % 2 == 0) {
                out_ptr[i] = 2.0 / (double)(i + 1);
            } else {
                out_ptr[i] = 0.0;
            }
        }
        return;
    }

    if (absx <= 3.0) {
        int norder_cut_off = 0;
        if (absx <= 0.5)      norder_cut_off = 7;
        else if (absx <= 1.0) norder_cut_off = 8;
        else if (absx <= 2.0) norder_cut_off = 13;
        else                  norder_cut_off = 16;

        double pow_minus_x[17];
        pow_minus_x[0] = 1.0;
        double neg_x = -x_val;
        
        for(int m = 1; m < norder_cut_off; ++m){
            pow_minus_x[m] = pow_minus_x[m-1] * neg_x;
        }

        for (int i = 0; i < 13; ++i) {
            double sum = 0.0;
            for (int m = 0; m < norder_cut_off; ++m) {
                sum += pow_minus_x[m] * taylor_coeffs[i * 16 + m];
            }
            out_ptr[i] = sum;
        }
        return;
    }

    double inv_x = 1.0 / x_val;
    double expx = exp(x_val);
    double expmx = 1.0 / expx; 
    
    double val_curr = (expx - expmx) * inv_x;
    out_ptr[0] = val_curr;

    for (int i = 1; i < 13; ++i) {
        double term;
        if (i % 2 == 1) {
            term = -expx - expmx;
        } else {
            term = expx - expmx;
        }
        double val_next = (i * val_curr + term) * inv_x;
        out_ptr[i] = val_next;
        val_curr = val_next;
    }
}


__global__ void rotation_transform_kernel(
    const int n_pairs,
    const double* __restrict__ S_local,  // Input: (N, 3, 3, 3)
    const double* __restrict__ C_tensor, // Input: (N, 3, 5, 5)
    double* __restrict__ di_out          // Output: (N, 9, 9)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;

    // ival[shell_idx][local_k_index]
    // Shell 0 (S): k=2 maps to 0
    // Shell 1 (P): k=1 maps to 2, k=2 maps to 3, k=3 maps to 1
    // Shell 2 (D): k=0..4 maps to 8..4
    const int ival[3][5] = {
        {0, 0, 0, 0, -1},       
        {-1, 2, 3, 1, -1},      
        {8, 7, 6, 5, 4}         
    };

    const double* s_ptr = S_local + idx * 27; 
    const double* c_ptr = C_tensor + idx * 75; 
    double* out_ptr = di_out + idx * 81;

    for (int i = 0; i < 3; ++i) { 
        int k_start = 2 - i;
        int k_end = 3 + i; 

        for (int j = 0; j < 3; ++j) { 
            int l_start = 2 - j;
            int l_end = 3 + j;

            double aa = (j == 1) ? -1.0 : 1.0;
            double bb = (j == 2) ? -1.0 : 1.0;

            double val_sigma = s_ptr[i * 9 + j * 3 + 0];
            double val_pi    = s_ptr[i * 9 + j * 3 + 1];
            double val_delta = s_ptr[i * 9 + j * 3 + 2];

            for (int k = k_start; k < k_end; ++k) {
                int idx_a = ival[i][k];
                if (idx_a < 0) continue;

                for (int l = l_start; l < l_end; ++l) {
                    int idx_b = ival[j][l];
                    if (idx_b < 0) continue;

                    double c3_a = c_ptr[i*25 + k*5 + 2]; 
                    double c4_a = c_ptr[i*25 + k*5 + 3]; 
                    double c2_a = c_ptr[i*25 + k*5 + 1]; 
                    double c5_a = c_ptr[i*25 + k*5 + 4]; 
                    double c1_a = c_ptr[i*25 + k*5 + 0]; 

                    double c3_b = c_ptr[j*25 + l*5 + 2];
                    double c4_b = c_ptr[j*25 + l*5 + 3];
                    double c2_b = c_ptr[j*25 + l*5 + 1];
                    double c5_b = c_ptr[j*25 + l*5 + 4];
                    double c1_b = c_ptr[j*25 + l*5 + 0];

                    double term = val_sigma * (c3_a * c3_b) * aa;

                    if (i > 0 && j > 0) {
                        term += val_pi * (c4_a * c4_b + c2_a * c2_b) * bb;
                        if (i > 1 && j > 1) {
                            term += val_delta * (c5_a * c5_b + c1_a * c1_b);
                        }
                    }

                    out_ptr[idx_a * 9 + idx_b] += term;
                }
            }
        }
    }
}


__global__ void ss_summation_kernel(
    const int n_pairs,
    const int* __restrict__ ia_in,
    const int* __restrict__ ib_in,
    const int* __restrict__ ic_in,
    const int* __restrict__ id_in,
    const int* __restrict__ m_in,
    const int* __restrict__ iab_in,
    const double* __restrict__ af,
    const double* __restrict__ bf,
    const double* __restrict__ binom,
    double* __restrict__ out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_pairs) return;

    int ia = ia_in[tid];

    if (ia < 0) {
        out[tid] = 0.0;
        return;
    }

    int ib = ib_in[tid];
    int ic = ic_in[tid];
    int id = id_in[tid];
    int m  = m_in[tid];
    int iab = iab_in[tid];

    const double* af_row = af + tid * 20;
    const double* bf_row = bf + tid * 13;

    double total_sum = 0.0;

    #pragma unroll 2
    for (int k1 = 0; k1 <= ia; ++k1) {
        double b_ia = binom[IDX2(ia, k1)];

        #pragma unroll 2
        for (int k2 = 0; k2 <= ib; ++k2) {
            double b_ib = binom[IDX2(ib, k2)];

            for (int k3 = 0; k3 <= ic; ++k3) {
                double b_ic = binom[IDX2(ic, k3)];

                for (int k4 = 0; k4 <= id; ++k4) {
                    double b_id = binom[IDX2(id, k4)];

                    for (int k5 = 0; k5 <= m; ++k5) {
                        double b_m5 = binom[IDX2(m, k5)];
                        int iaf_idx = iab - k1 - k2 + k3 + k4 + 2 * k5;
                        double val_af = af_row[iaf_idx];

                        for (int k6 = 0; k6 <= m; ++k6) {
                            double b_m6 = binom[IDX2(m, k6)];
                            int ibf_idx = k1 + k2 + k3 + k4 + 2 * k6;
                            double val_bf = bf_row[ibf_idx];
                            
                            double sgn = 1.0 - 2.0 * ((m + k2 + k4 + k5 + k6) & 1);
                            
                            total_sum += sgn * b_id * b_ic * b_ib * b_ia 
                                                 * b_m5 * b_m6 * val_af * val_bf;
                        }
                    }
                }
            }
        }
    }
    out[tid] = total_sum;
}

extern "C" {

void launch_ss_kernel_c(
    int n_pairs,
    int* ia, int* ib, int* ic, int* id, int* m, int* iab,
    double* af, double* bf,
    double* binom,
    double* out
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;

    ss_summation_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, ia, ib, ic, id, m, iab, af, bf, binom, out
    );
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_afn_kernel_c(
    int n_pairs,
    const double* p_vec,
    double* af_out
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;
    afn_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, p_vec, af_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_bfn_kernel_c(
    int n_pairs,
    const double* x,
    const double* taylor_coeffs,
    double* bf_out
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;
    bfn_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, x, taylor_coeffs, bf_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_rotation_transform_kernel(
    int n_pairs,
    const double* S_local,
    const double* C_tensor,
    double* di_out
)
{
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;
    rotation_transform_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, S_local, C_tensor, di_out
    );
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

}

} // extern "C"