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

// nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC -arch=sm_70 eri_1c2e_kernel.cu -o liberi_1c2e_kernel.so

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BINOM_DIM 30
#define IDX2(r, c) ((r) * (BINOM_DIM) + (c))


__global__ void rsc_kernel(
    const int n_tasks,
    const double hartree2ev,
    const int* __restrict__ k_vec,
    const int* __restrict__ na_vec, const double* __restrict__ ea_vec,
    const int* __restrict__ nb_vec, const double* __restrict__ eb_vec,
    const int* __restrict__ nc_vec, const double* __restrict__ ec_vec,
    const int* __restrict__ nd_vec, const double* __restrict__ ed_vec,
    const double* __restrict__ fx_table, // Size 30
    const double* __restrict__ b_table,  // Size 30*30 flattened
    double* __restrict__ out_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tasks) return;

    int k  = k_vec[idx];
    int na = na_vec[idx]; double ea = ea_vec[idx];
    int nb = nb_vec[idx]; double eb = eb_vec[idx];
    int nc = nc_vec[idx]; double ec = ec_vec[idx];
    int nd = nd_vec[idx]; double ed = ed_vec[idx];

    const double log2 = 0.6931471805599453;

    double aea = log(ea);
    double aeb = log(eb);
    double aec = log(ec);
    double aed = log(ed);

    double eab = ea + eb;
    double ecd = ec + ed;
    double e   = eab + ecd;
    
    int nab = na + nb;
    int ncd = nc + nd;
    int n   = nab + ncd;

    double ae = log(e);
    double aeab = log(eab);
    double aecd = log(ecd);
    
    double term_denom = sqrt(fx_table[2*na] * fx_table[2*nb] * fx_table[2*nc] * fx_table[2*nd]);
    double ff = fx_table[n-1] / term_denom;

    double exponent_sum = na*aea + nb*aeb + nc*aec + nd*aed 
                        + 0.5*(aea + aeb + aec + aed) 
                        + log2*(n+2) - ae*n;
    
    double c_val = hartree2ev * ff * exp(exponent_sum);

    double s0 = 1.0 / e;
    double s1 = 0.0;
    double s2 = 0.0;

    int m = ncd - k;
    int m2 = ncd + k + 1;

    double ratio = e / ecd;

    for (int i = 0; i < m; ++i) {
        s0 *= ratio;
        
        double b1 = b_table[IDX2(ncd - k - 1, i)];
        double b2 = b_table[IDX2(m2 - 1, i)];
        double b3 = b_table[IDX2(n - 1, i)];

        s1 += s0 * (b1 - b2) / b3;
    }

    for (int i = m; i < m2; ++i) {
        s0 *= ratio;
        
        double b2 = b_table[IDX2(m2 - 1, i)];
        double b3 = b_table[IDX2(n - 1, i)];

        s2 += s0 * b2 / b3;
    }

    double b_last = b_table[IDX2(n - 1, m2 - 1)];
    double s3 = exp(ae*n - aecd*m2 - aeab*(nab-k)) / b_last;

    out_val[idx] = c_val * (s1 - s2 + s3);
}


extern "C" {

void launch_rsc_kernel_c(
    const int n_tasks,
    const double hartree2ev,
    const int* k_vec,
    const int* na, const double* ea,
    const int* nb, const double* eb,
    const int* nc, const double* ec,
    const int* nd, const double* ed,
    const double* fx_table,
    const double* b_table,
    double* out_val
) {
    int threads = 128;
    int blocks = (n_tasks + threads - 1) / threads;

    rsc_kernel<<<blocks, threads>>>(
        n_tasks, hartree2ev, k_vec,
        na, ea, nb, eb, nc, ec, nd, ed,
        fx_table, b_table, out_val
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (rsc_kernel): %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"