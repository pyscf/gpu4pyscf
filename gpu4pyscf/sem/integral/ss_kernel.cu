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

extern "C" {

#define IDX2(r, c, stride) ((r) * (stride) + (c))

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


    for (int k1 = 0; k1 <= ia; ++k1) {
        double b_ia = binom[IDX2(ia, k1, 21)];

        for (int k2 = 0; k2 <= ib; ++k2) {
            double b_ib = binom[IDX2(ib, k2, 21)];

            for (int k3 = 0; k3 <= ic; ++k3) {
                double b_ic = binom[IDX2(ic, k3, 21)];

                for (int k4 = 0; k4 <= id; ++k4) {
                    double b_id = binom[IDX2(id, k4, 21)];

                    for (int k5 = 0; k5 <= m; ++k5) {
                        double b_m5 = binom[IDX2(m, k5, 21)];
                        int iaf_idx = iab - k1 - k2 + k3 + k4 + 2 * k5;
                        double val_af = af_row[iaf_idx];

                        for (int k6 = 0; k6 <= m; ++k6) {
                            double b_m6 = binom[IDX2(m, k6, 21)];
                            int ibf_idx = k1 + k2 + k3 + k4 + 2 * k6;
                            if (ibf_idx >= 0 && ibf_idx < 13) {
                                double val_bf = bf_row[ibf_idx];
                                int parity = (m + k2 + k4 + k5 + k6) & 1;
                                double sgn = (parity == 0) ? 1.0 : -1.0;

                                total_sum += sgn * b_id * b_ic * b_ib * b_ia 
                                                 * b_m5 * b_m6 * val_af * val_bf;
                            }
                        }
                        
                    }
                }
            }
        }
    }
    out[tid] = total_sum;
}

} // extern "C"