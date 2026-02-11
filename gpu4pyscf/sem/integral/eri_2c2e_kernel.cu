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

// nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC -arch=sm_70 eri_2c2e_kernel.cu -o liberi_2c2e_kernel.so

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define INV_SQRT(x) (1.0 / sqrt(x))
#define SQR(x) ((x) * (x))
#define SQR(x) ((x) * (x))

__device__ double charg_kernel_device(
    double r, 
    int l1, int l2, int m, 
    double da, double db, 
    double add
) {
    double rsq = r * r;

    // Q-Q
    if (l1 == 0 && l2 == 0) {
        return INV_SQRT(rsq + add);
    }

    // Z-Q
    if (l1 == 1 && l2 == 0) {
        double t1 = INV_SQRT((r + da)*(r + da) + add);
        double t2 = INV_SQRT((r - da)*(r - da) + add);
        return 0.5 * (t2 - t1);
    }
    // Q-Z
    if (l1 == 0 && l2 == 1) {
        double t1 = INV_SQRT((r + db)*(r + db) + add);
        double t2 = INV_SQRT((r - db)*(r - db) + add);
        return 0.5 * (t1 - t2);
    }
    if (l1 == 1 && l2 == 1) {
        // Z-Z (Sigma) m=0
        if (m == 0) {
            double t1 = INV_SQRT((r + da - db)*(r + da - db) + add);
            double t2 = INV_SQRT((r - da + db)*(r - da + db) + add);
            double t3 = INV_SQRT((r - da - db)*(r - da - db) + add);
            double t4 = INV_SQRT((r + da + db)*(r + da + db) + add);
            return 0.25 * (t1 + t2 - t3 - t4);
        }
        // X-X (Pi) m=1
        else if (m == 1) {
            double t1 = INV_SQRT(rsq + (da - db)*(da - db) + add);
            double t2 = INV_SQRT(rsq + (da + db)*(da + db) + add);
            return 0.25 * (2.0 * t1 - 2.0 * t2);
        }
    }

    // Q-ZZ
    if (l1 == 0 && l2 == 2) {
        double t1 = INV_SQRT((r - db)*(r - db) + add);
        double t2 = INV_SQRT(rsq + db*db + add);
        double t3 = INV_SQRT((r + db)*(r + db) + add);
        return 0.25 * (t1 - 2.0 * t2 + t3);
    }
    // ZZ-Q
    if (l1 == 2 && l2 == 0) {
        double t1 = INV_SQRT((r - da)*(r - da) + add);
        double t2 = INV_SQRT(rsq + da*da + add);
        double t3 = INV_SQRT((r + da)*(r + da) + add);
        return 0.25 * (t1 - 2.0 * t2 + t3);
    }

    // Z-ZZ (m=0)
    if (l1 == 1 && l2 == 2 && m == 0) {
        double t1 = INV_SQRT(SQR(r - da - db) + add);
        double t2 = INV_SQRT(SQR(r - da) + db*db + add);
        double t3 = INV_SQRT(SQR(r + db - da) + add);
        double t4 = INV_SQRT(SQR(r - db + da) + add);
        double t5 = INV_SQRT(SQR(r + da) + db*db + add);
        double t6 = INV_SQRT(SQR(r + da + db) + add);
        return 0.125 * (t1 - 2.0*t2 + t3 - t4 + 2.0*t5 - t6);
    }
    // ZZ-Z (m=0)
    if (l1 == 2 && l2 == 1 && m == 0) {
        double t1 = INV_SQRT(SQR(r - da - db) + add);
        double t2 = INV_SQRT(SQR(r - db) + da*da + add);
        double t3 = INV_SQRT(SQR(r + da - db) + add);
        double t4 = INV_SQRT(SQR(r - da + db) + add);
        double t5 = INV_SQRT(SQR(r + db) + da*da + add);
        double t6 = INV_SQRT(SQR(r + da + db) + add);
        return 0.125 * (-t1 + 2.0*t2 - t3 + t4 - 2.0*t5 + t6);
    }
    // X-ZX m=1
    if (l1 == 1 && l2 == 2 && m == 1) {
        double ab = db / 1.4142135623730951;
        double t1 = INV_SQRT(SQR(r - ab) + SQR(da - ab) + add);
        double t2 = INV_SQRT(SQR(r + ab) + SQR(da - ab) + add);
        double t3 = INV_SQRT(SQR(r - ab) + SQR(da + ab) + add);
        double t4 = INV_SQRT(SQR(r + ab) + SQR(da + ab) + add);
        return 0.125 * (-2.0*t1 + 2.0*t2 + 2.0*t3 - 2.0*t4);
    }
    // ZX-X m=1
    if (l1 == 2 && l2 == 1 && m == 1) {
        double aa = da / 1.4142135623730951;
        double t1 = INV_SQRT(SQR(r + aa) + SQR(aa - db) + add);
        double t2 = INV_SQRT(SQR(r - aa) + SQR(aa - db) + add);
        double t3 = INV_SQRT(SQR(r + aa) + SQR(aa + db) + add);
        double t4 = INV_SQRT(SQR(r - aa) + SQR(aa + db) + add);
        return 0.125 * (-2.0*t1 + 2.0*t2 + 2.0*t3 - 2.0*t4);
    }

    if (l1 == 2 && l2 == 2) {
        // ZZ-ZZ m=0
        if (m == 0) {
            double t1 = INV_SQRT(SQR(r - da - db) + add);
            double t2 = INV_SQRT(SQR(r + da + db) + add);
            double t3 = INV_SQRT(SQR(r - da + db) + add);
            double t4 = INV_SQRT(SQR(r + da - db) + add);
            
            double t5 = INV_SQRT(pow(r - da, 2) + db*db + add);
            double t6 = INV_SQRT(pow(r - db, 2) + da*da + add);
            double t7 = INV_SQRT(pow(r + da, 2) + db*db + add);
            double t8 = INV_SQRT(pow(r + db, 2) + da*da + add);
            
            double t9  = INV_SQRT(rsq + pow(da - db, 2) + add);
            double t10 = INV_SQRT(rsq + pow(da + db, 2) + add);
            
            double zzzz = t1 + t2 + t3 + t4 - 2.0*(t5 + t6 + t7 + t8) + 2.0*(t9 + t10);
            
            double t11 = INV_SQRT(rsq + da*da + db*db + add);
            double xyxy = 4.0*t9 + 4.0*t10 - 8.0*t11;
            
            return (zzzz / 16.0) - (xyxy / 64.0);
        }
        // ZX-ZX m=1
        if (m == 1) {
            double aa = da / 1.4142135623730951;
            double ab = db / 1.4142135623730951;
            
            double v1 = INV_SQRT(pow(r + aa - ab, 2) + pow(aa - ab, 2) + add);
            double v2 = INV_SQRT(pow(r + aa + ab, 2) + pow(aa - ab, 2) + add);
            double v3 = INV_SQRT(pow(r - aa - ab, 2) + pow(aa - ab, 2) + add);
            double v4 = INV_SQRT(pow(r - aa + ab, 2) + pow(aa - ab, 2) + add);
            
            double v5 = INV_SQRT(pow(r + aa - ab, 2) + pow(aa + ab, 2) + add);
            double v6 = INV_SQRT(pow(r + aa + ab, 2) + pow(aa + ab, 2) + add);
            double v7 = INV_SQRT(pow(r - aa - ab, 2) + pow(aa + ab, 2) + add);
            double v8 = INV_SQRT(pow(r - aa + ab, 2) + pow(aa + ab, 2) + add);
            
            return 0.0625 * (2.0*v1 - 2.0*v2 - 2.0*v3 + 2.0*v4 - 2.0*v5 + 2.0*v6 + 2.0*v7 - 2.0*v8);
        }
        if (m == 2) {
            double t1 = INV_SQRT(rsq + pow(da - db, 2) + add);
            double t2 = INV_SQRT(rsq + pow(da + db, 2) + add);
            double t3 = INV_SQRT(rsq + da*da + db*db + add);
            return 0.0625 * (4.0*t1 + 4.0*t2 - 8.0*t3);
        }
    }

    return 0.0;
}

__global__ void multipole_eval_kernel(
    const int n_pairs,
    const double* __restrict__ r_vec,   // (n_pairs,)
    const int* __restrict__ l1_vec,     // (n_pairs,)
    const int* __restrict__ l2_vec,     // (n_pairs,)
    const int* __restrict__ m_vec,      // (n_pairs,)
    const double* __restrict__ da_vec,  // (n_pairs,)
    const double* __restrict__ db_vec,  // (n_pairs,)
    const double* __restrict__ add_vec, // (n_pairs,)
    double* __restrict__ out_vec        // (n_pairs,)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;

    out_vec[idx] = charg_kernel_device(
        r_vec[idx],
        l1_vec[idx],
        l2_vec[idx],
        m_vec[idx],
        da_vec[idx],
        db_vec[idx],
        add_vec[idx]
    );
}

extern "C" {
void launch_multipole_eval_kernel_c(
    const int n_pairs,
    const double* r_vec,
    const int* l1_vec,
    const int* l2_vec,
    const int* m_vec,
    const double* da_vec,
    const double* db_vec,
    const double* add_vec,
    double* out_vec
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;
    multipole_eval_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, r_vec, l1_vec, l2_vec, m_vec, da_vec, db_vec, add_vec, out_vec
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}
} // extern "C"