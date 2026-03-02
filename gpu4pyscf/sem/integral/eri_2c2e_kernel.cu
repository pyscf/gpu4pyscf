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

__device__ double rijkl_device(
    int ni, int nj,                         // 0-based atom indices
    int ij, int kl,                         // Orbital pair combinations (0..44)
    int li, int lj,                         // Angular momentum for left pair (0=s, 1=p, 2=d)
    int lk, int ll,                         // Angular momentum for right pair
    int ic,                                 // Core flag: 1=left is core, 2=right is core, 0=normal
    double r,                               // Interatomic distance in Bohr
    int n_atom,                             // Total number of atoms (for tensor striding)
    const double* __restrict__ po_tensor,   // Shape: (3, 3, 3, n_atom)
    const double* __restrict__ ddp_tensor,  // Shape: (3, 3, n_atom)
    const double* __restrict__ core_rho,    // Shape: (n_atom,)
    const double* __restrict__ ch           // Shape: (45, 3, 5)
) {

    int l1min = abs(li - lj); 
    if (l1min > 2) l1min = 2;
    int l1max = li + lj;      
    if (l1max > 2) l1max = 2;
    int l2min = abs(lk - ll); 
    if (l2min > 2) l2min = 2;
    int l2max = lk + ll;      
    if (l2max > 2) l2max = 2;

    double total = 0.0;

    for (int l1 = l1min; l1 <= l1max; ++l1) {
        double pij = 0.0;
        double dij = 0.0;
        
        if (l1 == 0) {
            // Special case: electron-core interaction for left atom
            if (ic == 1 && li == 0 && lj == 0) {
                pij = core_rho[ni];
            } else {
                pij = po_tensor[li * (9 * n_atom) + lj * (3 * n_atom) + 0 * n_atom + ni];
            }
            dij = 0.0;
        } else {
            pij = po_tensor[li * (9 * n_atom) + lj * (3 * n_atom) + l1 * n_atom + ni];
            dij = ddp_tensor[li * (3 * n_atom) + lj * n_atom + ni];
        }

        for (int l2 = l2min; l2 <= l2max; ++l2) {
            double pkl = 0.0;
            double dkl = 0.0;
            
            if (l2 == 0) {
                // Special case: electron-core interaction for right atom
                if (ic == 2 && lk == 0 && ll == 0) {
                    pkl = core_rho[nj];
                } else {
                    pkl = po_tensor[lk * (9 * n_atom) + ll * (3 * n_atom) + 0 * n_atom + nj];
                }
                dkl = 0.0;
            } else {
                pkl = po_tensor[lk * (9 * n_atom) + ll * (3 * n_atom) + l2 * n_atom + nj];
                dkl = ddp_tensor[lk * (3 * n_atom) + ll * n_atom + nj];
            }

            double add = (pij + pkl) * (pij + pkl);
            int lmin_m = l1 < l2 ? l1 : l2;

            double s1 = 0.0;
            for (int m = -lmin_m; m <= lmin_m; ++m) {
                double c1 = ch[ij * 15 + l1 * 5 + (m + 2)];
                double c2 = ch[kl * 15 + l2 * 5 + (m + 2)];
                double ccc = c1 * c2;

                if (ccc == 0.0) continue;

                int mm = abs(m);
                s1 += charg_kernel_device(r, l1, l2, mm, dij, dkl, add) * ccc;
            }
            total += s1;
        }
    }
    return total;
}


// this function only used for debug!
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

__global__ void solve_poij_kernel(
    const int n_atoms,
    const int* __restrict__ l_vec,     // (N,)
    const double* __restrict__ d_vec,  // (N,)
    const double* __restrict__ fg_vec, // (N,)
    double* __restrict__ rho_vec,      // (N,) Output
    const double hartree2ev            // Constant passed from Python
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    int l = l_vec[idx];
    double d = d_vec[idx];
    double fg = fg_vec[idx];

    const int niter = 100;
    const double epsil = 1e-8;
    const double g1 = 0.382;
    const double g2 = 0.618;
    
    const double ev4 = hartree2ev / 4.0;
    const double ev8 = hartree2ev / 8.0;

    if (l == 0) {
        if (fabs(fg) < 1e-12) {
            rho_vec[idx] = 1e6;
        } else {
            rho_vec[idx] = 0.5 * hartree2ev / fg;
        }
        return;
    }

    double dsq = d * d;
    double a1 = 0.1;
    double a2 = 5.0;
    double f1 = 0.0;
    double f2 = 0.0;
    
    for (int i = 0; i < niter; ++i) {
        double delta = a2 - a1;
        if (delta < epsil) {
            break;
        }

        double y1 = a1 + delta * g1;
        double y2 = a1 + delta * g2;

        if (l == 1) {
            double term1_y1 = ev4 * (1.0/y1 - INV_SQRT(y1*y1 + dsq));
            double term1_y2 = ev4 * (1.0/y2 - INV_SQRT(y2*y2 + dsq));
            f1 = (term1_y1 - fg) * (term1_y1 - fg);
            f2 = (term1_y2 - fg) * (term1_y2 - fg);
        } else {
            double t1_y1 = INV_SQRT(y1*y1 + dsq * 0.5);
            double t2_y1 = INV_SQRT(y1*y1 + dsq);
            double term_y1 = ev8 * (1.0/y1 - 2.0 * t1_y1 + t2_y1);

            double t1_y2 = INV_SQRT(y2*y2 + dsq * 0.5);
            double t2_y2 = INV_SQRT(y2*y2 + dsq);
            double term_y2 = ev8 * (1.0/y2 - 2.0 * t1_y2 + t2_y2);

            f1 = (term_y1 - fg) * (term_y1 - fg);
            f2 = (term_y2 - fg) * (term_y2 - fg);
        }

        if (f1 < f2) {
            a2 = y2;
        } else {
            a1 = y1;
        }
    }

    if (f1 >= f2) {
        rho_vec[idx] = a2;
    } else {
        rho_vec[idx] = a1;
    }
}


// this function only used for debug!
__global__ void test_rijkl_kernel(
    int n_tasks, 
    int n_atom,
    const int* __restrict__ ni_vec, const int* __restrict__ nj_vec,
    const int* __restrict__ ij_vec, const int* __restrict__ kl_vec,
    const int* __restrict__ li_vec, const int* __restrict__ lj_vec,
    const int* __restrict__ lk_vec, const int* __restrict__ ll_vec,
    const int* __restrict__ ic_vec, const double* __restrict__ r_vec,
    const double* __restrict__ po_tensor,
    const double* __restrict__ ddp_tensor,
    const double* __restrict__ core_rho,
    const double* __restrict__ ch,
    double* __restrict__ out_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tasks) return;

    out_val[idx] = rijkl_device(
        ni_vec[idx], nj_vec[idx], 
        ij_vec[idx], kl_vec[idx],
        li_vec[idx], lj_vec[idx], 
        lk_vec[idx], ll_vec[idx],
        ic_vec[idx], r_vec[idx], 
        n_atom,
        po_tensor, ddp_tensor, core_rho, ch
    );
}


extern "C" {
// this function only used for debug!
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

void launch_solve_poij_kernel_c(
    const int n_atoms,
    const int* l_vec,
    const double* d_vec,
    const double* fg_vec,
    double* rho_vec,
    const double hartree2ev
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_atoms + threads_per_block - 1) / threads_per_block;

    solve_poij_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_atoms, l_vec, d_vec, fg_vec, rho_vec, hartree2ev
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (solve_poij): %s\n", cudaGetErrorString(err));
    }
}

void launch_test_rijkl_kernel_c(
    int n_tasks, int n_atom,
    const int* ni_vec, const int* nj_vec,
    const int* ij_vec, const int* kl_vec,
    const int* li_vec, const int* lj_vec,
    const int* lk_vec, const int* ll_vec,
    const int* ic_vec, const double* r_vec,
    const double* po_tensor, const double* ddp_tensor,
    const double* core_rho, const double* ch,
    double* out_val
) {
    int threads = 128;
    int blocks = (n_tasks + threads - 1) / threads;
    
    test_rijkl_kernel<<<blocks, threads>>>(
        n_tasks, n_atom,
        ni_vec, nj_vec, ij_vec, kl_vec,
        li_vec, lj_vec, lk_vec, ll_vec,
        ic_vec, r_vec,
        po_tensor, ddp_tensor, core_rho, ch,
        out_val
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (test_rijkl_kernel): %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"