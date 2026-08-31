/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "multigrid.cuh"

__device__ static
void init_orth_data(double *pool, int *grid_start,
                    MGridEnvVars envs, MGridBounds bounds, double *ri, double *rj,
                    double ai, double aj, int l)
{
    int thread_id = threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int ngrid_span = bounds.ngrid_radius * 2;
    int l1 = l + 1;
    int nl_gridx = ngrid_span * l1;
    int xs_size = nl_gridx * WARP_SIZE;
    int z = warp_id % 3;
    int w = warp_id / 3;
    double aij = ai + aj;
    double aj_aij = aj / aij;
    if (w < 2) {
        int nx_per_cell = bounds.mesh[z];
        double *exp_cache = pool + z * xs_size;
        double xjxi = rj[z] - ri[z];
        double xij = xjxi * aj_aij + ri[z];
        double dx = envs.lattice_params[z] / nx_per_cell;
        int xij_latt = static_cast<int>(floor(xij / dx));
        if (w == 0) {
            int x0_latt = xij_latt + 1 - bounds.ngrid_radius;
            grid_start[z*WARP_SIZE] = x0_latt;
        }
        double base_x = dx * xij_latt;
        double x0xij = base_x - xij;
        double _x0x0 = -aij * x0xij * x0xij;
        double _dxdx = -aij * dx * dx;
        double _x0dx = -2 * aij * x0xij * dx;
        double exp_2dxdx = exp(2 * _dxdx);
        double exp_x0x0 = exp(_x0x0);
        int istart = bounds.ngrid_radius - 1;
        if (_x0x0 < -40) { // 2e-18
            exp_x0x0 = 0.;
            exp_2dxdx = 0.;
            _x0dx = 0.; // exp(_x0dx) may singular when aij is large
        }
        if (w == 0) {
            double exp_x0dx = exp(_dxdx - _x0dx);
            for (int i = istart; i >= 0; --i) {
                exp_cache[i*WARP_SIZE] = exp_x0x0;
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
            }
        } else {
            double exp_x0dx = exp(_dxdx + _x0dx);
            for (int i = istart+1; i < ngrid_span; ++i) {
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
                exp_cache[i*WARP_SIZE] = exp_x0x0;
            }
        }
    }
    __syncthreads();

    for (int z = 0; z < 3; ++z) {
        int nx_per_cell = bounds.mesh[z];
        int x0_latt = grid_start[z*WARP_SIZE];
        double dx = envs.lattice_params[z] / nx_per_cell;
        double x0xi = x0_latt * dx - ri[z];
        double *xs_exp = pool + z * xs_size;

        for (int i = warp_id; i < ngrid_span; i += WARPS) {
            double gridx = x0xi + i * dx;
            double s1 = xs_exp[i*WARP_SIZE];
            for (int m = 1; m <= l; m++) {
                s1 *= gridx;
                xs_exp[(m*ngrid_span+i)*WARP_SIZE] = s1;
            }
        }
        __syncthreads();

        // add up contributions from all images to the reference image
        if (ngrid_span >= nx_per_cell) {
            for (int n = warp_id; n < nx_per_cell*l1; n += WARPS) {
                int i = n % nx_per_cell;
                int m = n / nx_per_cell;
                double s1 = 0.;
                double *xe = xs_exp + m*ngrid_span*WARP_SIZE;
                for (; i < ngrid_span; i += nx_per_cell) {
                    s1 += xe[i*WARP_SIZE];
                }
                i = n % nx_per_cell;
                xs_exp[(m*ngrid_span + i)*WARP_SIZE] = s1;
            }
        }
        if (warp_id == 0) {
            x0_latt = (x0_latt % nx_per_cell + nx_per_cell) % nx_per_cell;
            grid_start[z*WARP_SIZE] = x0_latt;
        }
    }
    __syncthreads();
}

__device__ inline
int load_xs(double *xs_cache, double *xs_exp, int ix0, int ngridx,
            int l, int batch_size, int xs_stride, int warp_id)
{
    int nx = MIN(ngridx - ix0, batch_size);
    double *_xs_exp = xs_exp + ix0 * WARP_SIZE;
    for (int i = warp_id; i < nx; i += WARPS) {
        for (int m = 0; m <= l; ++m) {
            xs_cache[(m*batch_size+i)*WARP_SIZE] = _xs_exp[(m*xs_stride+i)*WARP_SIZE];
        }
    }
    __syncthreads();
    return nx;
}

__device__ static
double reduce_warps(double val, int ngridx, int thread_id, int sp_id, int warp_id)
{
    __shared__ double cache[THREADS];
    cache[thread_id] = val;
    __syncthreads();
    for (int stride = 4; stride > 0; stride /= 2) {
        if (warp_id < stride && warp_id + stride < ngridx) {
            cache[thread_id] += cache[thread_id + stride*WARP_SIZE];
        }
        __syncthreads();
    }
    return cache[sp_id];
}

template <int L> __device__ static
void fill_dm_xyz(double *dm_xyz, double *gx_dmyz, double *xs_exp,
                 int ngridx, int ngrid_span)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    constexpr int L1 = L + 1;
    constexpr int nf2 = (L+1)*(L+2)/2;
#if 0
    // this algorithm seems more efficient for large L
    double r3[(L1*nf2+WARPS-1)/WARPS];
#pragma unroll
    for (int n = 0; n < (L1*nf2+WARPS-1)/WARPS; ++n) {
        r3[n] = 0.;
    }
    extern __shared__ double cache[];
    double *xs_cache = cache + sp_id;
    double *yz_cache = cache + L1 * WARP_SIZE + sp_id;
    for (int ix = 0; ix < ngridx; ++ix) {
        __syncthreads();
        double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
        for (int m = warp_id; m < nf2; m += WARPS) {
            yz_cache[m*WARP_SIZE] = pgx[m*WARP_SIZE];
        }
        for (int m = warp_id; m <= L; m += WARPS) {
            xs_cache[m*WARP_SIZE] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE];
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < (L1*nf2)/WARPS; ++n) {
            int m = n * WARPS + warp_id;
            int myz = m / L1;
            int mx  = m % L1;
            r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
        }
        int n = (L1*nf2)/WARPS;
        int m = n * WARPS + warp_id;
        if (m < nf2*L1) {
            int myz = m / L1;
            int mx  = m % L1;
            r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
        }
    }
#pragma unroll
    for (int n = 0; n < (L1*nf2)/WARPS; ++n) {
        int m = n * WARPS + warp_id;
        dm_xyz[m*WARP_SIZE] = r3[n];
    }
    int n = (L1*nf2)/WARPS;
    int m = n * WARPS + warp_id;
    if (m < nf2*L1) {
        dm_xyz[m*WARP_SIZE] = r3[n];
    }
#else
    constexpr int nf3 = nf2*(L+3)/3;
    if (L <= 3) {
        double r2[nf3];
        double r1[L1];
#pragma unroll
        for (int m = 0; m < nf3; ++m) {
            r2[m] = 0.;
        }
        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
            for (int mx = 0; mx <= L; ++mx) {
                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
            }
            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
#pragma unroll
            for (int my = 0; my <= L; ++my) {
#pragma unroll
                for (int mz = 0; mz <= L-my; ++mz) {
                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                    for (int mx = 0; mx <= L-my-mz; ++mx) {
                        r2[ADDR3(L,mx,my,mz)] += r1[mx] * t;
                    }
                }
            }
        }
#pragma unroll
        for (int mx = 0; mx <= L; ++mx) {
#pragma unroll
            for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
                for (int mz = 0; mz <= L-mx-my; ++mz) {
                    int n = ADDR3(L,mx,my,mz);
                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
                    if (warp_id == 0) {
                        dm_xyz[(ADDR2(L,my,mz)*L1+mx)*WARP_SIZE] = r2[n];
                    }
                }
            }
        }
    } else {
        double r3[(L1*nf2+WARPS-1)/WARPS];
#pragma unroll
        for (int n = 0; n < (L1*nf2+WARPS-1)/WARPS; ++n) {
            r3[n] = 0.;
        }
        extern __shared__ double cache[];
        double *xs_cache = cache + sp_id;
        double *yz_cache = cache + L1 * WARP_SIZE + sp_id;
        for (int ix = 0; ix < ngridx; ++ix) {
            __syncthreads();
            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
            for (int m = warp_id; m < nf2; m += WARPS) {
                yz_cache[m*WARP_SIZE] = pgx[m*WARP_SIZE];
            }
            for (int m = warp_id; m <= L; m += WARPS) {
                xs_cache[m*WARP_SIZE] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE];
            }
            __syncthreads();
#pragma unroll
            for (int n = 0; n < (L1*nf2)/WARPS; ++n) {
                int m = n * WARPS + warp_id;
                int myz = m / L1;
                int mx  = m % L1;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
            int n = (L1*nf2)/WARPS;
            int m = n * WARPS + warp_id;
            if (m < nf2*L1) {
                int myz = m / L1;
                int mx  = m % L1;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
        }
#pragma unroll
        for (int n = 0; n < (L1*nf2)/WARPS; ++n) {
            int m = n * WARPS + warp_id;
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
        int n = (L1*nf2)/WARPS;
        int m = n * WARPS + warp_id;
        if (m < nf2*L1) {
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
//    } else if (L == 7) {
//        double r2[64];
//        double r1[8];
//        // reuse the global memory in pool. (L+1)*(L+2)*(L+3)/6 is generally smaller
//        // than (ngrid_span * (L+1) * 2) occupied by ys_exp and zs_exp
//        int offset = (L+1)*(L+2)/2 + L*(L+1)/2;
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= 1; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int my = 0; my <= L; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= MIN(1, L-my-mz); ++mx) {
//                        r2[ADDR3(L,mx,my,mz)] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int mx = 0; mx <= 1; ++mx) {
//#pragma unroll
//            for (int my = 0; my <= L-mx; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-mx-my; ++mz) {
//                    int n = ADDR3(L,mx,my,mz);
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L1+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//
//#pragma unroll
//        for (int m = 0; m < nf3-offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 2; mx <= L; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int my = 0; my <= L-2; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-2-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 2; mx <= L-my-mz; ++mx) {
//                        r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int mx = 2; mx <= L; ++mx) {
//#pragma unroll
//            for (int my = 0; my <= L-mx; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-mx-my; ++mz) {
//                    int n = ADDR3(L,mx,my,mz);
//                    r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L1+mx)*WARP_SIZE] = r2[n-offset];
//                    }
//                }
//            }
//        }
//    } else if (L == 8) {
//        double r2[64];
//        double r1[9];
//        // reuse the global memory in pool. (L+1)*(L+2)*(L+3)/6 is generally smaller
//        // than (ngrid_span * (L+1) * 2) occupied by ys_exp and zs_exp
//        int offset = (L+1)*(L+2)/2;
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//            r1[0] = xs_exp[ix*WARP_SIZE];
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int my = 0; my <= L; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//                    r2[ADDR3(L,0,my,mz)] += r1[0] * t;
//                }
//            }
//        }
//#pragma unroll
//        for (int my = 0; my <= L; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//                int n = ADDR3(L,0,my,mz);
//                r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                if (warp_id == 0) {
//                    dm_xyz[ADDR2(L,my,mz)*L1*WARP_SIZE] = r2[n];
//                }
//            }
//        }
//
//#pragma unroll
//        for (int m = 0; m < L*(L+1)/2 + (L-1)*L/2; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 1; mx <= 2; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int my = 0; my <= L-1; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-1-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 1; mx <= MIN(2, L-my-mz); ++mx) {
//                        r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int mx = 1; mx <= 2; ++mx) {
//#pragma unroll
//            for (int my = 0; my <= L-mx; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-mx-my; ++mz) {
//                    int n = ADDR3(L,mx,my,mz);
//                    r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L1+mx)*WARP_SIZE] = r2[n-offset];
//                    }
//                }
//            }
//        }
//
//        offset = (L+1)*(L+2)/2 + L*(L+1)/2 + (L-1)*L/2;
//#pragma unroll
//        for (int m = 0; m < nf3 - offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 3; mx <= L; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int my = 0; my <= L-3; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-3-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 3; mx <= L-my-mz; ++mx) {
//                        r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int mx = 3; mx <= L; ++mx) {
//#pragma unroll
//            for (int my = 0; my <= L-mx; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-mx-my; ++mz) {
//                    int n = ADDR3(L,mx,my,mz);
//                    r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L1+mx)*WARP_SIZE] = r2[n-offset];
//                    }
//                }
//            }
//        }
    }
#endif
}
