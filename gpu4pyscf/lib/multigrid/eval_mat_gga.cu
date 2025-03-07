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
#include "cart2xyz.cu"
#include "loader.cu"

template <int L> __device__ static
void fill_dm_xyz_ip1(double *dm_xyz, double *gx_dmyz, double *xs_exp,
                     int ngridx, int ngrid_span)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    constexpr int L1 = L + 1;
    constexpr int L2 = L + 2;
    constexpr int nf2 = (L+1)*(L+2)/2;
#if 0
    // this algorithm seems more efficient for large L
    double r3[(L2*nf2+WARPS-1)/WARPS];
#pragma unroll
    for (int n = 0; n < (L2*nf2+WARPS-1)/WARPS; ++n) {
        r3[n] = 0.;
    }
    extern __shared__ double cache[];
    double *xs_cache = cache + sp_id;
    double *yz_cache = cache + (L+2) * WARP_SIZE + sp_id;
    for (int ix = 0; ix < ngridx; ++ix) {
        __syncthreads();
        double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
        for (int m = warp_id; m < nf2; m += WARPS) {
            yz_cache[m*WARP_SIZE] = pgx[m*WARP_SIZE];
        }
        for (int m = warp_id; m <= L1; m += WARPS) {
            xs_cache[m*WARP_SIZE] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE];
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < (L2*nf2)/WARPS; ++n) {
            int m = n * WARPS + warp_id;
            int myz = m / L2;
            int mx  = m % L2;
            r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
        }
        int n = (L2*nf2)/WARPS;
        int m = n * WARPS + warp_id;
        if (m < nf2*L2) {
            int myz = m / L2;
            int mx  = m % L2;
            r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
        }
    }
#pragma unroll
    for (int n = 0; n < (L2*nf2)/WARPS; ++n) {
        int m = n * WARPS + warp_id;
        dm_xyz[m*WARP_SIZE] = r3[n];
    }
    int n = (L2*nf2)/WARPS;
    int m = n * WARPS + warp_id;
    if (m < nf2*L2) {
        dm_xyz[m*WARP_SIZE] = r3[n];
    }
#else
    constexpr int nf3 = nf2*(L+3)/3;
    if (L <= 3) {
        double r2[nf3 + nf2];
        double r1[L2];
#pragma unroll
        for (int m = 0; m < nf3+nf2; ++m) {
            r2[m] = 0.;
        }
        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
            for (int mx = 0; mx <= L1; ++mx) {
                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
            }
            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
#pragma unroll
            for (int n = 0, my = 0; my <= L; ++my) {
#pragma unroll
                for (int mz = 0; mz <= L-my; ++mz) {
                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
                        r2[n] += r1[mx] * t;
                    }
                }
            }
        }
#pragma unroll
        for (int n = 0, my = 0; my <= L; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-my; ++mz) {
#pragma unroll
                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
                    if (warp_id == 0) {
                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
                    }
                }
            }
        }
    } else {
        double r3[(L2*nf2+WARPS-1)/WARPS];
#pragma unroll
        for (int n = 0; n < (L2*nf2+WARPS-1)/WARPS; ++n) {
            r3[n] = 0.;
        }
        extern __shared__ double cache[];
        double *xs_cache = cache + sp_id;
        double *yz_cache = cache + (L+2) * WARP_SIZE + sp_id;
        for (int ix = 0; ix < ngridx; ++ix) {
            __syncthreads();
            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
            for (int m = warp_id; m < nf2; m += WARPS) {
                yz_cache[m*WARP_SIZE] = pgx[m*WARP_SIZE];
            }
            for (int m = warp_id; m <= L1; m += WARPS) {
                xs_cache[m*WARP_SIZE] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE];
            }
            __syncthreads();
#pragma unroll
            for (int n = 0; n < (L2*nf2)/WARPS; ++n) {
                int m = n * WARPS + warp_id;
                int myz = m / L2;
                int mx  = m % L2;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
            int n = (L2*nf2)/WARPS;
            int m = n * WARPS + warp_id;
            if (m < nf2*L2) {
                int myz = m / L2;
                int mx  = m % L2;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
        }
#pragma unroll
        for (int n = 0; n < (L2*nf2)/WARPS; ++n) {
            int m = n * WARPS + warp_id;
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
        int n = (L2*nf2)/WARPS;
        int m = n * WARPS + warp_id;
        if (m < nf2*L2) {
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
//    } else if (L == 6) {
//        double r2[62];
//        double r1[L2];
//        int offset = 62; // corresponding to my=0..1, mz=0..L-my, mx=0..L1-my-mz
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 0; my <= 1; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 0; my <= 1; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//
//#pragma unroll
//        for (int m = 0; m < nf3+nf2-offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1-2; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 2; my <= L; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 2; my <= L; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//    } else if (L == 7) {
//        double r2[79];
//        double r1[L2];
//        int offset = 79; // corresponding to my=0..1, mz=0..L-my, mx=0..L1-my-mz
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 0; my <= 1; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 0; my <= 1; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//
//#pragma unroll
//        for (int m = 0; m < nf3+nf2-offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1-2; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 2; my <= L; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 2; my <= L; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//    } else if (L == 8) {
//        double r2[79];
//        double r1[L2];
//        int offset = 54; // corresponding to my=0, mz=0..L-my, mx=0..L1-my-mz
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, mz = 0; mz <= L; ++mz) {
//                double t = pgx[ADDR2(L,0,mz)*WARP_SIZE];
//#pragma unroll
//                for (int mx = 0; mx <= L1-mz; ++mx, ++n) {
//                    r2[n] += r1[mx] * t;
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, mz = 0; mz <= L; ++mz) {
//#pragma unroll
//            for (int mx = 0; mx <= L1-mz; ++mx, ++n) {
//                r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                if (warp_id == 0) {
//                    dm_xyz[(ADDR2(L,0,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                }
//            }
//        }
//
//        offset = 79; // corresponding to my=1..3, mz=0..L-my, mx=0..L1-my-mz
//#pragma unroll
//        for (int m = 0; m < offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1-1; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 1; my <= 2; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 1; my <= 2; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
//
//        offset = 133; // corresponding to my=0..3, mz=0..L-my, mx=0..L1-my-mz
//#pragma unroll
//        for (int m = 0; m < nf3+nf2-offset; ++m) {
//            r2[m] = 0.;
//        }
//        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
//#pragma unroll
//            for (int mx = 0; mx <= L1-3; ++mx) {
//                r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
//            }
//            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
//#pragma unroll
//            for (int n = 0, my = 3; my <= L; ++my) {
//#pragma unroll
//                for (int mz = 0; mz <= L-my; ++mz) {
//                    double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
//#pragma unroll
//                    for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                        r2[n] += r1[mx] * t;
//                    }
//                }
//            }
//        }
//#pragma unroll
//        for (int n = 0, my = 3; my <= L; ++my) {
//#pragma unroll
//            for (int mz = 0; mz <= L-my; ++mz) {
//#pragma unroll
//                for (int mx = 0; mx <= L1-my-mz; ++mx, ++n) {
//                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
//                    if (warp_id == 0) {
//                        dm_xyz[(ADDR2(L,my,mz)*L2+mx)*WARP_SIZE] = r2[n];
//                    }
//                }
//            }
//        }
    }
#endif
}

template <int L> __device__ inline
double sub_dm_xyz_to_dm(int lx_i, int ly_i, int lz_i, int lx_j, int ly_j, int lz_j,
                        int lj2, double ai2, double aj2,
                        double *cx, double *cy, double *cz, double *dm_xyz)
{
    int L2 = L + 2;
    double dm_ij = 0.;
    double fac;
    for (int jx = 0; jx <= lx_j; ++jx) {
        double fac_cx = cx[(jx+lx_j*lj2)*WARP_SIZE];
        int lx = lx_i + jx;
        for (int jy = 0; jy <= ly_j; ++jy) {
            double cxy = fac_cx * cy[(jy+ly_j*lj2)*WARP_SIZE];
            int ly = ly_i + jy;
            int xy_idx = ADDR2(L, lx, ly);
            if (lz_i > 0) {
                fac = lz_i * cxy;
                for (int jz = 0; jz <= lz_j; ++jz) {
                    int lz = lz_i - 1 + jz;
                    double cxyz = fac * cz[(jz+lz_j*lj2)*WARP_SIZE];
                    dm_ij += cxyz * dm_xyz[(xy_idx*L2+lz)*WARP_SIZE];
                }
            }
            fac = ai2 * cxy;
            for (int jz = 0; jz <= lz_j; ++jz) {
                int lz = lz_i + 1 + jz;
                double cxyz = fac * cz[(jz+lz_j*lj2)*WARP_SIZE];
                dm_ij += cxyz * dm_xyz[(xy_idx*L2+lz)*WARP_SIZE];
            }
            if (lz_j > 0) {
                fac = lz_j * cxy;
                for (int jz = 0; jz <= lz_j-1; ++jz) {
                    int lz = lz_i + jz;
                    double cxyz = fac * cz[(jz+(lz_j-1)*lj2)*WARP_SIZE];
                    dm_ij += cxyz * dm_xyz[(xy_idx*L2+lz)*WARP_SIZE];
                }
            }
            fac = aj2 * cxy;
            for (int jz = 0; jz <= lz_j+1; ++jz) {
                int lz = lz_i + jz;
                double cxyz = fac * cz[(jz+(lz_j+1)*lj2)*WARP_SIZE];
                dm_ij += cxyz * dm_xyz[(xy_idx*L2+lz)*WARP_SIZE];
            }
        }
    }
    return dm_ij;
}

template <int L> __device__ static
void _dm_xyz_to_dm_sigmax(double *dm, double *dm_yzx, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj1 = lj + 1;
    int lj2 = lj + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj2 * lj2 * WARP_SIZE;
    double *cz = cy + lj2 * lj2 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj2*lj2*WARP_SIZE, ri[n], rj[n], lj1);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    Fold2Index *i_fold2idx = c_i_in_fold2idx + li*nfi/3;
    Fold2Index *j_fold2idx = c_i_in_fold2idx + lj*nfj/3;
    for (int n = warp_id; n < nfij; n += WARPS) {
        int i = n / nfj;
        int j = n % nfj;
        int lx_i = i_fold2idx[i].x;
        int ly_i = i_fold2idx[i].y;
        int lz_i = li - lx_i - ly_i;
        int lx_j = j_fold2idx[j].x;
        int ly_j = j_fold2idx[j].y;
        int lz_j = lj - lx_j - ly_j;
        double dm_ij = sub_dm_xyz_to_dm<L>(ly_i, lz_i, lx_i, ly_j, lz_j, lx_j,
                                           lj2, ai2, aj2, cy, cz, cx, dm_yzx);
        atomicAdd(dm+i*nao+j, cicj * dm_ij);
    }
}

template <int L> __device__ static
void _dm_xyz_to_dm_sigmay(double *dm, double *dm_xzy, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj1 = lj + 1;
    int lj2 = lj + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj2 * lj2 * WARP_SIZE;
    double *cz = cy + lj2 * lj2 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj2*lj2*WARP_SIZE, ri[n], rj[n], lj1);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    Fold2Index *i_fold2idx = c_i_in_fold2idx + li*nfi/3;
    Fold2Index *j_fold2idx = c_i_in_fold2idx + lj*nfj/3;
    for (int n = warp_id; n < nfij; n += WARPS) {
        int i = n / nfj;
        int j = n % nfj;
        int lx_i = i_fold2idx[i].x;
        int ly_i = i_fold2idx[i].y;
        int lz_i = li - lx_i - ly_i;
        int lx_j = j_fold2idx[j].x;
        int ly_j = j_fold2idx[j].y;
        int lz_j = lj - lx_j - ly_j;
        double dm_ij = sub_dm_xyz_to_dm<L>(lx_i, lz_i, ly_i, lx_j, lz_j, ly_j,
                                           lj2, ai2, aj2, cx, cz, cy, dm_xzy);
        atomicAdd(dm+i*nao+j, cicj * dm_ij);
    }
}

template <int L> __device__ static
void _dm_xyz_to_dm_sigmaz(double *dm, double *dm_xyz, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj1 = lj + 1;
    int lj2 = lj + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj2 * lj2 * WARP_SIZE;
    double *cz = cy + lj2 * lj2 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj2*lj2*WARP_SIZE, ri[n], rj[n], lj1);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    Fold2Index *i_fold2idx = c_i_in_fold2idx + li*nfi/3;
    Fold2Index *j_fold2idx = c_i_in_fold2idx + lj*nfj/3;
    for (int n = warp_id; n < nfij; n += WARPS) {
        int i = n / nfj;
        int j = n % nfj;
        int lx_i = i_fold2idx[i].x;
        int ly_i = i_fold2idx[i].y;
        int lz_i = li - lx_i - ly_i;
        int lx_j = j_fold2idx[j].x;
        int ly_j = j_fold2idx[j].y;
        int lz_j = lj - lx_j - ly_j;
        double dm_ij = sub_dm_xyz_to_dm<L>(lx_i, ly_i, lz_i, lx_j, ly_j, lz_j,
                                           lj2, ai2, aj2, cx, cy, cz, dm_xyz);
        atomicAdd(dm+i*nao+j, cicj * dm_ij);
    }
}

template <int L, int TILE> __device__ static
void _eval_mat_gga_kernel(double *out, double *rho, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t pair_idx0)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int npairs_this_block = MIN(bounds.nshl_pair - pair_idx0, WARP_SIZE);
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t pair_idx = pair_idx0 + sp_id;
    if (sp_id >= npairs_this_block) {
        pair_idx = pair_idx0;
    }
    int bas_ij = bounds.bas_ij_idx[pair_idx];
    int nbas_j = envs.nbas_j;
    int ish = bas_ij / nbas_j;
    int jsh = bas_ij % nbas_j;
    int li = bas[ish*PRIMBAS_SLOTS+PRIMBAS_ANG];
    int lj = bas[jsh*PRIMBAS_SLOTS+PRIMBAS_ANG];
    double ai = env[bas[ish*PRIMBAS_SLOTS+PRIMBAS_EXP]];
    double aj = env[bas[jsh*PRIMBAS_SLOTS+PRIMBAS_EXP]];
    double ci = env[bas[ish*PRIMBAS_SLOTS+PRIMBAS_COEFF]];
    double cj = env[bas[jsh*PRIMBAS_SLOTS+PRIMBAS_COEFF]];
    double *ri = env + bas[ish*PRIMBAS_SLOTS+PRIMBAS_COORD];
    double *rj = env + bas[jsh*PRIMBAS_SLOTS+PRIMBAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    double aij = ai + aj;
    double theta_ij = ai * aj / aij;
    double cicj = ci * cj * exp(-theta_ij * rr_ij);
    if (sp_id >= npairs_this_block) {
        cicj = 0.;
    }
    double ai2 = -2. * ai;
    double aj2 = -2. * aj;
    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    out += i0*nao+j0;

    constexpr int nf2 = (L+1)*(L+2)/2;
    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int mesh_xyz = mesh_x * mesh_yz;
    double *vx = rho + mesh_xyz;
    double *vy = vx + mesh_xyz;
    double *vz = vy + mesh_xyz;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = ngrid_span * (L+2) * WARP_SIZE;
    int *grid_start = (int *)pool + sp_id;
    double *xs_exp = pool + WARP_SIZE*3 + sp_id;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *gx_dmyz = zs_exp + xs_size;
    init_orth_data(xs_exp, grid_start, envs, bounds, ri, rj, ai, aj, L+1);

    extern __shared__ double cache[];
    double *xs_cache, *ys_cache, *zs_cache;
    double *dm_xyz = gx_dmyz + nf2 * ngrid_span * WARP_SIZE;
    if (L < 4) {
        dm_xyz = cache + (MIN(LMAX,L)+2)*(MIN(LMAX,L)+2)*3*WARP_SIZE + sp_id;
    }

    int nx0 = grid_start[0*WARP_SIZE];
    int ny0 = grid_start[1*WARP_SIZE];
    int nz0 = grid_start[2*WARP_SIZE];
    // Using translation vectors to shift the first grid to cell 0.
    // This can simplify the address computation for the wrapped-around grid.
    nx0 = (nx0 % mesh_x + mesh_x) % mesh_x;
    ny0 = (ny0 % mesh_y + mesh_y) % mesh_y;
    nz0 = (nz0 % mesh_z + mesh_z) % mesh_z;
    int ngridx = ngrid_span;
    int ngridy = ngrid_span;
    int ngridz = ngrid_span;
    if (ngrid_span > mesh_x) {
        ngridx = mesh_x;
    }
    if (ngrid_span > mesh_y) {
        ngridy = mesh_y;
    }
    if (ngrid_span > mesh_z) {
        ngridz = mesh_z;
    }

    ys_cache = cache + sp_id;
    zs_cache = ys_cache + TILE * (L+1) * WARP_SIZE;
    for (int n = warp_id; n < ngridx*nf2; n += WARPS) {
        gx_dmyz[n*WARP_SIZE] = 0.;
    }
    for (int iz0 = 0; iz0 < ngridz; iz0 += TILE) {
        __syncthreads();
        int nz = load_xs(zs_cache, zs_exp, iz0, ngridz, L, TILE, ngrid_span, warp_id);
        for (int iy0 = 0; iy0 < ngridy; iy0 += TILE) {
            __syncthreads();
            int ny = load_xs(ys_cache, ys_exp, iy0, ngridy, L, TILE, ngrid_span, warp_id);
            for (int ix = warp_id; ix < ngridx; ix += WARPS) {
                int tx = (ix + nx0) % mesh_x;
                double r2[nf2];
                double r1[L+1];
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    r2[m] = 0.;
                }
                for (int iy = 0; iy < ny; ++iy) {
                    int ty = (iy + iy0 + ny0) % mesh_y;
#pragma unroll
                    for (int mz = 0; mz <= L; ++mz) {
                        r1[mz] = 0.;
                    }
                    for (int iz = 0; iz < nz; ++iz) {
                        int tz = (iz + iz0 + nz0) % mesh_z;
                        double r = rho[tx*mesh_yz+ty*mesh_z+tz];
#pragma unroll
                        for (int mz = 0; mz <= L; ++mz) {
                            r1[mz] += zs_cache[(mz*TILE+iz)*WARP_SIZE] * r;
                        }
                    }
#pragma unroll
                    for (int my = 0; my <= L; ++my) {
                        double fac = ys_cache[(my*TILE+iy)*WARP_SIZE];
#pragma unroll
                        for (int mz = 0; mz <= L-my; ++mz) {
                            r2[ADDR2(L,my,mz)] += fac * r1[mz];
                        }
                    }
                }
                double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgx[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz<L>(dm_xyz, gx_dmyz, xs_exp, ngridx, ngrid_span);
    dm_xyz_to_dm<L>(out, dm_xyz, nao, li, lj, ri, rj, cicj, cache,
                    npairs_this_block);
    __syncthreads();

    // sigmax: contract mesh_y, mesh_z
    for (int n = warp_id; n < ngridx*nf2; n += WARPS) {
        gx_dmyz[n*WARP_SIZE] = 0.;
    }
    for (int iz0 = 0; iz0 < ngridz; iz0 += TILE) {
        __syncthreads();
        int nz = load_xs(zs_cache, zs_exp, iz0, ngridz, L, TILE, ngrid_span, warp_id);
        for (int iy0 = 0; iy0 < ngridy; iy0 += TILE) {
            __syncthreads();
            int ny = load_xs(ys_cache, ys_exp, iy0, ngridy, L, TILE, ngrid_span, warp_id);
            for (int ix = warp_id; ix < ngridx; ix += WARPS) {
                int tx = (ix + nx0) % mesh_x;
                double r2[nf2];
                double r1[L+1];
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    r2[m] = 0.;
                }
                for (int iy = 0; iy < ny; ++iy) {
                    int ty = (iy + iy0 + ny0) % mesh_y;
#pragma unroll
                    for (int mz = 0; mz <= L; ++mz) {
                        r1[mz] = 0.;
                    }
                    for (int iz = 0; iz < nz; ++iz) {
                        int tz = (iz + iz0 + nz0) % mesh_z;
                        double r = vx[tx*mesh_yz+ty*mesh_z+tz];
#pragma unroll
                        for (int mz = 0; mz <= L; ++mz) {
                            r1[mz] += zs_cache[(mz*TILE+iz)*WARP_SIZE] * r;
                        }
                    }
#pragma unroll
                    for (int my = 0; my <= L; ++my) {
                        double fac = ys_cache[(my*TILE+iy)*WARP_SIZE];
#pragma unroll
                        for (int mz = 0; mz <= L-my; ++mz) {
                            r2[ADDR2(L,my,mz)] += fac * r1[mz];
                        }
                    }
                }
                double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgx[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz_ip1<L>(dm_xyz, gx_dmyz, xs_exp, ngridx, ngrid_span);
    _dm_xyz_to_dm_sigmax<L>(out, dm_xyz, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
    __syncthreads();

    // sigmay: contract mesh_x, mesh_z
    double *gy_dmxz = gx_dmyz;
    xs_cache = cache + sp_id;
    zs_cache = xs_cache + TILE * (L+1) * WARP_SIZE;
    for (int n = warp_id; n < ngridy*nf2; n += WARPS) {
        gy_dmxz[n*WARP_SIZE] = 0.;
    }
    for (int iz0 = 0; iz0 < ngridz; iz0 += TILE) {
        __syncthreads();
        int nz = load_xs(zs_cache, zs_exp, iz0, ngridz, L, TILE, ngrid_span, warp_id);
        for (int ix0 = 0; ix0 < ngridx; ix0 += TILE) {
            __syncthreads();
            int nx = load_xs(xs_cache, xs_exp, ix0, ngridx, L, TILE, ngrid_span, warp_id);
            for (int iy = warp_id; iy < ngridy; iy += WARPS) {
                int ty = (iy + ny0) % mesh_y;
                double r2[nf2];
                double r1[L+1];
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    r2[m] = 0.;
                }
                for (int ix = 0; ix < nx; ++ix) {
                    int tx = (ix + ix0 + nx0) % mesh_x;
#pragma unroll
                    for (int mz = 0; mz <= L; ++mz) {
                        r1[mz] = 0.;
                    }
                    for (int iz = 0; iz < nz; ++iz) {
                        int tz = (iz + iz0 + nz0) % mesh_z;
                        double r = vy[tx*mesh_yz+ty*mesh_z+tz];
#pragma unroll
                        for (int mz = 0; mz <= L; ++mz) {
                            r1[mz] += zs_cache[(mz*TILE+iz)*WARP_SIZE] * r;
                        }
                    }
#pragma unroll
                    for (int mx = 0; mx <= L; ++mx) {
                        double fac = xs_cache[(mx*TILE+ix)*WARP_SIZE];
#pragma unroll
                        for (int mz = 0; mz <= L-mx; ++mz) {
                            r2[ADDR2(L,mx,mz)] += fac * r1[mz];
                        }
                    }
                }
                double *pgy = gy_dmxz + iy * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgy[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz_ip1<L>(dm_xyz, gy_dmxz, ys_exp, ngridy, ngrid_span);
    _dm_xyz_to_dm_sigmay<L>(out, dm_xyz, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
    __syncthreads();

    // sigmaz: contract mesh_x, mesh_y
    double *gz_dmxy = gx_dmyz;
    xs_cache = cache + sp_id;
    ys_cache = xs_cache + TILE * (L+1) * WARP_SIZE;
    for (int n = warp_id; n < ngridz*nf2; n += WARPS) {
        gz_dmxy[n*WARP_SIZE] = 0.;
    }
    for (int ix0 = 0; ix0 < ngridx; ix0 += TILE) {
        __syncthreads();
        int nx = load_xs(xs_cache, xs_exp, ix0, ngridx, L, TILE, ngrid_span, warp_id);
        for (int iy0 = 0; iy0 < ngridy; iy0 += TILE) {
            __syncthreads();
            int ny = load_xs(ys_cache, ys_exp, iy0, ngridy, L, TILE, ngrid_span, warp_id);
            for (int iz = warp_id; iz < ngridz; iz += WARPS) {
                int tz = (iz + nz0) % mesh_z;
                double r2[nf2];
                double r1[L+1];
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    r2[m] = 0.;
                }
                for (int ix = 0; ix < nx; ++ix) {
                    int tx = (ix + ix0 + nx0) % mesh_x;
#pragma unroll
                    for (int my = 0; my <= L; ++my) {
                        r1[my] = 0.;
                    }
                    for (int iy = 0; iy < ny; ++iy) {
                        int ty = (iy + iy0 + ny0) % mesh_y;
                        double r = vz[tx*mesh_yz+ty*mesh_z+tz];
#pragma unroll
                        for (int my = 0; my <= L; ++my) {
                            r1[my] += ys_cache[(my*TILE+iy)*WARP_SIZE] * r;
                        }
                    }
#pragma unroll
                    for (int mx = 0; mx <= L; ++mx) {
                        double fac = xs_cache[(mx*TILE+ix)*WARP_SIZE];
#pragma unroll
                        for (int my = 0; my <= L-mx; ++my) {
                            r2[ADDR2(L,mx,my)] += fac * r1[my];
                        }
                    }
                }
                double *pgz = gz_dmxy + iz * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgz[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz_ip1<L>(dm_xyz, gz_dmxy, zs_exp, ngridz, ngrid_span);
    _dm_xyz_to_dm_sigmaz<L>(out, dm_xyz, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
}

template <int L, int TILE> __global__
void eval_mat_gga_kernel(double *out, double *rho, MGridEnvVars envs,
                         MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+2) * ngrid_span;
    int nf2 = (L+1)*(L+2)/2;
    int l3 = nf2*(L+2);
    pool += (xs_size*3 + nf2*ngrid_span + 3 + l3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_mat_gga_kernel<L, TILE>(out, rho, envs, bounds, pool, pair_idx0);
        if (thread_id == 0) {
            pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
        }
        __syncthreads();
    }
}

static size_t buflen_gga(int l, int tile)
{
    int lj = MIN(l, LMAX);
    size_t len1 = WARP_SIZE * tile * (l+1) * 2;
    size_t len2 = WARP_SIZE * (lj+2)*(lj+2) * 3;
    if (l < 4) {
        len2 += WARP_SIZE * (l+1)*(l+2)/2*(l+2);
    }
    return MAX(len1, len2) * sizeof(double);
}

extern "C" {
int MG_eval_mat_gga_orth(double *out, double *rho, MGridEnvVars envs,
                         int l, int n_radius, int *mesh, int nshl_pair,
                         int *bas_ij_idx, double *pool, int workers)
{
    MGridBounds bounds = {
        nshl_pair, bas_ij_idx, n_radius, {mesh[0], mesh[1], mesh[2]},
    };
    uint32_t *batch_head;
    cudaMalloc(reinterpret_cast<void **>(&batch_head), sizeof(uint32_t) * 1);
    cudaMemset(batch_head, 0, sizeof(uint32_t));

    switch (l) {
    case 0: eval_mat_gga_kernel<0,32> <<<workers, THREADS, buflen_gga(0,32)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 1: eval_mat_gga_kernel<1,32> <<<workers, THREADS, buflen_gga(1,32)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 2: eval_mat_gga_kernel<2,16> <<<workers, THREADS, buflen_gga(2,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 3: eval_mat_gga_kernel<3,16> <<<workers, THREADS, buflen_gga(3,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 4: eval_mat_gga_kernel<4,16> <<<workers, THREADS, buflen_gga(4,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 5: eval_mat_gga_kernel<5, 8> <<<workers, THREADS, buflen_gga(5, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 6: eval_mat_gga_kernel<6, 8> <<<workers, THREADS, buflen_gga(6, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 7: eval_mat_gga_kernel<7, 8> <<<workers, THREADS, buflen_gga(7, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 8: eval_mat_gga_kernel<8, 8> <<<workers, THREADS, buflen_gga(8, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    default: 
        fprintf(stderr, "MG_eval_mat_gga_orth does not support l>8\n");
        cudaFree(batch_head);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_mat_gga_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}
}
