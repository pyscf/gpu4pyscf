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
void fill_gx_dmyz(double *gx_dmyz, double *dm_xyz, double *xs_exp,
                  int ngridx, int ngrid_span, int npairs_this_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    constexpr int L2 = L + 2;
    constexpr int nf2 = (L+1)*(L+2)/2;
    constexpr int nf3 = nf2*(L+3)/3;
    int xs_stride = ngrid_span * WARP_SIZE;
    double r1[L+3];
    extern __shared__ double cache[];
    double *dm_cache = cache + sp_id;
    double *gx_local = gx_dmyz + sp_id * nf2*ngridx;
    dm_xyz += sp_id;
    xs_exp += sp_id;
    if (L <= 6) {
        for (int n = warp_id; n < nf3+nf2*2; n += WARPS) {
            dm_cache[n*WARP_SIZE] = dm_xyz[n*WARP_SIZE];
        }
        __syncthreads();
        if (sp_id >= npairs_this_block) {
            return;
        }
        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
            if (ix < ngridx) {
                double *xs_local = xs_exp + ix * WARP_SIZE;
#pragma unroll
                for (int m = 0; m <= L2; ++m) {
                    r1[m] = xs_local[m*xs_stride];
                }
#pragma unroll
                for (int n = 0, my = 0; my <= L; ++my) {
#pragma unroll
                    for (int mz = 0; mz <= L-my; ++mz) {
                        double val = 0.;
#pragma unroll
                        for (int mx = 0; mx <= L2-my-mz; ++mx, ++n) {
                            val += r1[mx] * dm_cache[n*WARP_SIZE];
                        }
                        gx_local[ix*nf2+ADDR2(L,my,mz)] = val;
                    }
                }
            }
        }
    } else {
        if (sp_id >= npairs_this_block) {
            return;
        }
        for (int ix = warp_id; ix < ngridx; ix += WARPS) {
            if (ix < ngridx) {
                double *xs_local = xs_exp + ix * WARP_SIZE;
#pragma unroll
                for (int m = 0; m <= L2; ++m) {
                    r1[m] = xs_local[m*xs_stride];
                }
#pragma unroll
                for (int n = 0, my = 0; my <= L; ++my) {
#pragma unroll
                    for (int mz = 0; mz <= L-my; ++mz) {
                        double val = 0.;
#pragma unroll
                        for (int mx = 0; mx <= L2-my-mz; ++mx, ++n) {
                            val += r1[mx] * dm_xyz[n*WARP_SIZE];
                        }
                        gx_local[ix*nf2+ADDR2(L,my,mz)] = val;
                    }
                }
            }
        }
    }
}

//template <int L> __device__ inline
//double sub_dm_to_dm_xyz(int lx, int ly, int lz, int li, int lj, int nao,
//                        double ai2, double aj2,
//                        double *cx, double *cy, double *cz, double *dm)
//{
//    constexpr int L3 = L + 3;
//    int lj3 = lj + 3;
//    double out = 0.;
//    // -2*ai xi, -2*aj xj
//    for (int lx_i = MIN(lx-1, li); lx_i >= 0; --lx_i) {
//    for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
//        int lz_i = li - lx_i - ly_i;
//        if (lz < lz_i) continue;
//        int jx = lx - (lx_i+1);
//        int jy = ly - ly_i;
//        int jz = lz - lz_i;
//        int i = cart_address(li, lx_i, ly_i, lz_i);
//        for (int lx_j = lj; lx_j >= MAX(jx-1, 0); --lx_j) {
//        for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
//            int lz_j = lj - lx_j - ly_j;
//            if (lz_j < jz) continue;
//            int j = cart_address(lj, lx_j, ly_j, lz_j);
//            double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
//            double cxyz = ai2 * aj2 * cyz * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
//            out += cxyz * dm[i*nao+j];
//        } }
//    } }
//    // -2*ai xi, lj/xj
//    for (int lx_i = MIN(lx-1, li); lx_i >= 0; --lx_i) {
//    for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
//        int lz_i = li - lx_i - ly_i;
//        if (lz < lz_i) continue;
//        int jx = lx - (lx_i+1);
//        int jy = ly - ly_i;
//        int jz = lz - lz_i;
//        int i = cart_address(li, lx_i, ly_i, lz_i);
//        for (int lx_j = lj; lx_j-1 >= jx; --lx_j) {
//        for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
//            int lz_j = lj - lx_j - ly_j;
//            if (lz_j < jz) continue;
//            int j = cart_address(lj, lx_j, ly_j, lz_j);
//            double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
//            double cxyz = ai2 * lx_j * cyz * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
//            out += cxyz * dm[i*nao+j];
//        } }
//    } }
//    // li/xi, -2*aj xj
//    for (int lx_i = MIN(lx+1, li); lx_i >= 1; --lx_i) {
//    for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
//        int lz_i = li - lx_i - ly_i;
//        if (lz < lz_i) continue;
//        int jx = lx - (lx_i-1);
//        int jy = ly - ly_i;
//        int jz = lz - lz_i;
//        int i = cart_address(li, lx_i, ly_i, lz_i);
//        for (int lx_j = lj; lx_j >= MIN(jx-1, 0); --lx_j) {
//        for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
//            int lz_j = lj - lx_j - ly_j;
//            if (lz_j < jz) continue;
//            int j = cart_address(lj, lx_j, ly_j, lz_j);
//            double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
//            double cxyz = lx_i * aj2 * cyz * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
//            out += cxyz * dm[i*nao+j];
//        } }
//    } }
//    // li/xi, lj/xj
//    for (int lx_i = MIN(lx+1, li); lx_i >= 1; --lx_i) {
//    for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
//        int lz_i = li - lx_i - ly_i;
//        if (lz < lz_i) continue;
//        int jx = lx - (lx_i-1);
//        int jy = ly - ly_i;
//        int jz = lz - lz_i;
//        int i = cart_address(li, lx_i, ly_i, lz_i);
//        for (int lx_j = lj; lx_j-1 >= jx; --lx_j) {
//        for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
//            int lz_j = lj - lx_j - ly_j;
//            if (lz_j < jz) continue;
//            int j = cart_address(lj, lx_j, ly_j, lz_j);
//            double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
//            double cxyz = lx_i * lx_j * cyz * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
//            out += cxyz * dm[i*nao+j];
//        } }
//    } }
//    return out;
//}

template <int L> __device__ static
void _dm_to_dm_xyz_derivx(double *dm_xyz, double *dm, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    constexpr int L2 = L + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj3 * lj3 * WARP_SIZE;
    double *cz = cy + lj3 * lj3 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj3*lj3*WARP_SIZE, ri[n], rj[n], lj2);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }
    if (warp_id != 0) {
        return;
    }

    dm_xyz += sp_id;
    for (int n = 0, ly = 0; ly <= L; ++ly) {
    for (int lz = 0; lz <= L-ly; ++lz) {
    for (int lx = 0; lx <= L2-ly-lz; ++lx, ++n) {
#if 0
        double out = sub_dm_to_dm_xyz<L>(lx, ly, lz, li, lj, nao,
                                         ai2, aj2, cx, cy, cz, dm);
#else
        double out = 0.;
        // -2*ai xi, -2*aj xj
        for (int lx_i = MIN(lx-1, li); lx_i >= 0; --lx_i) {
        for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
            int lz_i = li - lx_i - ly_i;
            if (lz < lz_i) continue;
            int jx = lx - (lx_i+1);
            int jy = ly - ly_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lx_j = lj; lx_j >= MAX(jx-1, 0); --lx_j) {
            for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
                int lz_j = lj - lx_j - ly_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * aj2 * cyz * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // -2*ai xi, lj/xj
        for (int lx_i = MIN(lx-1, li); lx_i >= 0; --lx_i) {
        for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
            int lz_i = li - lx_i - ly_i;
            if (lz < lz_i) continue;
            int jx = lx - (lx_i+1);
            int jy = ly - ly_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lx_j = lj; lx_j-1 >= jx; --lx_j) {
            for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
                int lz_j = lj - lx_j - ly_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * lx_j * cyz * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/xi, -2*aj xj
        for (int lx_i = MIN(lx+1, li); lx_i >= 1; --lx_i) {
        for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
            int lz_i = li - lx_i - ly_i;
            if (lz < lz_i) continue;
            int jx = lx - (lx_i-1);
            int jy = ly - ly_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lx_j = lj; lx_j >= MAX(jx-1, 0); --lx_j) {
            for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
                int lz_j = lj - lx_j - ly_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = lx_i * aj2 * cyz * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/xi, lj/xj
        for (int lx_i = MIN(lx+1, li); lx_i >= 1; --lx_i) {
        for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; --ly_i) {
            int lz_i = li - lx_i - ly_i;
            if (lz < lz_i) continue;
            int jx = lx - (lx_i-1);
            int jy = ly - ly_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lx_j = lj; lx_j-1 >= jx; --lx_j) {
            for (int ly_j = lj-lx_j; ly_j >= jy; --ly_j) {
                int lz_j = lj - lx_j - ly_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cyz = cy[(jy+ly_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = lx_i * lx_j * cyz * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
#endif
            dm_xyz[n*WARP_SIZE] = out * cicj;
    } } }
}

template <int L> __device__ static
void _dm_to_dm_xyz_derivy(double *dm_xyz, double *dm, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    constexpr int L2 = L + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj3 * lj3 * WARP_SIZE;
    double *cz = cy + lj3 * lj3 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj3*lj3*WARP_SIZE, ri[n], rj[n], lj2);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }
    if (warp_id != 0) {
        return;
    }

    dm_xyz += sp_id;
    for (int n = 0, lx = 0; lx <= L; ++lx) {
    for (int lz = 0; lz <= L-lx; ++lz) {
    for (int ly = 0; ly <= L2-lx-lz; ++ly, ++n) {
#if 0
        double out = sub_dm_to_dm_xyz<L>(ly, lx, lz, li, lj, nao,
                                         ai2, aj2, cy, cz, cx, dm);
#else
        double out = 0.;
        // -2*ai yi, -2*aj yj
        for (int ly_i = MIN(ly-1, li); ly_i >= 0; --ly_i) {
        for (int lx_i = MIN(lx, li-ly_i); lx_i >= 0; --lx_i) {
            int lz_i = li - ly_i - lx_i;
            if (lz < lz_i) continue;
            int jy = ly - (ly_i+1);
            int jx = lx - lx_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int ly_j = lj; ly_j >= MAX(jy-1, 0); --ly_j) {
            for (int lx_j = lj-ly_j; lx_j >= jx; --lx_j) {
                int lz_j = lj - ly_j - lx_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxz = cx[(jx+lx_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * aj2 * cxz * cy[(jy+(ly_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // -2*ai yi, lj/yj
        for (int ly_i = MIN(ly-1, li); ly_i >= 0; --ly_i) {
        for (int lx_i = MIN(lx, li-ly_i); lx_i >= 0; --lx_i) {
            int lz_i = li - ly_i - lx_i;
            if (lz < lz_i) continue;
            int jy = ly - (ly_i+1);
            int jx = lx - lx_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int ly_j = lj; ly_j-1 >= jy; --ly_j) {
            for (int lx_j = lj-ly_j; lx_j >= jx; --lx_j) {
                int lz_j = lj - ly_j - lx_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxz = cx[(jx+lx_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * ly_j * cxz * cy[(jy+(ly_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/yi, -2*aj yj
        for (int ly_i = MIN(ly+1, li); ly_i >= 1; --ly_i) {
        for (int lx_i = MIN(lx, li-ly_i); lx_i >= 0; --lx_i) {
            int lz_i = li - ly_i - lx_i;
            if (lz < lz_i) continue;
            int jy = ly - (ly_i-1);
            int jx = lx - lx_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int ly_j = lj; ly_j >= MAX(jy-1, 0); --ly_j) {
            for (int lx_j = lj-ly_j; lx_j >= jx; --lx_j) {
                int lz_j = lj - ly_j - lx_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxz = cx[(jx+lx_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ly_i * aj2 * cxz * cy[(jy+(ly_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/yi, lj/yj
        for (int ly_i = MIN(ly+1, li); ly_i >= 1; --ly_i) {
        for (int lx_i = MIN(lx, li-ly_i); lx_i >= 0; --lx_i) {
            int lz_i = li - ly_i - lx_i;
            if (lz < lz_i) continue;
            int jy = ly - (ly_i-1);
            int jx = lx - lx_i;
            int jz = lz - lz_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int ly_j = lj; ly_j-1 >= jy; --ly_j) {
            for (int lx_j = lj-ly_j; lx_j >= jx; --lx_j) {
                int lz_j = lj - ly_j - lx_j;
                if (lz_j < jz) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxz = cx[(jx+lx_j*lj3)*WARP_SIZE] * cz[(jz+lz_j*lj3)*WARP_SIZE];
                double cxyz = ly_i * ly_j * cxz * cy[(jy+(ly_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
#endif
        dm_xyz[n*WARP_SIZE] = out * cicj;
    } } }
}

template <int L> __device__ static
void _dm_to_dm_xyz_derivz(double *dm_xyz, double *dm, int nao, int li, int lj,
                          double *ri, double *rj, double ai2, double aj2,
                          double cicj, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    constexpr int L2 = L + 2;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj3 * lj3 * WARP_SIZE;
    double *cz = cy + lj3 * lj3 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj3*lj3*WARP_SIZE, ri[n], rj[n], lj2);
    }
    __syncthreads();

    if (sp_id >= npairs_per_block) {
        return;
    }
    if (warp_id != 0) {
        return;
    }

    dm_xyz += sp_id;
    for (int n = 0, lx = 0; lx <= L; ++lx) {
    for (int ly = 0; ly <= L-lx; ++ly) {
    for (int lz = 0; lz <= L2-lx-ly; ++lz, ++n) {
#if 0
        double out = sub_dm_to_dm_xyz<L>(lz, lx, ly, li, lj, nao,
                                         ai2, aj2, cz, cx, cy, dm);
#else
        double out = 0.;
        // -2*ai zi, -2*aj zj
        for (int lz_i = MIN(lz-1, li); lz_i >= 0; --lz_i) {
        for (int lx_i = MIN(lx, li-lz_i); lx_i >= 0; --lx_i) {
            int ly_i = li - lz_i - lx_i;
            if (ly < ly_i) continue;
            int jz = lz - (lz_i+1);
            int jx = lx - lx_i;
            int jy = ly - ly_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lz_j = lj; lz_j >= MAX(jz-1, 0); --lz_j) {
            for (int lx_j = lj-lz_j; lx_j >= jx; --lx_j) {
                int ly_j = lj - lz_j - lx_j;
                if (ly_j < jy) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxy = cx[(jx+lx_j*lj3)*WARP_SIZE] * cy[(jy+ly_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * aj2 * cxy * cz[(jz+(lz_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // -2*ai zi, lj/zj
        for (int lz_i = MIN(lz-1, li); lz_i >= 0; --lz_i) {
        for (int lx_i = MIN(lx, li-lz_i); lx_i >= 0; --lx_i) {
            int ly_i = li - lz_i - lx_i;
            if (ly < ly_i) continue;
            int jz = lz - (lz_i+1);
            int jx = lx - lx_i;
            int jy = ly - ly_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lz_j = lj; lz_j-1 >= jz; --lz_j) {
            for (int lx_j = lj-lz_j; lx_j >= jx; --lx_j) {
                int ly_j = lj - lz_j - lx_j;
                if (ly_j < jy) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxy = cx[(jx+lx_j*lj3)*WARP_SIZE] * cy[(jy+ly_j*lj3)*WARP_SIZE];
                double cxyz = ai2 * lz_j * cxy * cz[(jz+(lz_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/zi, -2*aj zj
        for (int lz_i = MIN(lz+1, li); lz_i >= 1; --lz_i) {
        for (int lx_i = MIN(lx, li-lz_i); lx_i >= 0; --lx_i) {
            int ly_i = li - lz_i - lx_i;
            if (ly < ly_i) continue;
            int jz = lz - (lz_i-1);
            int jx = lx - lx_i;
            int jy = ly - ly_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lz_j = lj; lz_j >= MAX(jz-1, 0); --lz_j) {
            for (int lx_j = lj-lz_j; lx_j >= jx; --lx_j) {
                int ly_j = lj - lz_j - lx_j;
                if (ly_j < jy) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxy = cx[(jx+lx_j*lj3)*WARP_SIZE] * cy[(jy+ly_j*lj3)*WARP_SIZE];
                double cxyz = lz_i * aj2 * cxy * cz[(jz+(lz_j+1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
        // li/zi, lj/zj
        for (int lz_i = MIN(lz+1, li); lz_i >= 1; --lz_i) {
        for (int lx_i = MIN(lx, li-lz_i); lx_i >= 0; --lx_i) {
            int ly_i = li - lz_i - lx_i;
            if (ly < ly_i) continue;
            int jz = lz - (lz_i-1);
            int jx = lx - lx_i;
            int jy = ly - ly_i;
            int i = cart_address(li, lx_i, ly_i, lz_i);
            for (int lz_j = lj; lz_j-1 >= jz; --lz_j) {
            for (int lx_j = lj-lz_j; lx_j >= jx; --lx_j) {
                int ly_j = lj - lz_j - lx_j;
                if (ly_j < jy) continue;
                int j = cart_address(lj, lx_j, ly_j, lz_j);
                double cxy = cx[(jx+lx_j*lj3)*WARP_SIZE] * cy[(jy+ly_j*lj3)*WARP_SIZE];
                double cxyz = lz_i * lz_j * cxy * cz[(jz+(lz_j-1)*lj3)*WARP_SIZE];
                out += cxyz * dm[i*nao+j];
            } }
        } }
#endif
        dm_xyz[n*WARP_SIZE] = out * cicj;
    } } }
}

template <int L> __device__ static
void _eval_tau_orth_kernel(double *rho, double *dm, MGridEnvVars envs,
                           MGridBounds bounds, double *pool, uint32_t pair_idx0)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
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
    double cicj = .5 * ci * cj * exp(-theta_ij * rr_ij);
    if (sp_id >= npairs_this_block) {
        cicj = 0.;
    }
    double ai2 = -2. * ai;
    double aj2 = -2. * aj;
    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];

    constexpr int L3 = L + 3;
    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = L3 * ngrid_span * WARP_SIZE;
    constexpr int nf2 = (L+1)*(L+2)/2;
    int *grid_start = (int *)pool;
    double *xs_exp = pool + WARP_SIZE*3;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *dm_xyz = zs_exp + xs_size;
    double *gx_dmyz = dm_xyz + nf2*L3 * WARP_SIZE;
    init_orth_data(xs_exp+sp_id, grid_start+sp_id, envs, bounds, ri, rj, ai, aj, L+2);

    double r1[L+1];
    double r2[L+1];
    extern __shared__ double cache[];

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

    // dx * dx
    _dm_to_dm_xyz_derivx<L>(dm_xyz, dm+i0*nao+j0, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
    __syncthreads();
    fill_gx_dmyz<L>(gx_dmyz, dm_xyz, xs_exp, ngridx, ngrid_span, npairs_this_block);
    int ngridxz = ngridx * ngridz;
    int iy_stride = 1;
    if (ngridxz * 2 < THREADS) {
        iy_stride = THREADS / ngridxz;
        iy_stride = MIN(iy_stride, ngridy);
    }
    int y_ngridxz = iy_stride * ngridxz;
    for (int sp_id = 0; sp_id < npairs_this_block; ++sp_id) {
        int nx0 = grid_start[0*WARP_SIZE+sp_id];
        int ny0 = grid_start[1*WARP_SIZE+sp_id];
        int nz0 = grid_start[2*WARP_SIZE+sp_id];
        double *ys_cache = cache;
        double *zs_cache = ys_cache + (L+1)*ngridy;
        double *xs_cache = zs_cache + (L+1)*ngridz;
        __syncthreads();
        for (int n = thread_id; n < ngridy*(L+1); n += THREADS) {
            int m = n / ngridy;
            int iy = n % ngridy;
            ys_cache[n] = ys_exp[(m*ngrid_span+iy)*WARP_SIZE+sp_id];
        }
        for (int n = thread_id; n < ngridz*(L+1); n += THREADS) {
            int m = n / ngridz;
            int iz = n % ngridz;
            zs_cache[n] = zs_exp[(m*ngrid_span+iz)*WARP_SIZE+sp_id];
        }
        double *pgx = gx_dmyz + sp_id * nf2 * ngridx;
        for (int n = thread_id; n < ngridx*nf2; n += THREADS) {
            xs_cache[n] = pgx[n];
        }
        __syncthreads();

        for (int xyz = thread_id; xyz < y_ngridxz; xyz += THREADS) {
            int iy_inc = xyz / ngridxz;
            if (iy_inc >= ngridy) {
                continue;
            }

            int ixz = xyz % ngridxz;
            int ix = ixz / ngridz;
            int iz = ixz % ngridz;
#pragma unroll
            for (int mz = 0; mz <= L; ++mz) {
                r1[mz] = zs_cache[mz*ngridz+iz];
            }
#pragma unroll
            for (int my = 0; my <= L; ++my) {
                r2[my] = 0.;
#pragma unroll
                for (int mz = 0; mz <= L-my; ++mz) {
                    r2[my] += xs_cache[ix*nf2+ADDR2(L,my,mz)] * r1[mz];
                }
            }
            int tx = (ix + nx0) % mesh_x;
            int tz = (iz + nz0) % mesh_z;
            double *rho_local = rho + tx * mesh_yz + tz;
            for (int iy = iy_inc; iy < ngridy; iy += iy_stride) {
                int ty = (iy + ny0) % mesh_y;
                double val = 0.;
#pragma unroll
                for (int my = 0; my <= L; ++my) {
                    val += ys_cache[my*ngridy+iy] * r2[my];
                }
                atomicAdd(rho_local+ty*mesh_z, val);
            }
        }
    }
    __syncthreads();

    // dy * dy
    double *gy_dmxz = gx_dmyz;
    _dm_to_dm_xyz_derivy<L>(dm_xyz, dm+i0*nao+j0, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
    __syncthreads();
    fill_gx_dmyz<L>(gy_dmxz, dm_xyz, ys_exp, ngridy, ngrid_span, npairs_this_block);
    int ngridyz = ngridy * ngridz;
    int ix_stride = 1;
    if (ngridyz * 2 < THREADS) {
        ix_stride = THREADS / ngridyz;
        ix_stride = MIN(ix_stride, ngridx);
    }
    int x_ngridyz = ix_stride * ngridyz;
    for (int sp_id = 0; sp_id < npairs_this_block; ++sp_id) {
        int nx0 = grid_start[0*WARP_SIZE+sp_id];
        int ny0 = grid_start[1*WARP_SIZE+sp_id];
        int nz0 = grid_start[2*WARP_SIZE+sp_id];
        double *xs_cache = cache;
        double *zs_cache = xs_cache + (L+1)*ngridx;
        double *ys_cache = zs_cache + (L+1)*ngridz;
        __syncthreads();
        for (int n = thread_id; n < ngridx*(L+1); n += THREADS) {
            int m = n / ngridx;
            int ix = n % ngridx;
            xs_cache[n] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE+sp_id];
        }
        for (int n = thread_id; n < ngridz*(L+1); n += THREADS) {
            int m = n / ngridz;
            int iz = n % ngridz;
            zs_cache[n] = zs_exp[(m*ngrid_span+iz)*WARP_SIZE+sp_id];
        }
        double *pgy = gy_dmxz + sp_id * nf2 * ngridy;
        for (int n = thread_id; n < ngridy*nf2; n += THREADS) {
            ys_cache[n] = pgy[n];
        }
        __syncthreads();

        for (int xyz = thread_id; xyz < x_ngridyz; xyz += THREADS) {
            int ix_inc = xyz / ngridyz;
            if (ix_inc >= ngridx) {
                continue;
            }

            int iyz = xyz % ngridyz;
            int iy = iyz / ngridz;
            int iz = iyz % ngridz;
#pragma unroll
            for (int mz = 0; mz <= L; ++mz) {
                r1[mz] = zs_cache[mz*ngridz+iz];
            }
#pragma unroll
            for (int mx = 0; mx <= L; ++mx) {
                r2[mx] = 0.;
#pragma unroll
                for (int mz = 0; mz <= L-mx; ++mz) {
                    r2[mx] += ys_cache[iy*nf2+ADDR2(L,mx,mz)] * r1[mz];
                }
            }
            int ty = (iy + ny0) % mesh_y;
            int tz = (iz + nz0) % mesh_z;
            int addr_yz = ty * mesh_z + tz;
            double *rho_local = rho + addr_yz;
            for (int ix = ix_inc; ix < ngridx; ix += ix_stride) {
                int tx = (ix + nx0) % mesh_x;
                double val = 0.;
#pragma unroll
                for (int mx = 0; mx <= L; ++mx) {
                    val += xs_cache[mx*ngridx+ix] * r2[mx];
                }
                atomicAdd(rho_local+tx*mesh_yz, val);
            }
        }
    }
    __syncthreads();

    // dz * dz
    double *gz_dmxy = gx_dmyz;
    _dm_to_dm_xyz_derivz<L>(dm_xyz, dm+i0*nao+j0, nao, li, lj, ri, rj, ai2, aj2,
                            cicj, npairs_this_block);
    __syncthreads();
    fill_gx_dmyz<L>(gz_dmxy, dm_xyz, zs_exp, ngridz, ngrid_span, npairs_this_block);
    for (int sp_id = 0; sp_id < npairs_this_block; ++sp_id) {
        int nx0 = grid_start[0*WARP_SIZE+sp_id];
        int ny0 = grid_start[1*WARP_SIZE+sp_id];
        int nz0 = grid_start[2*WARP_SIZE+sp_id];
        double *xs_cache = cache;
        double *ys_cache = xs_cache + (L+1)*ngridx;
        double *zs_cache = ys_cache + (L+1)*ngridy;
        __syncthreads();
        for (int n = thread_id; n < ngridx*(L+1); n += THREADS) {
            int m = n / ngridx;
            int ix = n % ngridx;
            xs_cache[n] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE+sp_id];
        }
        for (int n = thread_id; n < ngridy*(L+1); n += THREADS) {
            int m = n / ngridy;
            int iy = n % ngridy;
            ys_cache[n] = ys_exp[(m*ngrid_span+iy)*WARP_SIZE+sp_id];
        }
        double *pgz = gz_dmxy + sp_id * nf2 * ngridz;
        for (int n = thread_id; n < ngridz*nf2; n += THREADS) {
            zs_cache[n] = pgz[n];
        }
        __syncthreads();

        for (int xyz = thread_id; xyz < x_ngridyz; xyz += THREADS) {
            int ix_inc = xyz / ngridyz;
            if (ix_inc >= ngridx) {
                continue;
            }

            int iyz = xyz % ngridyz;
            int iy = iyz / ngridz;
            int iz = iyz % ngridz;
#pragma unroll
            for (int my = 0; my <= L; ++my) {
                r1[my] = ys_cache[my*ngridy+iy];
            }
#pragma unroll
            for (int mx = 0; mx <= L; ++mx) {
                r2[mx] = 0.;
#pragma unroll
                for (int my = 0; my <= L-mx; ++my) {
                    r2[mx] += zs_cache[iz*nf2+ADDR2(L,mx,my)] * r1[my];
                }
            }
            int ty = (iy + ny0) % mesh_y;
            int tz = (iz + nz0) % mesh_z;
            double *rho_local = rho + ty * mesh_z + tz;
            for (int ix = ix_inc; ix < ngridx; ix += ix_stride) {
                int tx = (ix + nx0) % mesh_x;
                double val = 0.;
#pragma unroll
                for (int mx = 0; mx <= L; ++mx) {
                    val += xs_cache[mx*ngridx+ix] * r2[mx];
                }
                atomicAdd(rho_local+tx*mesh_yz, val);
            }
        }
    }
}

template <int L> __global__
void eval_tau_orth_kernel(double *rho, double *dm, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+3) * ngrid_span;
    int nf2 = (L+1)*(L+2)/2;
    int nf3 = nf2 * (L+3);
    pool += (xs_size*3 + nf3 + nf2*ngrid_span + 3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_tau_orth_kernel<L>(rho, dm, envs, bounds, pool, pair_idx0);
        if (thread_id == 0) {
            pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
        }
        __syncthreads();
    }
}

static size_t buflen(int l, MGridBounds *bounds)
{
    int ngrid_span = bounds->ngrid_radius * 2;
    int lj = MIN(l, LMAX);
    int nf2 = (l+1)*(l+2)/2;
    int nf3 = nf2*(l+3)/3;
    size_t len1 = (nf3+nf2*2) * WARP_SIZE; 
    size_t len2 = (lj+3)*(lj+3) * 3 * WARP_SIZE;
    size_t len3 = (l+1) * ngrid_span * 2 + nf2 * ngrid_span;
    len2 = MAX(len2, len3);
    if (l <= 6) {
        len2 = MAX(len1, len2);
    }
    return len2 * sizeof(double);
}

extern "C" {
int MG_eval_tau_orth(double *rho, double *dm, MGridEnvVars envs,
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
    case 0: eval_tau_orth_kernel<0> <<<workers, THREADS, buflen(0, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 1: eval_tau_orth_kernel<1> <<<workers, THREADS, buflen(1, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 2: eval_tau_orth_kernel<2> <<<workers, THREADS, buflen(2, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 3: eval_tau_orth_kernel<3> <<<workers, THREADS, buflen(3, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 4: eval_tau_orth_kernel<4> <<<workers, THREADS, buflen(4, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 5: eval_tau_orth_kernel<5> <<<workers, THREADS, buflen(5, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 6: eval_tau_orth_kernel<6> <<<workers, THREADS, buflen(6, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 7: eval_tau_orth_kernel<7> <<<workers, THREADS, buflen(7, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 8: eval_tau_orth_kernel<8> <<<workers, THREADS, buflen(8, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    default: return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_tau_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}
}
