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

#include <stdint.h>
#include "multigrid.cuh"

__device__ inline
void dm_xyz_coeff(double *cx, double xi, double xj, int lmax)
{
    double xij = xi - xj;
    int lmax1 = lmax + 1;
    cx[0] = 1.;
    for (int lx = 1; lx <= lmax; lx++) {
        cx[lx*WARP_SIZE] = cx[(lx-1)*WARP_SIZE] * xij;
    }
    for (int l = 1; l <= lmax; l++){
        double binom = 1.;
        for (int lx = 0; lx <= l; lx++) {
            // binom = binomial(l, lx)
            cx[(l*lmax1+lx)*WARP_SIZE] = binom * cx[(l-lx)*WARP_SIZE];
            binom = (binom * (l-lx)) / (lx+1);
        }
    }
}

__device__ inline
int cart_address(int l, int x, int y, int z)
{
    // (l-x)*(l-x+1)/2+l-x-y
    int yz = l - x;
    return yz * (yz + 3) / 2 - y;
}

__device__ inline
double sub_dm_xyz(int lx, int ly, int lz, int li, int lj, int nao,
                  double *cx, double *cy, double *cz, double *dm)
{
    // TODO: unroll lij < 4
    int lj1 = lj + 1;
    double out = 0.;
    for (int lx_i = MIN(lx, li); lx_i >= 0; lx_i--) {
    for (int ly_i = MIN(ly, li-lx_i); ly_i >= 0; ly_i--) {
        int lz_i = li - lx_i - ly_i;
        if (lz < lz_i) continue;
        int jx = lx - lx_i;
        int jy = ly - ly_i;
        int jz = lz - lz_i;
        int i = cart_address(li, lx_i, ly_i, lz_i);
        // TODO: precomputing index
        for (int lx_j = lj; lx_j >= jx; lx_j--) {
        for (int ly_j = lj-lx_j; ly_j >= jy; ly_j--) {
            int lz_j = lj - lx_j - ly_j;
            if (lz_j < jz) continue;
            int j = cart_address(lj, lx_j, ly_j, lz_j);
            double cxyz = cx[(jx+lx_j*lj1)*WARP_SIZE]
                        * cy[(jy+ly_j*lj1)*WARP_SIZE]
                        * cz[(jz+lz_j*lj1)*WARP_SIZE];
            out += cxyz * dm[i*nao+j];
        } }
    } }
    return out;
}

template <int L> __device__ static
void dm_to_dm_xyz(double *dm_xyz, double *dm, int nao, int li, int lj,
                  double *ri, double *rj, double cicj)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj1 = lj + 1;
    extern __shared__ double cache[];
    double *cx = cache + sp_id;
    double *cy = cx + lj1 * lj1 * WARP_SIZE;
    double *cz = cy + lj1 * lj1 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj1*lj1*WARP_SIZE, ri[n], rj[n], lj);
    }
    __syncthreads();

    constexpr int nf3 = (L+1)*(L+2)*(L+3)/6;
    Fold3Index *fold3idx = c_i_in_fold3idx + L*nf3/4;
    for (int n = warp_id; n < nf3; n += WARPS) {
        int lx = fold3idx[n].x;
        int ly = fold3idx[n].y;
        int lz = fold3idx[n].z;
        double val = sub_dm_xyz(lx, ly, lz, li, lj, nao, cx, cy, cz, dm);
        dm_xyz[n*WARP_SIZE+sp_id] = val * cicj;
    }
    __syncthreads();
}

template <int L> __device__ static
void dm_xyz_to_dm(double *dm, double *dm_xyz, int nao, int li, int lj,
                  double *ri, double *rj, double cicj, double *cache,
                  int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj1 = lj + 1;
    constexpr int L1 = L + 1;
    double *cx = cache + sp_id;
    double *cy = cx + lj1 * lj1 * WARP_SIZE;
    double *cz = cy + lj1 * lj1 * WARP_SIZE;

    for (int n = warp_id; n < 3; n += WARPS) {
        dm_xyz_coeff(cx+n*lj1*lj1*WARP_SIZE, ri[n], rj[n], lj);
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
        double dm_ij = 0.;
        // TODO: precomputing index
        for (int jx = 0; jx <= lx_j; ++jx) {
            double fac = cicj * cx[(jx+lx_j*lj1)*WARP_SIZE];
            int lx = lx_i + jx;
            for (int jy = 0; jy <= ly_j; ++jy) {
                double cxy = fac * cy[(jy+ly_j*lj1)*WARP_SIZE];
                int ly = ly_i + jy;
                for (int jz = 0; jz <= lz_j; ++jz) {
                    int lz = lz_i + jz;
                    double cxyz = cxy * cz[(jz+lz_j*lj1)*WARP_SIZE];
                    dm_ij += cxyz * dm_xyz[(ADDR2(L,ly,lz)*L1+lx)*WARP_SIZE];
                }
            }
        }
        atomicAdd(dm+i*nao+j, dm_ij);
    }
}
