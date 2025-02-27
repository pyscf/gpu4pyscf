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
void fill_dm_xyz_ipip(double *dm_xyz, double *gx_dmyz, double *xs_exp,
                     int ngridx, int ngrid_span)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int L1 = L + 1;
    int L2 = L + 2;
    int L3 = L + 3;
    int nf2 = (L+1)*(L+2)/2;
    int nf3 = nf2*(L+3)/3;
    if (L <= 3) {
        double r2[(L+1)*(L+2)*(L+3)/6 + (L+1)*(L+2)];
        double r1[L+3];
#pragma unroll
        for (int m = 0; m < nf3+nf2*2; ++m) {
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
                    for (int mx = 0; mx <= L2-my-mz; ++mx, ++n) {
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
                for (int mx = 0; mx <= L2-my-mz; ++mx, ++n) {
                    r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
                    if (warp_id == 0) {
                        dm_xyz[(ADDR2(L,my,mz)*L3+mx)*WARP_SIZE] = r2[n];
                    }
                }
            }
        }
    } else {
        double r3[((L+3)*(L+1)*(L+2)/2+WARPS-1)/WARPS];
#pragma unroll
        for (int n = 0; n < (L3*nf2+WARPS-1)/WARPS; ++n) {
            r3[n] = 0.;
        }
        extern __shared__ double cache[];
        double *xs_cache = cache + sp_id;
        double *yz_cache = cache + (L+3) * WARP_SIZE + sp_id;
        for (int ix = 0; ix < ngridx; ++ix) {
            __syncthreads();
            double *pgx = gx_dmyz + ix * nf2*WARP_SIZE;
            for (int m = warp_id; m < nf2; m += WARPS) {
                yz_cache[m*WARP_SIZE] = pgx[m*WARP_SIZE];
            }
            for (int m = warp_id; m <= L2; m += WARPS) {
                xs_cache[m*WARP_SIZE] = xs_exp[(m*ngrid_span+ix)*WARP_SIZE];
            }
            __syncthreads();
#pragma unroll
            for (int n = 0; n < (L3*nf2)/WARPS; ++n) {
                int m = n * WARPS + warp_id;
                int myz = m / L3;
                int mx  = m % L3;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
            int n = (L3*nf2)/WARPS;
            int m = n * WARPS + warp_id;
            if (m < L3*nf2) {
                int myz = m / L3;
                int mx  = m % L3;
                r3[n] += yz_cache[myz*WARP_SIZE] * xs_cache[mx*WARP_SIZE];
            }
        }
#pragma unroll
        for (int n = 0; n < (L3*nf2)/WARPS; ++n) {
            int m = n * WARPS + warp_id;
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
        int n = (L3*nf2)/WARPS;
        int m = n * WARPS + warp_id;
        if (m < nf2*L3) {
            dm_xyz[m*WARP_SIZE] = r3[n];
        }
    }
}

template <int L> __device__ static
void _dm_xyz_to_dm_derivx(double *dm, double *dm_yzx, int nao, int li, int lj,
                          double *ri, double *rj, double ai, double aj, double cicj,
                          double *cache, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    int L3 = L + 3;
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

    double ai2 = -2. * ai;
    double aj2 = -2. * aj;
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
        double fac;
        for (int jy = 0; jy <= ly_j; ++jy) {
            double fac_cy = cicj * cy[(jy+ly_j*lj3)*WARP_SIZE];
            int ly = ly_i + jy;
            for (int jz = 0; jz <= lz_j; ++jz) {
                double cyz = fac_cy * cz[(jz+lz_j*lj3)*WARP_SIZE];
                int lz = lz_i + jz;
                int yz_idx = ADDR2(L, ly, lz);
                if (lx_i > 0) {
                    if (lx_j > 0) {
                        fac = lx_i * lx_j * cyz;
                        for (int jx = 0; jx <= lx_j-1; ++jx) {
                            int lx = lx_i - 1 + jx;
                            double cyzx = fac * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
                            dm_ij += cyzx * dm_yzx[(yz_idx*L3+lx)*WARP_SIZE];
                        }
                    }
                    fac = lx_i * aj2 * cyz;
                    for (int jx = 0; jx <= lx_j+1; ++jx) {
                        int lx = lx_i - 1 + jx;
                        double cyzx = fac * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
                        dm_ij += cyzx * dm_yzx[(yz_idx*L3+lx)*WARP_SIZE];
                    }
                }
                if (lx_j > 0) {
                    fac = ai2 * lx_j * cyz;
                    for (int jx = 0; jx <= lx_j-1; ++jx) {
                        int lx = lx_i + 1 + jx;
                        double cyzx = fac * cx[(jx+(lx_j-1)*lj3)*WARP_SIZE];
                        dm_ij += cyzx * dm_yzx[(yz_idx*L3+lx)*WARP_SIZE];
                    }
                }
                fac = ai2 * aj2 * cyz;
                for (int jx = 0; jx <= lx_j+1; ++jx) {
                    int lx = lx_i + 1 + jx;
                    double cyzx = fac * cx[(jx+(lx_j+1)*lj3)*WARP_SIZE];
                    dm_ij += cyzx * dm_yzx[(yz_idx*L3+lx)*WARP_SIZE];
                }
            }
        }
        atomicAdd(dm+i*nao+j, dm_ij);
    }
}

template <int L> __device__ static
void _dm_xyz_to_dm_derivy(double *dm, double *dm_xzy, int nao, int li, int lj,
                          double *ri, double *rj, double ai, double aj, double cicj,
                          double *cache, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    int L3 = L + 3;
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

    double ai2 = -2. * ai;
    double aj2 = -2. * aj;
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
        double fac;
        for (int jx = 0; jx <= lx_j; ++jx) {
            double fac_cx = cicj * cx[(jx+lx_j*lj3)*WARP_SIZE];
            int lx = lx_i + jx;
            for (int jz = 0; jz <= lz_j; ++jz) {
                double cxz = fac_cx * cz[(jz+lz_j*lj3)*WARP_SIZE];
                int lz = lz_i + jz;
                int xz_idx = ADDR2(L, lx, lz);
                if (ly_i > 0) {
                    if (ly_j > 0) {
                        fac = ly_i * ly_j * cxz;
                        for (int jy = 0; jy <= ly_j-1; ++jy) {
                            int ly = ly_i - 1 + jy;
                            double cxzy = fac * cy[(jy+(ly_j-1)*lj3)*WARP_SIZE];
                            dm_ij += cxzy * dm_xzy[(xz_idx*L3+ly)*WARP_SIZE];
                        }
                    }
                    fac = ly_i * aj2 * cxz;
                    for (int jy = 0; jy <= ly_j+1; ++jy) {
                        int ly = ly_i + 1 + jy;
                        double cxzy = fac * cy[(jy+(ly_j+1)*lj3)*WARP_SIZE];
                        dm_ij += cxzy * dm_xzy[(xz_idx*L3+ly)*WARP_SIZE];
                    }
                }
                if (ly_j > 0) {
                    fac = ai2 * ly_j * cxz;
                    for (int jy = 0; jy <= ly_j-1; ++jy) {
                        int ly = ly_i + 1 + jy;
                        double cxzy = fac * cy[(jy+(ly_j-1)*lj3)*WARP_SIZE];
                        dm_ij += cxzy * dm_xzy[(xz_idx*L3+ly)*WARP_SIZE];
                    }
                }
                fac = ai2 * aj2 * cxz;
                for (int jy = 0; jy <= ly_j+1; ++jy) {
                    int ly = ly_i + 1 + jy;
                    double cxzy = fac * cy[(jy+(ly_j+1)*lj3)*WARP_SIZE];
                    dm_ij += cxzy * dm_xzy[(xz_idx*L3+ly)*WARP_SIZE];
                }
            }
        }
        atomicAdd(dm+i*nao+j, dm_ij);
    }
}

template <int L> __device__ static
void _dm_xyz_to_dm_derivz(double *dm, double *dm_xyz, int nao, int li, int lj,
                          double *ri, double *rj, double ai, double aj, double cicj,
                          double *cache, int npairs_per_block)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int lj2 = lj + 2;
    int lj3 = lj + 3;
    int L3 = L + 3;
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

    double ai2 = -2. * ai;
    double aj2 = -2. * aj;
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
        double fac;
        for (int jx = 0; jx <= lx_j; ++jx) {
            double fac_cx = cicj * cx[(jx+lx_j*lj3)*WARP_SIZE];
            int lx = lx_i + jx;
            for (int jy = 0; jy <= ly_j; ++jy) {
                double cxy = fac_cx * cy[(jy+ly_j*lj3)*WARP_SIZE];
                int ly = ly_i + jy;
                int xy_idx = ADDR2(L, lx, ly);
                if (lz_i > 0) {
                    if (lz_j > 0) {
                        fac = lz_i * lz_j * cxy;
                        for (int jz = 0; jz <= lz_j-1; ++jz) {
                            int lz = lz_i - 1 + jz;
                            double cxyz = fac * cz[(jz+(lz_j-1)*lj3)*WARP_SIZE];
                            dm_ij += cxyz * dm_xyz[(xy_idx*L3+lz)*WARP_SIZE];
                        }
                    }
                    fac = lz_i * aj2 * cxy;
                    for (int jz = 0; jz <= lz_j+1; ++jz) {
                        int lz = lz_i - 1 + jz;
                        double cxyz = fac * cz[(jz+(lz_j+1)*lj3)*WARP_SIZE];
                        dm_ij += cxyz * dm_xyz[(xy_idx*L3+lz)*WARP_SIZE];
                    }
                }
                if (lz_j > 0) {
                    fac = ai2 * lz_j * cxy;
                    for (int jz = 0; jz <= lz_j-1; ++jz) {
                        int lz = lz_i + 1 + jz;
                        double cxyz = fac * cz[(jz+(lz_j-1)*lj3)*WARP_SIZE];
                        dm_ij += cxyz * dm_xyz[(xy_idx*L3+lz)*WARP_SIZE];
                    }
                }
                fac = ai2 * aj2 * cxy;
                for (int jz = 0; jz <= lz_j+1; ++jz) {
                    int lz = lz_i + 1 + jz;
                    double cxyz = fac * cz[(jz+(lz_j+1)*lj3)*WARP_SIZE];
                    dm_ij += cxyz * dm_xyz[(xy_idx*L3+lz)*WARP_SIZE];
                }
            }
        }
        atomicAdd(dm+i*nao+j, dm_ij);
    }
}

template <int L, int TILE> __device__ static
void _eval_mat_tau_kernel(double *out, double *rho, MGridEnvVars envs,
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
    double cicj = .5 * ci * cj * exp(-theta_ij * rr_ij);
    if (sp_id >= npairs_this_block) {
        cicj = 0.;
    }
    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    out += i0*nao+j0;

    int nf2 = (L+1)*(L+2)/2;
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
    int xs_size = ngrid_span * (L+3) * WARP_SIZE;
    int *grid_start = (int *)pool + sp_id;
    double *xs_exp = pool + WARP_SIZE*3 + sp_id;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *gx_dmyz = zs_exp + xs_size;
    init_orth_data(xs_exp, grid_start, envs, bounds, ri, rj, ai, aj, L+2);

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
    // derivx: contract mesh_y, mesh_z
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
                double r2[(L+1)*(L+2)/2];
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

    fill_dm_xyz_ipip<L>(dm_xyz, gx_dmyz, xs_exp, ngridx, ngrid_span);
    _dm_xyz_to_dm_derivx<L>(out, dm_xyz, nao, li, lj, ri, rj, ai, aj,
                            cicj, cache, npairs_this_block);
    __syncthreads();

    // derivy: contract mesh_x, mesh_z
    xs_cache = cache + sp_id;
    zs_cache = xs_cache + TILE * (L+1) * WARP_SIZE;
    for (int n = warp_id; n < ngridx*nf2; n += WARPS) {
        gx_dmyz[n*WARP_SIZE] = 0.;
    }
    for (int iz0 = 0; iz0 < ngridz; iz0 += TILE) {
        __syncthreads();
        int nz = load_xs(zs_cache, zs_exp, iz0, ngridz, L, TILE, ngrid_span, warp_id);
        for (int ix0 = 0; ix0 < ngridx; ix0 += TILE) {
            __syncthreads();
            int nx = load_xs(xs_cache, xs_exp, ix0, ngridx, L, TILE, ngrid_span, warp_id);
            for (int iy = warp_id; iy < ngridy; iy += WARPS) {
                int ty = (iy + ny0) % mesh_y;
                double r2[(L+1)*(L+2)/2];
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
                double *pgy = gx_dmyz + iy * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgy[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz_ipip<L>(dm_xyz, gx_dmyz, ys_exp, ngridy, ngrid_span);
    _dm_xyz_to_dm_derivy<L>(out, dm_xyz, nao, li, lj, ri, rj, ai, aj,
                            cicj, cache, npairs_this_block);
    __syncthreads();

    // derivz: contract mesh_x, mesh_y
    xs_cache = cache + sp_id;
    ys_cache = xs_cache + TILE * (L+1) * WARP_SIZE;
    for (int n = warp_id; n < ngridx*nf2; n += WARPS) {
        gx_dmyz[n*WARP_SIZE] = 0.;
    }
    for (int ix0 = 0; ix0 < ngridx; ix0 += TILE) {
        __syncthreads();
        int nx = load_xs(xs_cache, xs_exp, ix0, ngridx, L, TILE, ngrid_span, warp_id);
        for (int iy0 = 0; iy0 < ngridy; iy0 += TILE) {
            __syncthreads();
            int ny = load_xs(ys_cache, ys_exp, iy0, ngridy, L, TILE, ngrid_span, warp_id);
            for (int iz = warp_id; iz < ngridz; iz += WARPS) {
                int tz = (iz + nz0) % mesh_z;
                double r2[(L+1)*(L+2)/2];
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
                double *pgz = gx_dmyz + iz * nf2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < nf2; ++m) {
                    pgz[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    fill_dm_xyz_ipip<L>(dm_xyz, gx_dmyz, zs_exp, ngridz, ngrid_span);
    _dm_xyz_to_dm_derivz<L>(out, dm_xyz, nao, li, lj, ri, rj, ai, aj,
                            cicj, cache, npairs_this_block);
}

template <int L, int TILE> __global__
void eval_mat_tau_kernel(double *out, double *rho, MGridEnvVars envs,
                         MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+3) * ngrid_span;
    int nf2 = (L+1)*(L+2)/2;
    int l3 = nf2*(L+3);
    pool += (xs_size*3 + nf2*ngrid_span + 3 + l3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_mat_tau_kernel<L, TILE>(out, rho, envs, bounds, pool, pair_idx0);
        if (thread_id == 0) {
            pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
        }
        __syncthreads();
    }
}

static size_t buflen_tau(int l, int tile)
{
    int lj = MIN(l, LMAX);
    size_t len1 = WARP_SIZE * tile * (l+1) * 2;
    size_t len2 = WARP_SIZE * (lj+3)*(lj+3) * 3;
    if (l < 4) {
        len2 += WARP_SIZE * (l+1)*(l+2)/2*(l+3);
    }
    return MAX(len1, len2) * sizeof(double);
}

extern "C" {
int MG_eval_mat_tau_orth(double *out, double *rho, MGridEnvVars envs,
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
    case 0: eval_mat_tau_kernel<0,32> <<<workers, THREADS, buflen_tau(0,32)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 1: eval_mat_tau_kernel<1,32> <<<workers, THREADS, buflen_tau(1,32)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 2: eval_mat_tau_kernel<2,16> <<<workers, THREADS, buflen_tau(2,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 3: eval_mat_tau_kernel<3,16> <<<workers, THREADS, buflen_tau(3,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 4: eval_mat_tau_kernel<4,16> <<<workers, THREADS, buflen_tau(4,16)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 5: eval_mat_tau_kernel<5, 8> <<<workers, THREADS, buflen_tau(5, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 6: eval_mat_tau_kernel<6, 8> <<<workers, THREADS, buflen_tau(6, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 7: eval_mat_tau_kernel<7, 8> <<<workers, THREADS, buflen_tau(7, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    case 8: eval_mat_tau_kernel<8, 8> <<<workers, THREADS, buflen_tau(8, 8)>>>(out, rho, envs, bounds, pool, batch_head); break;
    default: 
        fprintf(stderr, "MG_eval_mat_tau_orth does not support l>8\n");
        cudaFree(batch_head);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_mat_tau_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}
}
