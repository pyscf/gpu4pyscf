/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

template <int L, int TILE> __device__ static
void _eval_mat_lda_kernel(double *out, double *rho, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t pair_idx0)
{
    static_assert(L <= 6, "L too large!");
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

    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = ngrid_span * (L+1) * WARP_SIZE;
    double *xs_exp = pool + sp_id;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *gx_dmyz = zs_exp + xs_size;
    int *grid_start = (int *)(pool + xs_size*3 + ngrid_span*(L+1)*(L+2)/2*WARP_SIZE) + sp_id;
    init_orth_data(xs_exp, grid_start, envs, bounds,
                   ri, rj, ai, aj, L+1, warp_id);

    double r2[(L+1)*(L+2)*(L+3)/6];
    double r1[L+1];
    extern __shared__ double cache[];
    double *ys_cache = cache + sp_id;
    double *zs_cache = ys_cache + TILE * (L+1) * WARP_SIZE;

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

    for (int n = warp_id; n < ngridx*(L+1)*(L+2)/2; n += WARPS) {
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
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
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
#pragma unroll
                        for (int mz = 0; mz <= L-my; ++mz) {
                            r2[ADDR2(L,my,mz)] += ys_cache[(my*TILE+iy)*WARP_SIZE] * r1[mz];
                        }
                    }
                }
                double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
                    pgx[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < (L+1)*(L+2)*(L+3)/6; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
        for (int mx = 0; mx <= L; ++mx) {
            r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
        }
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
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
    double *dm_xyz = cache + sp_id;
#pragma unroll
    for (int mx = 0; mx <= L; ++mx) {
#pragma unroll
        for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-mx-my; ++mz) {
                int n = ADDR3(L,mx,my,mz);
                r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
                if (warp_id == 0) {
                    dm_xyz[n*WARP_SIZE] = r2[n];
                }
            }
        }
    }
    __syncthreads();

    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    _dm_xyz_to_dm(out+i0*nao+j0, dm_xyz, nao, li, lj, L, ri, rj, cicj,
                  cache+(L+1)*(L+2)*(L+3)/6*WARP_SIZE,
                  sp_id, warp_id, npairs_this_block);
}

template <> __device__
void _eval_mat_lda_kernel<7, 8>(double *out, double *rho, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t pair_idx0)
{
    int L = 7;
    int TILE = 8;
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

    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = ngrid_span * (L+1) * WARP_SIZE;
    double *xs_exp = pool + sp_id;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *gx_dmyz = zs_exp + xs_size;
    int *grid_start = (int *)(pool + xs_size*3 + ngrid_span*(L+1)*(L+2)/2*WARP_SIZE) + sp_id;
    init_orth_data(xs_exp, grid_start, envs, bounds,
                   ri, rj, ai, aj, L+1, warp_id);

    double r2[64];
    double r1[8];
    extern __shared__ double cache[];
    double *ys_cache = cache + sp_id;
    double *zs_cache = ys_cache + TILE * (L+1) * WARP_SIZE;

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

    for (int n = warp_id; n < ngridx*(L+1)*(L+2)/2; n += WARPS) {
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
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
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
#pragma unroll
                        for (int mz = 0; mz <= L-my; ++mz) {
                            r2[ADDR2(L,my,mz)] += ys_cache[(my*TILE+iy)*WARP_SIZE] * r1[mz];
                        }
                    }
                }
                double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
                    pgx[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    // reuse the global memory in pool. (L+1)*(L+2)*(L+3)/6 is generally smaller
    // than (ngrid_span * (L+1) * 2) occupied by ys_exp and zs_exp
    double *dm_xyz = ys_exp;
    int offset = (L+1)*(L+2)/2 + L*(L+1)/2;
#pragma unroll
    for (int m = 0; m < offset; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
        for (int mx = 0; mx <= 1; ++mx) {
            r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
        }
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
        for (int my = 0; my <= L; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-my; ++mz) {
                double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                for (int mx = 0; mx <= MIN(1, L-my-mz); ++mx) {
                    r2[ADDR3(L,mx,my,mz)] += r1[mx] * t;
                }
            }
        }
    }
#pragma unroll
    for (int mx = 0; mx <= 1; ++mx) {
#pragma unroll
        for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-mx-my; ++mz) {
                int n = ADDR3(L,mx,my,mz);
                r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
                if (warp_id == 0) {
                    dm_xyz[n*WARP_SIZE] = r2[n];
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < (L+1)*(L+2)*(L+3)/6-offset; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
        for (int mx = 2; mx <= L; ++mx) {
            r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
        }
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
        for (int my = 0; my <= L-2; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-2-my; ++mz) {
                double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                for (int mx = 2; mx <= L-my-mz; ++mx) {
                    r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
                }
            }
        }
    }
#pragma unroll
    for (int mx = 2; mx <= L; ++mx) {
#pragma unroll
        for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-mx-my; ++mz) {
                int n = ADDR3(L,mx,my,mz);
                r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
                if (warp_id == 0) {
                    dm_xyz[n*WARP_SIZE] = r2[n-offset];
                }
            }
        }
    }
    __syncthreads();

    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    _dm_xyz_to_dm(out+i0*nao+j0, dm_xyz, nao, li, lj, L, ri, rj, cicj, cache,
                  sp_id, warp_id, npairs_this_block);
}

template <> __device__
void _eval_mat_lda_kernel<8, 8>(double *out, double *rho, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t pair_idx0)
{
    int L = 8;
    int TILE = 8;
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

    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = ngrid_span * (L+1) * WARP_SIZE;
    double *xs_exp = pool + sp_id;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *gx_dmyz = zs_exp + xs_size;
    int *grid_start = (int *)(pool + xs_size*3 + ngrid_span*(L+1)*(L+2)/2*WARP_SIZE) + sp_id;
    init_orth_data(xs_exp, grid_start, envs, bounds,
                   ri, rj, ai, aj, L+1, warp_id);

    double r2[64];
    double r1[9];
    extern __shared__ double cache[];
    double *ys_cache = cache + sp_id;
    double *zs_cache = ys_cache + TILE * (L+1) * WARP_SIZE;

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

    for (int n = warp_id; n < ngridx*(L+1)*(L+2)/2; n += WARPS) {
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
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
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
#pragma unroll
                        for (int mz = 0; mz <= L-my; ++mz) {
                            r2[ADDR2(L,my,mz)] += ys_cache[(my*TILE+iy)*WARP_SIZE] * r1[mz];
                        }
                    }
                }
                double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
                for (int m = 0; m < (L+1)*(L+2)/2; ++m) {
                    pgx[m*WARP_SIZE] += r2[m];
                }
            }
        }
    }
    __syncthreads();

    // reuse the global memory in pool. (L+1)*(L+2)*(L+3)/6 is generally smaller
    // than (ngrid_span * (L+1) * 2) occupied by ys_exp and zs_exp
    double *dm_xyz = ys_exp;
    int offset = (L+1)*(L+2)/2;
#pragma unroll
    for (int m = 0; m < offset; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
        r1[0] = xs_exp[ix*WARP_SIZE];
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
        for (int my = 0; my <= L; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-my; ++mz) {
                double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
                r2[ADDR3(L,0,my,mz)] += r1[0] * t;
            }
        }
    }
#pragma unroll
    for (int my = 0; my <= L; ++my) {
#pragma unroll
        for (int mz = 0; mz <= L-my; ++mz) {
            int n = ADDR3(L,0,my,mz);
            r2[n] = reduce_warps(r2[n], ngridx, thread_id, sp_id, warp_id);
            if (warp_id == 0) {
                dm_xyz[n*WARP_SIZE] = r2[n];
            }
        }
    }

#pragma unroll
    for (int m = 0; m < L*(L+1)/2 + (L-1)*L/2; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
        for (int mx = 1; mx <= 2; ++mx) {
            r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
        }
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
        for (int my = 0; my <= L-1; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-1-my; ++mz) {
                double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                for (int mx = 1; mx <= MIN(2, L-my-mz); ++mx) {
                    r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
                }
            }
        }
    }
#pragma unroll
    for (int mx = 1; mx <= 2; ++mx) {
#pragma unroll
        for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-mx-my; ++mz) {
                int n = ADDR3(L,mx,my,mz);
                r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
                if (warp_id == 0) {
                    dm_xyz[n*WARP_SIZE] = r2[n-offset];
                }
            }
        }
    }

    offset = (L+1)*(L+2)/2 + L*(L+1)/2 + (L-1)*L/2;
#pragma unroll
    for (int m = 0; m < (L+1)*(L+2)*(L+3)/6 - offset; ++m) {
        r2[m] = 0.;
    }
    for (int ix = warp_id; ix < ngridx; ix += WARPS) {
#pragma unroll
        for (int mx = 3; mx <= L; ++mx) {
            r1[mx] = xs_exp[(mx*ngrid_span+ix)*WARP_SIZE];
        }
        double *pgx = gx_dmyz + ix * (L+1)*(L+2)/2*WARP_SIZE;
#pragma unroll
        for (int my = 0; my <= L-3; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-3-my; ++mz) {
                double t = pgx[ADDR2(L,my,mz)*WARP_SIZE];
#pragma unroll
                for (int mx = 3; mx <= L-my-mz; ++mx) {
                    r2[ADDR3(L,mx,my,mz)-offset] += r1[mx] * t;
                }
            }
        }
    }
#pragma unroll
    for (int mx = 3; mx <= L; ++mx) {
#pragma unroll
        for (int my = 0; my <= L-mx; ++my) {
#pragma unroll
            for (int mz = 0; mz <= L-mx-my; ++mz) {
                int n = ADDR3(L,mx,my,mz);
                r2[n-offset] = reduce_warps(r2[n-offset], ngridx, thread_id, sp_id, warp_id);
                if (warp_id == 0) {
                    dm_xyz[n*WARP_SIZE] = r2[n-offset];
                }
            }
        }
    }
    __syncthreads();

    int *ao_loc = envs.ao_loc;
    int nao = envs.nao;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    _dm_xyz_to_dm(out+i0*nao+j0, dm_xyz, nao, li, lj, L, ri, rj, cicj, cache,
                  sp_id, warp_id, npairs_this_block);
}

template <int L, int TILE> __global__
void eval_mat_lda_kernel(double *out, double *rho, MGridEnvVars envs,
                         MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+1) * ngrid_span;
    int nf2 = (L+1)*(L+2)/2;
    pool += (xs_size*3 + nf2*ngrid_span + 3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_mat_lda_kernel<L, TILE>(out, rho, envs, bounds, pool, pair_idx0);
        if (thread_id == 0) {
            pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
        }
        __syncthreads();
    }
}

static size_t buflen2(int l, int tile)
{
    int lj = MIN(l, LMAX);
    size_t len1 = WARP_SIZE * tile * (l+1) * 2;
    size_t len2 = WARP_SIZE * (lj+1)*(lj+1) * 3;
    if (l < 7) {
        len2 += WARP_SIZE * (l+1)*(l+2)*(l+3)/6;
    }
    return MAX(len1, len2) * sizeof(double);
}
int eval_mat_lda_orth(double *out, double *rho, MGridEnvVars *envs, MGridBounds *bounds,
                      int l, double *pool, uint32_t *batch_head, int workers)
{
    switch (l) {
    case 0: eval_mat_lda_kernel<0,32> <<<workers, THREADS, buflen2(0,32)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 1: eval_mat_lda_kernel<1,32> <<<workers, THREADS, buflen2(1,32)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 2: eval_mat_lda_kernel<2,16> <<<workers, THREADS, buflen2(2,16)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 3: eval_mat_lda_kernel<3,16> <<<workers, THREADS, buflen2(3,16)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 4: eval_mat_lda_kernel<4,16> <<<workers, THREADS, buflen2(4,16)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 5: eval_mat_lda_kernel<5, 8> <<<workers, THREADS, buflen2(5, 8)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 6: eval_mat_lda_kernel<6, 8> <<<workers, THREADS, buflen2(6, 8)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 7: eval_mat_lda_kernel<7, 8> <<<workers, THREADS, buflen2(7, 8)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    case 8: eval_mat_lda_kernel<8, 8> <<<workers, THREADS, buflen2(8, 8)>>>(out, rho, *envs, *bounds, pool, batch_head); break;
    default: return 1;
    }
    return 0;
}
