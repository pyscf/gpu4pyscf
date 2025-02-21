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
#include "cart2xyz.cu"

#define SHARED_RHO_SIZE 2097152

__device__ static
void init_orth_data(double *pool, int *grid_start,
                    MGridEnvVars envs, MGridBounds bounds, double *ri, double *rj,
                    double ai, double aj, int l1, int warp_id)
{
    int ngrid_span = bounds.ngrid_radius * 2;
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
            for (int m = 1; m < l1; m++) {
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
    }
    __syncthreads();
}

// load a small chunk [0:l+1, ix0:ix0+batch_size, 0:WARPS] from xs_exp
__device__ inline
int load_xs_chunk(double *xs_cache, double *xs_exp, int ix0, int ngridx,
                  int l, int batch_size, int xs_stride)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id % WARPS;
    int g_id  = thread_id / WARPS;
    int nx = MIN(ngridx - ix0, batch_size);
    for (int ix = g_id; ix < nx; ix += WARP_SIZE) {
        double *_xs_exp = xs_exp + (ix0+ix) * WARP_SIZE + sp_id;
        double *_xs_cache = xs_cache + ix * WARPS + sp_id;
        int batch_stride = batch_size * WARPS;
        int xs_size = xs_stride * WARP_SIZE;
        for (int m = 0; m <= l; ++m) {
            _xs_cache[m*batch_stride] = _xs_exp[m*xs_size];
        }
    }
    __syncthreads();
    return nx;
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


template <int L, int TILEy> __device__ static
void _eval_rho_orth_kernel(double *rho, double *dm, MGridEnvVars envs,
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

    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int mesh_xyz = mesh_x * mesh_yz;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+1) * ngrid_span * WARP_SIZE;
    double *xs_exp = pool;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *dm_xyz = zs_exp + xs_size;
    int nf3 = (L+1)*(L+2)*(L+3)/6;
    int *grid_start = (int *)(dm_xyz + nf3*WARP_SIZE);
    init_orth_data(xs_exp+sp_id, grid_start+sp_id, envs, bounds,
                   ri, rj, ai, aj, L+1, warp_id);

    int nao = envs.nao;
    int *ao_loc = envs.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    // TODO: multiple dms
    _dm_to_dm_xyz(dm_xyz, dm+i0*nao+j0, nao, li, lj, ri, rj, cicj, sp_id, warp_id);

    double *rho_local = rho;
    if (mesh_xyz < SHARED_RHO_SIZE) {
        rho_local += mesh_xyz * warp_id;
    }
    double r1[L+1];
    double dmxy_gz[L+1];
    double dmx_gyz[TILEy*(L+1)];
    extern __shared__ double cache[];
    double *xs_cache = cache;
    double *ys_cache = xs_cache + WARPS * WARP_SIZE * (L+1);

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
    //TODO: iz = thread_id % ngrid_span
    int iz = sp_id;
    for (int sp_start = 0; sp_start < npairs_this_block; sp_start += WARPS) {
        int sp_id = sp_start + warp_id;
        double *dm_local = dm_xyz + sp_id;
        int nx0 = grid_start[0*WARP_SIZE+sp_id];
        int ny0 = grid_start[1*WARP_SIZE+sp_id];
        int nz0 = grid_start[2*WARP_SIZE+sp_id];
        // Using translation vectors to shift the first grid to cell 0.
        // This can simplify the address computation for the wrapped-around grid.
        nx0 = (nx0 % mesh_x + mesh_x) % mesh_x;
        ny0 = (ny0 % mesh_y + mesh_y) % mesh_y;
        nz0 = (nz0 % mesh_z + mesh_z) % mesh_z;

        // Using ngrid_span for loop upper bound to ensure all warps process the
        // same number of iterations. ngridz might be different on different warps.
        for (int iz0 = 0; iz0 < ngridz; iz0 += WARP_SIZE) {
            int nz = MIN(ngridz - iz0, WARP_SIZE);
            if (iz < nz) {
                double *_zs_exp = zs_exp + (iz0+iz) * WARP_SIZE + sp_id;
                for (int mz = 0; mz <= L; ++mz) {
                    r1[mz] = _zs_exp[mz*ngrid_span*WARP_SIZE];
                }
            }
            for (int iy0 = 0; iy0 < ngridy; iy0 += TILEy) {
                __syncthreads();
                int ny = load_xs_chunk(ys_cache, ys_exp+sp_start, iy0, ngridy,
                                       L, TILEy, ngrid_span);
                if (iz < nz && sp_id < npairs_this_block) {
                    int n = 0;
#pragma unroll
                    for (int mx = 0; mx <= L; ++mx) {
                        for (int my = 0; my <= L-mx; ++my) {
                            double val = 0.;
                            for (int mz = 0; mz <= L-mx-my; ++mz, ++n) {
                                val += r1[mz] * dm_local[n*WARP_SIZE];
                            }
                            dmxy_gz[my] = val;
                        }
#pragma unroll
                        for (int iy = 0; iy < TILEy; ++iy) {
                            if (iy >= ny) {
                                break;
                            }
                            double val = 0.;
                            for (int my = 0; my <= L-mx; ++my) {
                                val += ys_cache[(my*TILEy+iy)*WARPS+warp_id] * dmxy_gz[my];
                            }
                            dmx_gyz[iy*(L+1)+mx] = val;
                        }
                    }
                }
                for (int ix0 = 0; ix0 < ngridx; ix0 += WARP_SIZE) {
                    __syncthreads();
                    int nx = load_xs_chunk(xs_cache, xs_exp+sp_start, ix0, ngridx,
                                           L, WARP_SIZE, ngrid_span);
                    if (iz < nz && sp_id < npairs_this_block) {
                        int tz = (iz+iz0 + nz0) % mesh_z;
                        if (mesh_xyz <= SHARED_RHO_SIZE) {
                            for (int ix = 0; ix < nx; ++ix) {
                                int tx = (ix+ix0 + nx0) % mesh_x;
#pragma unroll
                                for (int iy = 0; iy < TILEy; ++iy) {
                                    if (iy >= ny) {
                                        break;
                                    }
                                    double val = 0.;
#pragma unroll
                                    for (int mx = 0; mx <= L; ++mx) {
                                        val += xs_cache[(mx*WARP_SIZE+ix)*WARPS+warp_id] * dmx_gyz[iy*(L+1)+mx];
                                    }
                                    int ty = (iy+iy0 + ny0) % mesh_y;
                                    int addr = tx * mesh_yz + ty * mesh_z + tz;
                                    rho_local[addr] += val;
                                }
                            }
                        } else {
                            for (int ix = 0; ix < nx; ++ix) {
                                int tx = (ix+ix0 + nx0) % mesh_x;
#pragma unroll
                                for (int iy = 0; iy < TILEy; ++iy) {
                                    if (iy >= ny) {
                                        break;
                                    }
                                    double val = 0.;
#pragma unroll
                                    for (int mx = 0; mx <= L; ++mx) {
                                        val += xs_cache[(mx*WARP_SIZE+ix)*WARPS+warp_id] * dmx_gyz[iy*(L+1)+mx];
                                    }
                                    int ty = (iy+iy0 + ny0) % mesh_y;
                                    int addr = tx * mesh_yz + ty * mesh_z + tz;
                                    atomicAdd(rho_local+addr, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int L, int TILEy> __global__
void eval_rho_orth_kernel(double *rho, double *dm, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int *mesh = bounds.mesh;
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int mesh_yz = mesh_y * mesh_z;
    int mesh_xyz = mesh_x * mesh_yz;
    if (mesh_xyz <= SHARED_RHO_SIZE) {
        rho += mesh_xyz * WARPS * b_id;
    }
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+1) * ngrid_span;
    int nf3 = (L+1)*(L+2)*(L+3)/6;
    pool += (xs_size*3 + nf3 + 3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_rho_orth_kernel<L, TILEy>(rho, dm, envs, bounds, pool, pair_idx0);
        if (thread_id == 0) {
            pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
        }
        __syncthreads();
    }
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

static size_t buflen1(int l, int tiley)
{
    int lj = MIN(l, LMAX);
    size_t len1 = (WARP_SIZE + tiley) * WARPS * (l+1);
    size_t len2 = WARP_SIZE * (lj+1)*(lj+1) * 3;
    return MAX(len1, len2) * sizeof(double);
}
int eval_rho_orth(double *rho, double *dm, MGridEnvVars *envs, MGridBounds *bounds,
                  int l, double *pool, uint32_t *batch_head, int workers)
{
    switch (l) {
    case 0: eval_rho_orth_kernel<0,32> <<<workers, THREADS, buflen1(0,32)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 1: eval_rho_orth_kernel<1,32> <<<workers, THREADS, buflen1(1,32)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 2: eval_rho_orth_kernel<2,16> <<<workers, THREADS, buflen1(2,16)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 3: eval_rho_orth_kernel<3,16> <<<workers, THREADS, buflen1(3,16)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 4: eval_rho_orth_kernel<4,16> <<<workers, THREADS, buflen1(4,16)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 5: eval_rho_orth_kernel<5, 8> <<<workers, THREADS, buflen1(5, 8)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 6: eval_rho_orth_kernel<6, 8> <<<workers, THREADS, buflen1(6, 8)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 7: eval_rho_orth_kernel<7, 8> <<<workers, THREADS, buflen1(7, 8)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    case 8: eval_rho_orth_kernel<8, 8> <<<workers, THREADS, buflen1(8, 8)>>>(rho, dm, *envs, *bounds, pool, batch_head); break;
    default: return 1;
    }
    return 0;
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
