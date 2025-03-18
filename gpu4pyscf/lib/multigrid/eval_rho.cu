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
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+1) * ngrid_span * WARP_SIZE;
    constexpr int nf2 = (L+1)*(L+2)/2;
    constexpr int nf3 = nf2*(L+3)/3;
    double *xs_exp = pool;
    double *ys_exp = xs_exp + xs_size;
    double *zs_exp = ys_exp + xs_size;
    double *dm_xyz = zs_exp + xs_size;
    double *gy_dmxz = dm_xyz + nf3*WARP_SIZE;
    int *grid_start = (int *)(gy_dmxz + nf2*ngrid_span*WARP_SIZE);
    init_orth_data(xs_exp+sp_id, grid_start+sp_id, envs, bounds,
                   ri, rj, ai, aj, L);

    int nao = envs.nao;
    int *ao_loc = envs.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    // TODO: multiple dms
    dm_to_dm_xyz<L>(dm_xyz, dm+i0*nao+j0, nao, li, lj, ri, rj, cicj);

    double r1[L+1];
    double dmx_gyz[L+1];
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

    // contract ys_exp with dmxyz and cache results in gy_dmxz
    double *dm_cache = cache;
    int xs_stride = ngrid_span * WARP_SIZE;
    for (int n = warp_id; n < nf3; n += WARPS) {
        dm_cache[n*WARP_SIZE+sp_id] = dm_xyz[n*WARP_SIZE+sp_id];
    }
    __syncthreads();
    for (int iy = warp_id; iy < ngridy; iy += WARPS) {
        if (iy < ngridy && sp_id < npairs_this_block) {
            double *gy_local = gy_dmxz + sp_id * nf2*ngridy;
            double *ys_local = ys_exp + iy * WARP_SIZE + sp_id;
#pragma unroll
            for (int m = 0; m <= L; ++m) {
                r1[m] = ys_local[m*xs_stride];
            }
#pragma unroll
            for (int mx = 0; mx <= L; ++mx) {
#pragma unroll
                for (int mz = 0; mz <= L-mx; ++mz) {
                    double val = 0.;
#pragma unroll
                    for (int my = 0; my <= L-mx-mz; ++my) {
                        int n = ADDR3(L,mx,my,mz);
                        val += r1[my] * dm_cache[n*WARP_SIZE+sp_id];
                    }
                    gy_local[iy*nf2+ADDR2(L,mx,mz)] = val;
                }
            }
        }
    }

    int ngridyz = ngridy * ngridz;
    int ix_stride = 1;
    if (ngridyz * 2 < THREADS) { // for small mesh
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
        // load xs_exp, gy_dmxz and zs_exp into shared mem
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
            int ty = (iy + ny0) % mesh_y;
            int tz = (iz + nz0) % mesh_z;
#pragma unroll
            for (int mz = 0; mz <= L; ++mz) {
                r1[mz] = zs_cache[mz*ngridz+iz];
            }
#pragma unroll
            for (int mx = 0; mx <= L; ++mx) {
                dmx_gyz[mx] = 0.;
#pragma unroll
                for (int mz = 0; mz <= L-mx; ++mz) {
                    dmx_gyz[mx] += ys_cache[iy*nf2+ADDR2(L,mx,mz)] * r1[mz];
                }
            }
            int addr_yz = ty * mesh_z + tz;
            double *rho_local = rho + addr_yz;
            for (int ix = ix_inc; ix < ngridx; ix += ix_stride) {
                int tx = (ix + nx0) % mesh_x;
                double val = 0.;
#pragma unroll
                for (int mx = 0; mx <= L; ++mx) {
                    val += xs_cache[mx*ngridx+ix] * dmx_gyz[mx];
                }
                atomicAdd(rho_local+tx*mesh_yz, val);
            }
        }
    }
}

template <int L> __global__
void eval_rho_orth_kernel(double *rho, double *dm, MGridEnvVars envs,
                          MGridBounds bounds, double *pool, uint32_t *batch_head)
{
    int thread_id = threadIdx.x;
    int b_id = blockIdx.x;
    int ngrid_span = bounds.ngrid_radius * 2;
    int xs_size = (L+1) * ngrid_span;
    int nf2 = (L+1)*(L+2)/2;
    int nf3 = nf2*(L+3)/3;
    pool += (xs_size*3 + nf3 + nf2*ngrid_span + 3) * WARP_SIZE * b_id;

    __shared__ uint32_t pair_idx0;
    if (thread_id == 0) {
        pair_idx0 = atomicAdd(batch_head, WARP_SIZE);
    }
    __syncthreads();
    while (pair_idx0 < bounds.nshl_pair) {
        _eval_rho_orth_kernel<L>(rho, dm, envs, bounds, pool, pair_idx0);
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
    size_t len1 = nf3 * WARP_SIZE;
    size_t len2 = (lj+1)*(lj+1) * 3 * WARP_SIZE;
    size_t len3 = (l+1) * ngrid_span * 2 + nf2 * ngrid_span;
    len2 = MAX(len2, len3);
    return MAX(len1, len2) * sizeof(double);
}

extern "C" {
int MG_eval_rho_orth(double *rho, double *dm, MGridEnvVars envs,
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
    case 0: eval_rho_orth_kernel<0> <<<workers, THREADS, buflen(0, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 1: eval_rho_orth_kernel<1> <<<workers, THREADS, buflen(1, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 2: eval_rho_orth_kernel<2> <<<workers, THREADS, buflen(2, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 3: eval_rho_orth_kernel<3> <<<workers, THREADS, buflen(3, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 4: eval_rho_orth_kernel<4> <<<workers, THREADS, buflen(4, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 5: eval_rho_orth_kernel<5> <<<workers, THREADS, buflen(5, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 6: eval_rho_orth_kernel<6> <<<workers, THREADS, buflen(6, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 7: eval_rho_orth_kernel<7> <<<workers, THREADS, buflen(7, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    case 8: eval_rho_orth_kernel<8> <<<workers, THREADS, buflen(8, &bounds)>>>(rho, dm, envs, bounds, pool, batch_head); break;
    default: return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MG_eval_rho_orth: %s\n", cudaGetErrorString(err));
        cudaFree(batch_head);
        return 1;
    }
    cudaFree(batch_head);
    return 0;
}
}
