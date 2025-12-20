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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"
#include "int3c2e.cuh"
#include "int3c2e_create_tasks.cuh"

#define LMAX            4
#define LMAX1           (LMAX+1)
#define NF_AUX_MAX      28
#define GOUT_WIDTH      30

// lattice sum over j and k for (ij|k)
__global__ static
void contract_int3c2e_dm_kernel(double *out, double *dm,
                                PBCIntEnvVars envs, ImgIdxPage *page_pool,
                                int *shl_pair_offsets, uint32_t *bas_ij_idx,
                                int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                                float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int sp_block_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int ncells = envs.bvk_ncells;
    int nbas = envs.cell0_nbas * ncells;
    int ksh = blockIdx.x + nbas;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, lij, nroots;
    __shared__ int lk, kprim;
    __shared__ int nao;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish = bas_ij0 / nbas;
        int jsh = bas_ij0 % nbas;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = ao_loc[envs.cell0_nbas];
    }
    __syncthreads();
    int gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    int nfk = (lk + 1) * (lk + 2) / 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);
    double vj_xyz[NF_AUX_MAX];
    for (int n = 0; n < NF_AUX_MAX; ++n) {
        vj_xyz[n] = 0;
    }
    page_pool += get_smid() * PAGES_PER_BLOCK;

    __shared__ double xk, yk, zk;
    __shared__ double omega;
    __shared__ double *expk, *ck;
    if (thread_id == 0) {
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        xk = rk[0];
        yk = rk[1];
        zk = rk[2];
        omega = env[PTR_RANGE_OMEGA];
        expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
    }
    if (gout_id == 0) {
        rjri[0*nsp_per_block] = 0;
        rjri[1*nsp_per_block] = 0;
        rjri[2*nsp_per_block] = 0;
        Rpq[0*nsp_per_block] = 0;
        Rpq[1*nsp_per_block] = 0;
        Rpq[2*nsp_per_block] = 0;
        // Rpq[3] must be initialized. An uninitialized Rpq[3] might be nan,
        // which would cause illegal addresses in the rys_roots function
        Rpq[3*nsp_per_block] = 0;
    }

    __shared__ int num_pages, img_max;

    for (int pair_ij = shl_pair0+thread_id; pair_ij < shl_pair1+thread_id; pair_ij += THREADS) {
        __syncthreads();
        if (thread_id == 0) {
            num_pages = 0;
        }
        __syncthreads();
        if (pair_ij < shl_pair1) {
            _filter_images(num_pages, page_pool, envs, pair_ij, ksh,
                           ksh, li, lj, bas_ij_idx, img_idx, img_offsets,
                           diffuse_exps, diffuse_coefs, log_cutoff);
        }
        __syncthreads();
        if (num_pages >= PAGES_PER_BLOCK) {
            __trap();
        }

        for (int page_id = sp_id; page_id < num_pages+sp_id; page_id += nsp_per_block) {
            __syncthreads();
            ImgIdxPage *page = page_pool + page_id;
            if (page_id >= num_pages) {
                page = page_pool;
            }
            if (thread_id == 0) {
                img_max = 0;
            }
            __syncthreads();
            {
                int img_counts = page->nimgs;
                for (int offset = warpSize/2; offset > 0; offset /= 2) {
                    img_counts = max(img_counts, __shfl_down_sync(0xffffffff, img_counts, offset));
                }
                if (thread_id % warpSize == 0 && gout_id == 0) {
                    atomicMax(&img_max, img_counts);
                }
            }
            __syncthreads();
            // move ish, jsh, ai, aj, ci, cj, ri, rj in shm
            int bas_ij = bas_ij_idx[page->pair_ij];
            int ish = bas_ij / nbas;
            int jsh = bas_ij % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jsh];
            double *dm_local = dm + j0 * nao + i0;
            double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
            double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
            double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
            double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double cicj = PI_FAC * ci * cj;
            int ish_cell0 = ish;
            int jsh_cell0 = jsh % envs.cell0_nbas;
            if (ish_cell0 == jsh_cell0) {
                cicj *= .5;
            } else if (ish_cell0 < jsh_cell0) {
                cicj = 0;
            }

            int img_counts = page->nimgs;
            for (int img = 0; img < img_max; ++img) {
                __syncthreads();
                if (gout_id == 0) {
                    if (page_id < num_pages && img < img_counts) {
                        int img_j = page->img_j[img];
                        int img_k = page->img_k[img];
                        double xjL = img_coords[img_j*3+0];
                        double yjL = img_coords[img_j*3+1];
                        double zjL = img_coords[img_j*3+2];
                        double xjxi = rj[0] + xjL - ri[0];
                        double yjyi = rj[1] + yjL - ri[1];
                        double zjzi = rj[2] + zjL - ri[2];
                        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                        double Kab = theta_ij * rr_ij;
                        double fac_ij = exp(-Kab);
                        double xij = xjxi * aj_aij + ri[0];
                        double yij = yjyi * aj_aij + ri[1];
                        double zij = zjzi * aj_aij + ri[2];
                        double xk_L = xk + img_coords[img_k*3+0];
                        double yk_L = yk + img_coords[img_k*3+1];
                        double zk_L = zk + img_coords[img_k*3+2];
                        double xpq = xij - xk_L;
                        double ypq = yij - yk_L;
                        double zpq = zij - zk_L;
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rjri[0*nsp_per_block] = xjxi;
                        rjri[1*nsp_per_block] = yjyi;
                        rjri[2*nsp_per_block] = zjzi;
                        Rpq[0*nsp_per_block] = xpq;
                        Rpq[1*nsp_per_block] = ypq;
                        Rpq[2*nsp_per_block] = zpq;
                        Rpq[3*nsp_per_block] = rr;
                        gx[gx_len] = cicj * fac_ij;
                    } else {
                        gx[gx_len] = 0.;
                    }
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, Rpq[3*nsp_per_block], omega,
                                 rw, nsp_per_block, gout_id, gout_stride);
                    double s0x, s1x, s2x;
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                        }
                        double rt = rw[ irys*2   *nsp_per_block];
                        double rt_aa = rt / (aij + ak);

                        if (lij > 0) {
                            __syncthreads();
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * gx_len;
                                double xpa = rjri[n*nsp_per_block] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n*nsp_per_block];
                                s0x = _gx[0];
                                s1x = c0x * s0x;
                                _gx[nsp_per_block] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    s2x = c0x * s1x + i * b10 * s0x;
                                    _gx[(i+1)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        if (lk > 0) {
                            int lij3 = (lij+1)*3;
                            double rt_ak  = rt_aa * aij;
                            double b00 = .5 * rt_aa;
                            double b01 = .5/ak  * (1 - rt_ak);
                            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                                __syncthreads();
                                int i = n / 3; //for i in range(lij+1):
                                int _ix = n % 3; // TODO: remove _ix for nroots > 2
                                double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                                double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                                if (n < lij3) {
                                    s0x = _gx[0];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[-nsp_per_block];
                                    }
                                    _gx[stride_k*nsp_per_block] = s1x;
                                }
                                for (int k = 1; k < lk; ++k) {
                                    __syncthreads();
                                    if (n < lij3) {
                                        s2x = cpx*s1x + k*b01*s0x;
                                        if (i > 0) {
                                            s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                        }
                                        _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                        s0x = s1x;
                                        s1x = s2x;
                                    }
                                }
                            }
                        }

                        // hrr
                        if (lj > 0) {
                            __syncthreads();
                            if (page_id < num_pages) {
                                int lk3 = (lk+1)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nsp_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nsp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nsp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nsp_per_block];
                                            _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (page_id < num_pages) {
                            for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                                int i = ij % nfi;
                                int j = ij / nfi;
                                double dm_ij = dm_local[j*nao+i];
                                int ix = idx_i[i*3+0];
                                int iy = idx_i[i*3+1];
                                int iz = idx_i[i*3+2];
                                int jx = idx_j[j*3+0];
                                int jy = idx_j[j*3+1];
                                int jz = idx_j[j*3+2];
                                int ij_addrx = (ix + jx*stride_j) * nsp_per_block;
                                int ij_addry = (iy + jy*stride_j + g_size) * nsp_per_block;
                                int ij_addrz = (iz + jz*stride_j + g_size*2) * nsp_per_block;
                                int stride_k_n = stride_k * nsp_per_block;
#pragma unroll
                                for (int k = 0; k < NF_AUX_MAX; ++k) {
                                    if (k >= nfk) break;
                                    int kx = idx_k[k*3+0];
                                    int ky = idx_k[k*3+1];
                                    int kz = idx_k[k*3+2];
                                    int addrx = ij_addrx + kx * stride_k_n;
                                    int addry = ij_addry + ky * stride_k_n;
                                    int addrz = ij_addrz + kz * stride_k_n;
                                    vj_xyz[k] += gx[addrx] * gx[addry] * gx[addrz] * dm_ij;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int k0 = ao_loc[ksh] - ao_loc[nbas];
    double *vj = out + k0;
#pragma unroll
    for (int k = 0; k < NF_AUX_MAX; ++k) {
        if (k >= nfk) break;
        atomicAdd(vj + k, vj_xyz[k]);
    }
}

__global__ static
void contract_int3c2e_auxvec_kernel(double *out, double *auxvec,
                                    PBCIntEnvVars envs, ImgIdxPage *page_pool,
                                    int *ksh_offsets, int *ksh_idx, uint32_t *bas_ij_idx,
                                    int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                                    float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int ksh_block_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int ncells = envs.bvk_ncells;
    int nbas = envs.cell0_nbas * ncells;
    int pair_ij = blockIdx.x;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int kidx0, kidx1;
    __shared__ int ish, jsh, li, lj, lij, nroots;
    __shared__ int lk, nfk, kprim;
    if (thread_id == 0) {
        kidx0 = ksh_offsets[ksh_block_id];
        kidx1 = ksh_offsets[ksh_block_id+1];
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ksh = ksh_idx[kidx0];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nfk = (lk + 1) * (lk + 2) / 2;
    }
    __syncthreads();
    int gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);
    double vj[GOUT_WIDTH];
    for (int n = 0; n < GOUT_WIDTH; ++n) {
        vj[n] = 0;
    }
    page_pool += get_smid() * PAGES_PER_BLOCK;

    __shared__ double aij, aj_aij, theta_ij, cicj;
    __shared__ double xi, yi, zi, xj, yj, zj;
    __shared__ double omega;
    if (thread_id == 0) {
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        xi = ri[0];
        yi = ri[1];
        zi = ri[2];
        xj = rj[0];
        yj = rj[1];
        zj = rj[2];
        aij = ai + aj;
        aj_aij = aj / aij;
        theta_ij = ai * aj_aij;
        cicj = PI_FAC * ci * cj;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.cell0_nbas;
        if (ish_cell0 == jsh_cell0) {
            cicj *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            cicj = 0;
        }
        omega = env[PTR_RANGE_OMEGA];
    }
    if (gout_id == 0) {
        rjri[0*nsp_per_block] = 0;
        rjri[1*nsp_per_block] = 0;
        rjri[2*nsp_per_block] = 0;
        Rpq[0*nsp_per_block] = 0;
        Rpq[1*nsp_per_block] = 0;
        Rpq[2*nsp_per_block] = 0;
        // Rpq[3] must be initialized. An uninitialized Rpq[3] might be nan,
        // which would cause illegal addresses in the rys_roots function
        Rpq[3*nsp_per_block] = 0;
    }

    __shared__ int num_pages, img_max;

    for (int kidx = kidx0+thread_id; kidx < kidx1+thread_id; kidx += THREADS) {
        __syncthreads();
        if (thread_id == 0) {
            num_pages = 0;
        }
        __syncthreads();
        if (kidx < kidx1) {
            _filter_images(num_pages, page_pool, envs, pair_ij, ksh_idx[kidx],
                           kidx, li, lj, bas_ij_idx, img_idx, img_offsets,
                           diffuse_exps, diffuse_coefs, log_cutoff);
        }
        __syncthreads();
        if (num_pages >= PAGES_PER_BLOCK) {
            __trap();
        }

        for (int page_id = sp_id; page_id < num_pages+sp_id; page_id += nsp_per_block) {
            __syncthreads();
            ImgIdxPage *page = page_pool + page_id;
            if (page_id >= num_pages) {
                page = page_pool;
            }
            if (thread_id == 0) {
                img_max = 0;
            }
            __syncthreads();
            {
                int img_counts = page->nimgs;
                for (int offset = warpSize/2; offset > 0; offset /= 2) {
                    img_counts = max(img_counts, __shfl_down_sync(0xffffffff, img_counts, offset));
                }
                if (thread_id % warpSize == 0 && gout_id == 0) {
                    atomicMax(&img_max, img_counts);
                }
            }
            __syncthreads();
            int ksh = ksh_idx[page->k];
            int k0 = ao_loc[ksh] - ao_loc[nbas];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            int img_counts = page->nimgs;
            for (int img = 0; img < img_max; ++img) {
                __syncthreads();
                if (gout_id == 0) {
                    if (page_id < num_pages && img < img_counts) {
                        int img_j = page->img_j[img];
                        int img_k = page->img_k[img];
                        double xjL = img_coords[img_j*3+0];
                        double yjL = img_coords[img_j*3+1];
                        double zjL = img_coords[img_j*3+2];
                        double xjxi = xj + xjL - xi;
                        double yjyi = yj + yjL - yi;
                        double zjzi = zj + zjL - zi;
                        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                        double Kab = theta_ij * rr_ij;
                        double fac_ij = exp(-Kab);
                        double xij = xjxi * aj_aij + xi;
                        double yij = yjyi * aj_aij + yi;
                        double zij = zjzi * aj_aij + zi;
                        double xk = rk[0] + img_coords[img_k*3+0];
                        double yk = rk[1] + img_coords[img_k*3+1];
                        double zk = rk[2] + img_coords[img_k*3+2];
                        double xpq = xij - xk;
                        double ypq = yij - yk;
                        double zpq = zij - zk;
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rjri[0*nsp_per_block] = xjxi;
                        rjri[1*nsp_per_block] = yjyi;
                        rjri[2*nsp_per_block] = zjzi;
                        Rpq[0*nsp_per_block] = xpq;
                        Rpq[1*nsp_per_block] = ypq;
                        Rpq[2*nsp_per_block] = zpq;
                        Rpq[3*nsp_per_block] = rr;
                        gx[gx_len] = cicj * fac_ij;
                    } else {
                        gx[gx_len] = 0.;
                    }
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, Rpq[3*nsp_per_block], omega,
                                 rw, nsp_per_block, gout_id, gout_stride);
                    double s0x, s1x, s2x;
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                        }
                        double rt = rw[ irys*2   *nsp_per_block];
                        double rt_aa = rt / (aij + ak);

                        if (lij > 0) {
                            __syncthreads();
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * gx_len;
                                double xpa = rjri[n*nsp_per_block] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n*nsp_per_block];
                                s0x = _gx[0];
                                s1x = c0x * s0x;
                                _gx[nsp_per_block] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    s2x = c0x * s1x + i * b10 * s0x;
                                    _gx[(i+1)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        if (lk > 0) {
                            int lij3 = (lij+1)*3;
                            double rt_ak  = rt_aa * aij;
                            double b00 = .5 * rt_aa;
                            double b01 = .5/ak  * (1 - rt_ak );
                            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                                __syncthreads();
                                int i = n / 3; //for i in range(lij+1):
                                int _ix = n % 3; // TODO: remove _ix for nroots > 2
                                double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                                double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                                if (n < lij3) {
                                    s0x = _gx[0];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[-nsp_per_block];
                                    }
                                    _gx[stride_k*nsp_per_block] = s1x;
                                }
                                for (int k = 1; k < lk; ++k) {
                                    __syncthreads();
                                    if (n < lij3) {
                                        s2x = cpx*s1x + k*b01*s0x;
                                        if (i > 0) {
                                            s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                        }
                                        _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                        s0x = s1x;
                                        s1x = s2x;
                                    }
                                }
                            }
                        }

                        // hrr
                        if (lj > 0) {
                            __syncthreads();
                            if (page_id < num_pages && img < img_counts) {
                                int lk3 = (lk+1)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nsp_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nsp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nsp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nsp_per_block];
                                            _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (page_id < num_pages && img < img_counts) {
                            for (int k = 0; k < nfk; ++k) {
                                int kx = idx_k[k*3+0] * stride_k;
                                int ky = idx_k[k*3+1] * stride_k + g_size;
                                int kz = idx_k[k*3+2] * stride_k + g_size*2;
                                double rho = auxvec[k0+k];
#pragma unroll
                                for (int n = 0; n < GOUT_WIDTH; n++) {
                                    int ij = n*gout_stride+gout_id;
                                    if (ij >= nfij) break;
                                    int i = ij % nfi;
                                    int j = ij / nfi;
                                    int ix = idx_i[i*3+0];
                                    int iy = idx_i[i*3+1];
                                    int iz = idx_i[i*3+2];
                                    int jx = idx_j[j*3+0];
                                    int jy = idx_j[j*3+1];
                                    int jz = idx_j[j*3+2];
                                    int addrx = (ix + jx*stride_j + kx) * nsp_per_block;
                                    int addry = (iy + jy*stride_j + ky) * nsp_per_block;
                                    int addrz = (iz + jz*stride_j + kz) * nsp_per_block;
                                    vj[n] += gx[addrx] * gx[addry] * gx[addrz] * rho;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int nao = ao_loc[nbas];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    double *vj_ij = out + i0 * nao + j0;
#pragma unroll
    for (int n = 0; n < GOUT_WIDTH; n++) {
        int ij = n*gout_stride+gout_id;
        if (ij >= nfij) break;
        int i = ij % nfi;
        int j = ij / nfi;
        atomicAdd(vj_ij + i*nao+j, vj[n]);
    }
}

extern "C" {
int PBCcontract_int3c2e_dm(double *out, double *dm,
                           PBCIntEnvVars *envs, ImgIdxPage *pool, int shm_size,
                           int nbas_aux, int nbatches_shl_pair, int *shl_pair_offsets,
                           uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                           int *gout_stride_lookup,
                           float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int ncells = envs->bvk_ncells;
    cudaFuncSetAttribute(contract_int3c2e_dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(ncells*nbas_aux, nbatches_shl_pair);
    contract_int3c2e_dm_kernel<<<blocks, THREADS, shm_size>>>(
            out, dm, *envs, pool, shl_pair_offsets, bas_ij_idx,
            img_idx, img_offsets, gout_stride_lookup,
            diffuse_exps, diffuse_coefs, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCcontract_int3c2e_auxvec(double *out, double *auxvec,
                               PBCIntEnvVars *envs, ImgIdxPage *pool, int shm_size,
                               int npairs, int nbatches_ksh, int *ksh_offsets, int *ksh_idx,
                               uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                               int *gout_stride_lookup,
                               float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(contract_int3c2e_auxvec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(npairs, nbatches_ksh);
    contract_int3c2e_auxvec_kernel<<<blocks, THREADS, shm_size>>>(
            out, auxvec, *envs, pool, ksh_offsets, ksh_idx, bas_ij_idx,
            img_idx, img_offsets, gout_stride_lookup,
            diffuse_exps, diffuse_coefs, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
