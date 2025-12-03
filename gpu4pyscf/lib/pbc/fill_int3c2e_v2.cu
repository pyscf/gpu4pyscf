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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"
#include "int3c2e.cuh"
#include "int3c2e_create_tasks.cuh"

#define GOUT_WIDTH      54
#define BLOCK_SIZE      16

__device__ __forceinline__ unsigned get_smid() {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// lattice sum over j and k for (ij|k)
__global__ static
void pbc_int3c2e_latsum23_kernel(double *out, PBCIntEnvVars envs, PBCInt3c2eBounds bounds,
                                 ImgIdxPage *page_pool, float *diffuse_exps,
                                 float *diffuse_coefs, float log_cutoff)
{
    int nsp_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int sp_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;

    int ncells = envs.bvk_ncells;
    int nksh = bounds.nksh;
    int bvk_nksh = nksh * ncells;
    int nbas = envs.cell0_nbas * ncells;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int lij = li + lj;
    int nroots = bounds.nroots;
    int nfi = bounds.nfi;
    int nfj = bounds.nfj;
    int nfk = bounds.nfk;
    int nfij = nfi * nfj;
    int nf = nfij * nfk;
    int kprim = bounds.kprim;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int g_size = bounds.g_size;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lk);
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];

    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    double gout[GOUT_WIDTH];

    page_pool += get_smid() * PAGES_PER_BLOCK;
    __shared__ int num_pages;
    if (thread_id == 0) {
        num_pages = 0;
    }
    __syncthreads();
    int pair = sp_block_id * BLOCK_SIZE + thread_id / BLOCK_SIZE;
    int ksh = ksh_block_id * BLOCK_SIZE + thread_id % BLOCK_SIZE;
    if (pair < bounds.n_prim_pairs && ksh < bvk_nksh) {
        int kcell = ksh / nksh;
        int ksh_in_cell0 = ksh % nksh;
        int _ksh = kcell * bounds.nbas_aux + ksh_in_cell0 + bounds.ksh0;
        _filter_images(num_pages, page_pool, envs, pair, _ksh,
                       bounds.li, bounds.lj, bounds.bas_ij_idx, bounds.img_idx,
                       bounds.img_offsets, diffuse_exps, diffuse_coefs, log_cutoff);
    }
    __syncthreads();
    if (num_pages >= PAGES_PER_BLOCK) {
        __trap();
    }
    __shared__ int img_max;
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

        constexpr int gout_start = 0;
        //for (int gout_start = 0; gout_start < nfij*nfk; gout_start+=gout_stride*GOUT_WIDTH) {
            __syncthreads();
            int bas_ij = bounds.bas_ij_idx[page->pair_ij];
            int ish = bas_ij / nbas;
            int jsh = bas_ij % nbas;
            int ksh = page->ksh;
            double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
            double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
            double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
            double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double cicj = PI_FAC * ci * cj;
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

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
                    double omega2 = omega * omega;
                    double theta_fac = omega2 / (omega2 + theta);
                    double theta_rr = theta * Rpq[3*nsp_per_block];
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsp_per_block,
                              nsp_per_block, gout_id, gout_stride);
                    rys_roots(_nroots, theta_fac*theta_rr, rw,
                              nsp_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[ irys*2   *nsp_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsp_per_block] *= sqrt_theta_fac;
                    }
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
                                //for i in range(lij+1):
                                //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                                if (n < lij3) {
                                    s0x = _gx[0];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[-nsp_per_block];
                                    }
                                    _gx[stride_k*nsp_per_block] = s1x;
                                }
                                //for k in range(1, lk):
                                //    for i in range(lij+1):
                                //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
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
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                int ijk = gout_start + n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                int k  = ijk / nfij;
                                int ij = ijk % nfij;
                                int i = ij % nfi;
                                int j = ij / nfi;
                                int ix = idx_i[i*3+0];
                                int iy = idx_i[i*3+1];
                                int iz = idx_i[i*3+2];
                                int jx = idx_j[j*3+0];
                                int jy = idx_j[j*3+1];
                                int jz = idx_j[j*3+2];
                                int kx = idx_k[k*3+0];
                                int ky = idx_k[k*3+1];
                                int kz = idx_k[k*3+2];
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nsp_per_block;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nsp_per_block;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nsp_per_block;
                                gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                            }
                        }
                    }
                }
            }
            if (page_id < num_pages) {
                int *pair_mapping = bounds.pair_mapping;
                int nbas_aux = bounds.nbas_aux;
                size_t _bvk_nksh = bvk_nksh;
                size_t stride = bounds.n_ctr_pairs * _bvk_nksh;
                int _ksh = ksh - bounds.ksh0;
                int kcell = _ksh / nbas_aux;
                int ksh_in_cell0 = _ksh % nbas_aux;
                // store as [nfk,nfj,nfi,ijsh,Nimgs,nksh]
                double *eri_tensor = out + _bvk_nksh *
                    pair_mapping[page->pair_ij] + kcell*nksh + ksh_in_cell0;
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijk = gout_start + n*gout_stride+gout_id;
                    if (ijk >= nf) break;
                    atomicAdd(eri_tensor + ijk*stride, gout[n]);
                }
            }
        //}
    }
}

extern "C" {
int PBCsr_int3c2e_latsum23(double *out, PBCIntEnvVars *envs, ImgIdxPage *pool,
                           int *scheme, int *shls_slice, int nbas_aux,
                           int n_prim_pairs, int n_ctr_pairs,
                           uint32_t *bas_ij_idx, int *pair_mapping,
                           int *img_idx, uint32_t *img_offsets,
                           float *diffuse_exps, float *diffuse_coefs, float log_cutoff,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4] + nbas;
    int ksh1 = shls_slice[5] + nbas;
    int nksh = ksh1 - ksh0;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int naux = nfk * nksh;
    int order = li + lj + lk;
    int nroots = order / 2 + 1;
    nroots *= 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    // up to (gg|i)
    int g_size = stride_k * (lk + 1);
    PBCInt3c2eBounds bounds = {
        li, lj, lk, nroots, nfi, nfj, nfk, kprim,
        stride_j, stride_k, g_size, nbas_aux, nksh, ksh0, naux,
        n_prim_pairs, n_ctr_pairs,
        bas_ij_idx, pair_mapping, img_offsets, img_idx
    };
    int ncells = envs->bvk_ncells;

    if (1) {
        int nsp_per_block = scheme[0];
        int gout_stride = scheme[1];
        dim3 threads(nsp_per_block, gout_stride);
        int sp_blocks = (n_prim_pairs + BLOCK_SIZE-1) / BLOCK_SIZE;
        int ksh_blocks = (ncells*nksh + BLOCK_SIZE-1) / BLOCK_SIZE;
        dim3 blocks(sp_blocks, ksh_blocks);
        int buflen = (nroots*2+g_size*3+7) * nsp_per_block * sizeof(double);
        pbc_int3c2e_latsum23_kernel<<<blocks, threads, buflen>>>(
                out, *envs, bounds, pool, diffuse_exps, diffuse_coefs, log_cutoff);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCsr_int3c2e_latsum23_init(int shm_size)
{
    cudaFuncSetAttribute(pbc_int3c2e_latsum23_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
