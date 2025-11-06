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
#include "pbc.cuh"
#include "int3c2e.cuh"

#define GOUT_WIDTH      54
#define PAGE_SIZE       30
#define REMOTE_THRESHOLD 50
#define PAGES_PER_BLOCK  1048576
// approximately, 15000 images in each ijk shell triplet for 256 threads
#define KTASKS_PER_BLOCK 8

typedef struct {
    int pair_ij;
    uint16_t ksh;
    uint16_t nimgs;
    uint16_t img_j[PAGE_SIZE];
    uint16_t img_k[PAGE_SIZE];
} ImgIdxPage;

__device__ __forceinline__ unsigned get_smid() {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__
void _filter_images(int& num_pages, // is stored in shm
                    ImgIdxPage *page_pool, PBCIntEnvVars &envs,
                    PBCInt3c2eBounds &bounds, int pair_ij,
                    int kcount0, int kcount1, float log_cutoff)
{
    int thread_xy = threadIdx.x + blockDim.x * threadIdx.y;
    int threads_xy = blockDim.x * blockDim.y;
    int thread_id = thread_xy + threads_xy * threadIdx.z;
    if (thread_id == 0) {
        num_pages = 0;
    }
    __syncthreads();
    if (pair_ij < bounds.n_prim_pairs) {
        int nimgs = envs.nimgs;
        int *bas = envs.bas;
        double *env = envs.env;
        double *img_coords = envs.img_coords;
        int li = bounds.li;
        int lj = bounds.lj;
        int kprim = bounds.kprim;
        int nksh = bounds.nksh;
        int nbas_aux = bounds.naux;
        int ksh0 = bounds.ksh0;
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        uint32_t *sp_img_offsets = bounds.img_offsets;
        uint32_t img0 = sp_img_offsets[pair_ij];
        int nimgs_j = sp_img_offsets[pair_ij+1] - img0;
        int *ovlp_img_idx = bounds.img_idx + img0;
        float ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        float aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        float aij = ai + aj;
        float ai_aij = ai / aij;
        float aj_aij = aj / aij;
        float u = .5f / aij;
        float theta_ij = ai * aj / aij;
        float omega = env[PTR_RANGE_OMEGA];
        if (omega == 0) {
            omega = 0.1f;
        }
        float omega2 = omega * omega;
        float omega_aij = omega2 / (omega2 + aij);
        // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
        //           ~ between [0, 2]
        float fac_guess = .5f - logf(omega2)/4;
        // log(ci*cj * (pi/aij)**1.5)
        float log_fac = logf(fabsf(ci*cj)) + 1.717f - 1.5f*logf(aij) + fac_guess;
        // An addiitonal factor for Coulomb integrals
        // log_fac += .25 * logf(2./pi * aij)
        log_fac += .25f * logf(0.6366f * aij);
        log_cutoff -= log_fac;
        float xi = ri[0];
        float yi = ri[1];
        float zi = ri[2];
        float xj = rj[0];
        float yj = rj[1];
        float zj = rj[2];
        float xjxi = xj - xi;
        float yjyi = yj - yi;
        float zjzi = zj - zi;

        for (int kcount = kcount0+thread_xy; kcount < kcount1; kcount += threads_xy) {
            int kcell = kcount / nksh;
            int ksh_in_cell0 = kcount % nksh;
            int ksh = kcell * nbas_aux + ksh_in_cell0 + ksh0;
            float ak = env[bas[ksh*BAS_SLOTS+PTR_EXP] + kprim-1];
            float aij_ak = aij * ak;
            float theta = aij_ak * omega2 / (aij_ak + (aij + ak) * omega2);
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            float xk = rk[0];
            float yk = rk[1];
            float zk = rk[2];
            float xixk = xi - xk;
            float yiyk = yi - yk;
            float zizk = zi - zk;
            ImgIdxPage *page = NULL;
            int counts = PAGE_SIZE;
            for (int img = 0; img < nimgs_j*nimgs; ++img) {
                int jL = ovlp_img_idx[img / nimgs];
                int kL = img % nimgs;
                float xixkL = xixk - img_coords[kL*3+0];
                float yiykL = yiyk - img_coords[kL*3+1];
                float zizkL = zizk - img_coords[kL*3+2];
                float xjLxi = xjxi + img_coords[jL*3+0];
                float yjLyi = yjyi + img_coords[jL*3+1];
                float zjLzi = zjzi + img_coords[jL*3+2];
                float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                float theta_ij_rr = theta_ij * rr_ij;
                float xijk = xjLxi * aj_aij + xixkL;
                float yijk = yjLyi * aj_aij + yiykL;
                float zijk = zjLzi * aj_aij + zizkL;
                float rr = xijk * xijk + yijk * yijk + zijk * zijk;
                float theta_rr = theta * rr + theta_ij_rr;
                if (theta_rr > REMOTE_THRESHOLD) {
                    continue;
                }

                float rt_aij = omega_aij * sqrtf(rr);
                float dr = sqrtf(rr_ij);
                float dri = aj_aij * dr + rt_aij;
                float drj = ai_aij * dr + rt_aij;
                float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
                float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
                // TODO: an approx dri_fac and drj_fac
                float estimator = dri_fac + drj_fac - theta_rr;
                if (estimator > log_cutoff) {
                    if (counts == PAGE_SIZE) {
                        if (page != NULL) {
                            page->nimgs = PAGE_SIZE;
                        }
                        int page_offset = atomicAdd(&num_pages, 1);
                        if (page_offset >= PAGES_PER_BLOCK) {
                            printf("Page overflow\n");
                            __trap();
                        }
                        page = page_pool + page_offset;
                        page->pair_ij = pair_ij;
                        page->ksh = ksh;
                        counts = 0;
                    }
                    page->img_j[counts] = jL;
                    page->img_k[counts] = kL;
                    counts++;
                }
            }
            if (page != NULL) {
                page->nimgs = counts;
            }
        }
    }
}

// lattice sum over j and k for (ij|k)
__global__
void pbc_int3c2e_latsum23_kernel(double *out, PBCIntEnvVars envs, PBCInt3c2eBounds bounds,
                                 ImgIdxPage *page_pool, float log_cutoff)
{
    int k_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int nsp_per_block = blockDim.z;
    int k_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int thread_xz = threadIdx.x + blockDim.x * threadIdx.z;
    int threads_xz = blockDim.x * blockDim.z;
    int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    int ncells = envs.bvk_ncells;
    int nksp_per_block = k_per_block * nsp_per_block;
    int ksp_id = k_per_block * sp_id + k_id;
    int sp0_this_block = sp_block_id * nsp_per_block;
    int nksh_per_block = k_per_block * KTASKS_PER_BLOCK;
    int kcount0 = ksh_block_id * nksh_per_block;
    int kcount1 = min(ncells*bounds.nksh, kcount0 + nksh_per_block);

    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int lij = li + lj;
    int nroots = bounds.nroots;
    int nfij = bounds.nfij;
    int nfk = bounds.nfk;
    int nf = nfij * nfk;
    int kprim = bounds.kprim;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int g_size = bounds.g_size;
    int16_t *idx_ij = c_pair_idx + c_pair_offsets[li*L_AUX1+lj];
    int16_t *idy_ij = idx_ij + nfij;
    int16_t *idz_ij = idy_ij + nfij;
    int16_t *idx_k = c_pair_idx + c_pair_offsets[lk];
    int16_t *idy_k = idx_k + nfk;
    int16_t *idz_k = idy_k + nfk;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];

    int gx_len = g_size * nksp_per_block;
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + ksp_id;
    double *g = rw + nksp_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    double *rjri = gz + gx_len;
    double *Rpq = rjri + nksp_per_block * 3;
    double gout[GOUT_WIDTH];

    page_pool += get_smid() * PAGES_PER_BLOCK;
    int pair_ij = sp_id + sp0_this_block;
    int nbas = envs.cell0_nbas * ncells;
    __shared__ int num_pages;
    _filter_images(num_pages, page_pool, envs, bounds, pair_ij,
                   kcount0, kcount1, log_cutoff);
    __syncthreads();
    if (num_pages >= PAGES_PER_BLOCK) {
        __trap();
    }
    __shared__ int img_max;
    for (int page_id = thread_xz; page_id < num_pages+thread_xz; page_id += threads_xz) { 
        __syncthreads();
        ImgIdxPage *page = page_pool + page_id;
        if (page_id >= num_pages) {
            page = page_pool;
        }

        if (thread_id == 0) {
            img_max = 0;
        }
        __syncthreads();
        int img_counts = page->nimgs;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            img_counts = max(img_counts, __shfl_down_sync(0xffffffff, img_counts, offset));
        }
        if (thread_id % warpSize == 0 && gout_id == 0) {
            atomicMax(&img_max, img_counts);
        }

        for (int gout_start = 0; gout_start < nfij*nfk; gout_start+=gout_stride*GOUT_WIDTH) {
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
            double cicj = ci * cj;
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                gy[0] = PI_FAC * cicj;
            }
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            int img_counts = page->nimgs;
            for (int img = 0; img < img_max; ++img) {
                __syncthreads();
                int img_j = 0;
                int img_k = 0;
                if (page_id < num_pages && img < img_counts) {
                    img_j = page->img_j[img];
                    img_k = page->img_k[img];
                } else if (gout_id == 0) {
                    gy[0] = 0.;
                }
                double xjL = img_coords[img_j*3+0];
                double yjL = img_coords[img_j*3+1];
                double zjL = img_coords[img_j*3+2];
                double xjxi = rj[0] + xjL - ri[0];
                double yjyi = rj[1] + yjL - ri[1];
                double zjzi = rj[2] + zjL - ri[2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double Kab = theta_ij * rr_ij;
                double fac_ij = exp(-Kab);
                if (gout_id == 0) {
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
                    rjri[0*nksp_per_block] = xjxi;
                    rjri[1*nksp_per_block] = yjyi;
                    rjri[2*nksp_per_block] = zjzi;
                    Rpq[0*nksp_per_block] = xpq;
                    Rpq[1*nksp_per_block] = ypq;
                    Rpq[2*nksp_per_block] = zpq;
                    Rpq[3*nksp_per_block] = rr;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        double cijk = fac_ij * ck[kp];
                        gx[0] = cijk / (aij*ak*sqrt(aij+ak));
                    }
                    double omega2 = omega * omega;
                    double theta_fac = omega2 / (omega2 + theta);
                    double theta_rr = theta * Rpq[3*nksp_per_block];
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nksp_per_block,
                              nksp_per_block, gout_id, gout_stride);
                    rys_roots(_nroots, theta_fac*theta_rr, rw,
                              nksp_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[ irys*2   *nksp_per_block] *= theta_fac;
                        rw[(irys*2+1)*nksp_per_block] *= sqrt_theta_fac;
                    }
                    double s0x, s1x, s2x;
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gz[0] = rw[(irys*2+1)*nksp_per_block];
                        }
                        double rt = rw[ irys*2   *nksp_per_block];
                        double rt_aa = rt / (aij + ak);

                        if (lij > 0) {
                            __syncthreads();
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * gx_len;
                                double xpa = rjri[n*nksp_per_block] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n*nksp_per_block];
                                s0x = _gx[0];
                                s1x = c0x * s0x;
                                _gx[nksp_per_block] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    s2x = c0x * s1x + i * b10 * s0x;
                                    _gx[(i+1)*nksp_per_block] = s2x;
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
                                double *_gx = gx + (i + _ix * g_size) * nksp_per_block;
                                double cpx = rt_ak * Rpq[_ix*nksp_per_block];
                                //for i in range(lij+1):
                                //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                                if (n < lij3) {
                                    s0x = _gx[0];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[-nksp_per_block];
                                    }
                                    _gx[stride_k*nksp_per_block] = s1x;
                                }
                                //for k in range(1, lk):
                                //    for i in range(lij+1):
                                //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                                for (int k = 1; k < lk; ++k) {
                                    __syncthreads();
                                    if (n < lij3) {
                                        s2x = cpx*s1x + k*b01*s0x;
                                        if (i > 0) {
                                            s2x += i * b00 * _gx[(k*stride_k-1)*nksp_per_block];
                                        }
                                        _gx[(k*stride_k+stride_k)*nksp_per_block] = s2x;
                                        s0x = s1x;
                                        s1x = s2x;
                                    }
                                }
                            }
                        }

                        // hrr
                        // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                        // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                        if (lj > 0) {
                            __syncthreads();
                            if (page_id < num_pages) {
                                int lk3 = (lk+1)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nksp_per_block];
                                    double *_gx = g + (_ix*g_size + k*stride_k) * nksp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nksp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nksp_per_block];
                                            _gx[(ij+stride_j)*nksp_per_block] = s1x - xjxi * s0x;
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
                                int addrx = (idx_ij[ij] + idx_k[k] * stride_k) * nksp_per_block;
                                int addry = (idy_ij[ij] + idy_k[k] * stride_k) * nksp_per_block;
                                int addrz = (idz_ij[ij] + idz_k[k] * stride_k) * nksp_per_block;
                                gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                            }
                        }
                    }
                }
            }
            if (page_id < num_pages) {
                int *pair_mapping = bounds.pair_mapping;
                int nbas_aux = bounds.naux;
                int nksh = bounds.nksh;
                int bvk_nksh = nksh * ncells;
                size_t stride = bounds.n_ctr_pairs * bvk_nksh;
                int _ksh = ksh - bounds.ksh0;
                int kcell = _ksh / nbas_aux;
                int ksh_in_cell0 = _ksh % nbas_aux;
                // store as [nfk,nfj,nfi,ijsh,Nimgs,nksh]
                double *eri_tensor = out + bvk_nksh *
                    pair_mapping[page->pair_ij] + kcell*nksh + ksh_in_cell0;
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijk = gout_start + n*gout_stride+gout_id;
                    if (ijk >= nf) break;
                    atomicAdd(eri_tensor + ijk*stride, gout[n]);
                }
            }
        }
    }
}

extern "C" {
int PBCsr_int3c2e_latsum23(double *out, PBCIntEnvVars *envs, ImgIdxPage *pool,
                           int *scheme, int *shls_slice, int nbas_aux,
                           int n_prim_pairs, int n_ctr_pairs,
                           int *bas_ij_idx, int *pair_mapping,
                           int *img_idx, uint32_t *img_offsets, float log_cutoff,
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
    int nfij = nfi * nfj;
    int order = li + lj + lk;
    int nroots = order / 2 + 1;
    nroots *= 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    // up to (gg|i)
    int g_size = stride_k * (lk + 1);
    PBCInt3c2eBounds bounds = {
        li, lj, lk, nroots, nfij, nfk, kprim,
        stride_j, stride_k, g_size, nbas_aux, nksh, ksh0,
        n_prim_pairs, n_ctr_pairs,
        bas_ij_idx, pair_mapping, img_offsets, img_idx
    };
    int ncells = envs->bvk_ncells;

    if (1) {
        int nksh_per_block = scheme[0];
        int gout_stride = scheme[1];
        int nsp_per_block = scheme[2];
        dim3 threads(nksh_per_block, gout_stride, nsp_per_block);
        int sp_blocks = (n_prim_pairs + nsp_per_block - 1) / nsp_per_block;
        int ksh_blocks = (ncells*nksh + KTASKS_PER_BLOCK*nksh_per_block - 1)
            / (KTASKS_PER_BLOCK*nksh_per_block);
        dim3 blocks(sp_blocks, ksh_blocks);
        int buflen = (nroots*2+g_size*3+7) * (nksh_per_block * nsp_per_block) * sizeof(double);
        pbc_int3c2e_latsum23_kernel<<<blocks, threads, buflen>>>(out, *envs, bounds, pool, log_cutoff);
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
