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

#include "gint/cuda_alloc.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "gvhf-rys/create_tasks.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"

#define GOUT_WIDTH1     81
#define PAGE_SIZE       36
#define REMOTE_THRESHOLD 50
#define PAGES_PER_BLOCK  524288
#define WARPS           8

typedef struct {
    int li;
    int lj;
    int lk;
    int ll;
    int nfi;
    int nfj;
    int nfk;
    int nfl;
    int nroots;
    int stride_j;
    int stride_k;
    int stride_l;
    int g_size;
    int iprim;
    int jprim;
    int kprim;
    int lprim;
    int npairs_ij;
    int npairs_kl;
    int *pair_ij_mapping;
    int *pair_kl_mapping;
    float *q_cond;
    float *s_estimator;
    float *dm_cond;
    float cutoff;
    int ntiles_i;
    int ntiles_j;
    int ntiles_k;
    int ntiles_l;
    uint32_t *img_offsets; // offset img_idx for each shell-pair
    int *img_idx; // indices of img_coords in each shell-pair
} PBCInt2eBounds;

typedef struct {
    int bas_kl;
    int nimgs;
    uint16_t img_j[PAGE_SIZE];
    uint16_t img_k[PAGE_SIZE];
    uint16_t img_kl[PAGE_SIZE];
} ImgIdxPage;

extern __constant__ int _c_cartesian_lexical_xyz[];
extern __constant__ GXYZOffset c_gxyz_offset[];

#define allocate_page() (page_pool + atomicAdd(&num_pages, 1))

__device__ __forceinline__
void _filter_images(int &num_pages, int &pair_kl0,
                    ImgIdxPage *page_pool, PBCIntEnvVars &envs,
                    PBCInt2eBounds &bounds)
{
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    int pair_ij = blockIdx.x;
    if (thread_id == 0) {
        num_pages = 0;
    }
    __syncthreads();
    int warp_id = thread_id / warpSize;
    int sp_id = thread_id % warpSize;
    int nimgs = envs.nimgs;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int bas_ij = bounds.pair_ij_mapping[pair_ij];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
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
    float theta_ij = ai * aj_aij;
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
    float log_cutoff = bounds.cutoff - log_fac;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;

    ImgIdxPage *page = NULL;
    while (num_pages + warpSize * 240 < PAGES_PER_BLOCK // 240 pages ~= 8000 images
           && pair_kl0 < bounds.npairs_kl) {
        int pair_kl = pair_kl0 + sp_id;
        if (pair_kl < bounds.npairs_kl) {
            int bas_kl = bounds.pair_kl_mapping[pair_kl];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            uint32_t *sp_img_offsets = bounds.img_offsets;
            uint32_t img0 = sp_img_offsets[pair_kl];
            int nimgs_kl = sp_img_offsets[pair_kl+1] - img0;
            int *ovlp_img_idx = bounds.img_idx + img0;
            int nimgs_j = sp_img_offsets[pair_ij];

            float ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];
            float al = env[bas[lsh*BAS_SLOTS+PTR_EXP]];
            float akl = ak + al;
            float ak_akl = ak / akl;
            float al_akl = al / akl;
            float theta_kl = ak * al_akl;
            float omega_akl = omega2 / (omega2 + akl);
            float theta = aij * akl * omega2 / (aij * akl + (aij + akl) * omega2);
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            float xk = rk[0];
            float yk = rk[1];
            float zk = rk[2];
            float xl = rl[0];
            float yl = rl[1];
            float zl = rl[2];
            float xlxk0 = xl - xk;
            float ylyk0 = yl - yk;
            float zlzk0 = zl - zk;
            int counts = PAGE_SIZE;
            for (int img = warp_id; img < nimgs_kl*nimgs_j; img+=WARPS) {
                int jL = ovlp_img_idx[img / nimgs_kl];
                int img_kl = ovlp_img_idx[img % nimgs_kl];
                int kL = img_kl / nimgs;
                int klL = img_kl % nimgs; // klL = lL - kL
                float xlLxk = xlxk0 + img_coords[klL*3+0];
                float ylLyk = ylyk0 + img_coords[klL*3+1];
                float zlLzk = zlzk0 + img_coords[klL*3+2];
                float xjLxi = xjxi + img_coords[jL*3+0];
                float yjLyi = yjyi + img_coords[jL*3+1];
                float zjLzi = zjzi + img_coords[jL*3+2];
                float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                float theta_ij_rr = theta_ij * rr_ij;
                float rr_kl = xlLxk * xlLxk + ylLyk * ylLyk + zlLzk * zlLzk;
                float theta_kl_rr = theta_kl * rr_kl;
                float xpq = xjLxi * aj_aij - xlLxk * al_akl;
                float ypq = yjLyi * aj_aij - ylLyk * al_akl;
                float zpq = zjLzi * aj_aij - zlLzk * al_akl;
                float rr = xpq * xpq + ypq * ypq + zpq * zpq;
                float theta_rr = theta * rr + theta_ij_rr + theta_kl_rr;
                if (theta_rr > REMOTE_THRESHOLD) {
                    continue;
                }

                float r = sqrtf(rr);
                float rt_aij = omega_aij * r;
                float rt_akl = omega_akl * r;
                float r_ij = sqrtf(rr_ij);
                float r_kl = sqrtf(rr_kl);
                float dri = aj_aij * r_ij + rt_aij;
                float drj = ai_aij * r_ij + rt_aij;
                float drk = al_akl * r_kl + rt_akl;
                float drl = ak_akl * r_kl + rt_akl;
                // TODO: an approx dr_fac
                float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
                float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
                float drk_fac = .5f*lk * logf(drk*drk + lk*u + 1e-9f);
                float drl_fac = .5f*ll * logf(drl*drl + ll*u + 1e-9f);
                float estimator = dri_fac + drj_fac + drk_fac + drl_fac - theta_rr;
                if (estimator > log_cutoff) {
                    if (counts == PAGE_SIZE) {
                        if (page != NULL) {
                            page->nimgs = PAGE_SIZE;
                        }
                        page = allocate_page();
                        page->bas_kl = bas_kl;
                        counts = 0;
                    }
                    page->img_j[counts] = jL;
                    page->img_k[counts] = kL;
                    page->img_kl[counts] = klL;
                    counts++;
                }
            }
            if (page != NULL) {
                page->nimgs = counts;
            }
            if (thread_id == 0) {
                pair_kl0 += threads;
            }
        }
        __syncthreads();
    }
}

// gout_pattern = ((li == 0) >> 3) | ((lj == 0) >> 2) | ((lk == 0) >> 1) | (ll == 0);
__global__ static
void rys_k_kernel(PBCIntEnvVars envs, JKMatrix kmat, PBCInt2eBounds bounds,
                  ImgIdxPage *pool, GXYZOffset *gxyz_offsets,
                  int gout_pattern, int reserved_shm_size)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    int smid = get_smid();
    ImgIdxPage *page_pool = pool + smid * QUEUE_DEPTH;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = bounds.g_size;

    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *rlLrk = shared_memory + nsq_per_block * 3 + sq_id;
    double *rjLri = shared_memory + nsq_per_block * 6 + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 9 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 12 + sq_id;
    double *gx = shared_memory + nsq_per_block * 14 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+14) + sq_id;
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int *idx_i = (int*)(shared_memory + reserved_shm_size);
    int *idx_j = idx_i + ntiles_i * 9;
    int *idx_k = idx_j + ntiles_j * 9;
    int *idx_l = idx_k + ntiles_k * 9;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    if (t_id < ntiles_i * 9) {
        idx_i[t_id] = lex_xyz_address(li, t_id) * nsq_per_block;
        idx_i[t_id] += (t_id % 3) * nsq_per_block * g_size;
    }
    if (t_id < ntiles_j * 9) {
        idx_j[t_id] = lex_xyz_address(lj, t_id) * stride_j * nsq_per_block;
    }
    if (t_id < ntiles_k * 9) {
        idx_k[t_id] = lex_xyz_address(lk, t_id) * stride_k * nsq_per_block;
    }
    if (t_id < ntiles_l * 9) {
        idx_l[t_id] = lex_xyz_address(ll, t_id) * stride_l * nsq_per_block;
    }

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double ai;
    __shared__ double aj;
    __shared__ double cicj;
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[0] = env[ri_ptr+0];
        ri[1] = env[ri_ptr+1];
        ri[2] = env[ri_ptr+2];
        double xjxi = env[rj_ptr+0] - ri[0];
        double yjyi = env[rj_ptr+1] - ri[1];
        double zjzi = env[rj_ptr+2] - ri[2];
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double theta_ij = ai * aj_aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj = ci * cj * Kab;
    }
    __shared__ int num_pages, pair_kl0, img_max;
    if (thread_id == 0) {
        pair_kl0 = 0;
    }
    while (pair_kl0 < bounds.npairs_kl) {
        _filter_images(num_pages, pair_kl0, page_pool, envs, bounds);
        for (int page_id = sq_id; page_id < num_pages+nsq_per_block; page_id += nsq_per_block) { 
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

            int nbas = envs.cell0_nbas * envs.bvk_ncells;
            int *bas = envs.bas;
            double *env = envs.env;
            double *img_coords = envs.img_coords;
            int li = bounds.li;
            int lj = bounds.lj;
            int lk = bounds.lk;
            int ll = bounds.ll;
            int stride_j = bounds.stride_j;
            int stride_k = bounds.stride_k;
            int stride_l = bounds.stride_l;
            int g_size = bounds.g_size;

            int bas_kl = page->bas_kl;
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            double fac_sym = PI_FAC;
            if (page_id < num_pages) {
                if (ish == jsh) fac_sym *= .5;
                if (ksh == lsh) fac_sym *= .5;
                if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];
                double al = env[bas[lsh*BAS_SLOTS+PTR_EXP]];
                double aij = ai + aj;
                double akl = ak + al;
                akl_cache[0] = ak;
                akl_cache[nsq_per_block] = al;
                double ck = env[bas[ksh*BAS_SLOTS+PTR_COEFF]];
                double cl = env[bas[lsh*BAS_SLOTS+PTR_COEFF]];
                double ckcl = ck * cl;
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[0*nsq_per_block] = xlxk;
                rlrk[1*nsq_per_block] = ylyk;
                rlrk[2*nsq_per_block] = zlzk;
                gx[nsq_per_block*g_size] = fac_sym * ckcl * cicj / (aij*akl*sqrt(aij+akl));
            }

            double gout[GOUT_WIDTH1];
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH1; ++n) { gout[n] = 0; }

            img_counts = page->nimgs;
            for (int img = 0; img < img_max; ++img) {
                __syncthreads();
                int jL = 0;
                int kL = 0;
                int klL = 0;
                if (page_id < num_pages && img < img_counts) {
                    jL = page->img_j[img];
                    kL = page->img_k[img];
                    klL = page->img_kl[img];
                } else if (gout_id == 0) {
                    gx[nsq_per_block*g_size] = 0.;
                }
                __syncthreads();
                double ak = akl_cache[0];
                double al = akl_cache[nsq_per_block];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rlrk[0*nsq_per_block] + img_coords[klL*3+0];
                double ylyk = rlrk[1*nsq_per_block] + img_coords[klL*3+1];
                double zlzk = rlrk[2*nsq_per_block] + img_coords[klL*3+2];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rjri[0] + img_coords[jL*3+0];
                double yjyi = rjri[1] + img_coords[jL*3+1];
                double zjzi = rjri[2] + img_coords[jL*3+2];
                double xij = ri[0] + xjxi * aj_aij;
                double yij = ri[1] + yjyi * aj_aij;
                double zij = ri[2] + zjzi * aj_aij;
                double xkl = rk[0] + img_coords[kL*3+0] + xlxk * al_akl;
                double ykl = rk[1] + img_coords[kL*3+1] + ylyk * al_akl;
                double zkl = rk[2] + img_coords[kL*3+2] + zlzk * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                if (gout_id == 0) {
                    double theta_kl = ak * al_akl;
                    double theta_ij = ai * aj_aij;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    rjLri[0*nsq_per_block] = xjxi;
                    rjLri[1*nsq_per_block] = yjyi;
                    rjLri[2*nsq_per_block] = zjzi;
                    rlLrk[0*nsq_per_block] = xlxk;
                    rlLrk[1*nsq_per_block] = ylyk;
                    rlLrk[2*nsq_per_block] = zlzk;
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    gx[0] = exp(-theta_kl * rr_kl - theta_ij * rr_ij);
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                int nroots = bounds.nroots;
                rys_roots_for_k(nroots, theta, rr, rw, kmat.omega,
                                kmat.lr_factor, kmat.sr_factor);
                int lij = li + lj;
                int lkl = lk + ll;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[nsq_per_block*g_size*2] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double aij = ai + aj;
                    double ak = akl_cache[0];
                    double al = akl_cache[nsq_per_block];
                    double akl = ak + al;
                    double rt_aa = rt / (aij + akl);
                    double s0x, s1x, s2x;

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        double aj_aij = aj / aij;
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * g_size * nsq_per_block;
                            double Rpa = rjLri[n*nsq_per_block] * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n*nsq_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsq_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lkl > 0) {
                        double al_akl = al / akl;
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlLrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsq_per_block];
                                }
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x;
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
                            int lkl3 = (lkl+1)*3;
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xjxi = rjLri[_ix*nsq_per_block];
                                double *_gx = gx + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = lij + j*li; // = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nsq_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nsq_per_block];
                                        _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }
                    if (ll > 0) {
                        __syncthreads();
                        if (page_id < num_pages) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xlxk = rlLrk[_ix*nsq_per_block];
                                double *_gx = gx + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
                                    int kl = (lkl+l*lk)*stride_k; // = (lkl-l)*stride_k + l*stride_l;
                                    s1x = _gx[kl*nsq_per_block];
                                    for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                        s0x = _gx[kl*nsq_per_block];
                                        _gx[(kl+stride_l)*nsq_per_block] = s1x - xlxk * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (page_id >= num_pages) {
                        continue;
                    }
                    GXYZOffset goff = gxyz_offsets[gout_id];
                    int *addr_i = idx_i + goff.ioff*3;
                    int *addr_j = idx_j + goff.joff*3;
                    int *addr_k = idx_k + goff.koff*3;
                    int *addr_l = idx_l + goff.loff*3;
                    switch (gout_pattern) {
                    case 0 : inner_dot<3, 3, 3, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 1 : inner_dot<3, 3, 3, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 2 : inner_dot<3, 3, 1, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 3 : inner_dot<3, 3, 1, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 4 : inner_dot<3, 1, 3, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 5 : inner_dot<3, 1, 3, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 6 : inner_dot<3, 1, 1, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 7 : inner_dot<3, 1, 1, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 8 : inner_dot<1, 3, 3, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 9 : inner_dot<1, 3, 3, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 10: inner_dot<1, 3, 1, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 11: inner_dot<1, 3, 1, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 12: inner_dot<1, 1, 3, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 13: inner_dot<1, 1, 3, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 14: inner_dot<1, 1, 1, 3>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    case 15: inner_dot<1, 1, 1, 1>(gout, gx, addr_i, addr_j, addr_k, addr_l); break;
                    }
                }
            }
            __syncthreads();

            for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                GXYZOffset goff = gxyz_offsets[gout_id];
                int ioff = goff.ioff;
                int joff = goff.joff;
                int koff = goff.koff;
                int loff = goff.loff;
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                int nfi = bounds.nfi;
                int nfj = bounds.nfj;
                int nfk = bounds.nfk;
                int nfl = bounds.nfl;
                int ldi = bounds.ntiles_i * 3;
                int ldj = bounds.ntiles_j * 3;
                int ldk = bounds.ntiles_k * 3;
                int ldl = bounds.ntiles_l * 3;
                double *dm_cache = shared_memory + sq_id;
                int active = page_id < num_pages;
                double *dm = kmat.dm + i_dm * nao * nao;
                double *vk = kmat.vk + i_dm * nao * nao;
                //FIXME
                load_dm(dm+j0*nao+k0, dm_cache, nao, nfj, nfk, ldj, ldk, active);
                dot_dm<1, 3, 9, 27>(vk, dm_cache, gout, nao, i0, l0,
                                    ioff, joff, koff, loff, ldk, nfi, nfl, active);
                load_dm(dm+j0*nao+l0, dm_cache, nao, nfj, nfl, ldj, ldl, active);
                dot_dm<1, 3, 27, 9>(vk, dm_cache, gout, nao, i0, k0,
                                    ioff, joff, loff, koff, ldl, nfi, nfk, active);
                load_dm(dm+i0*nao+k0, dm_cache, nao, nfi, nfk, ldi, ldk, active);
                dot_dm<3, 1, 9, 27>(vk, dm_cache, gout, nao, j0, l0,
                                    joff, ioff, koff, loff, ldk, nfj, nfl, active);
                load_dm(dm+i0*nao+l0, dm_cache, nao, nfi, nfl, ldi, ldl, active);
                dot_dm<3, 1, 27, 9>(vk, dm_cache, gout, nao, j0, k0,
                                    joff, ioff, loff, koff, ldl, nfj, nfk, active);
            }
        }
    }
}

GXYZOffset *RYS_make_gxyz_offset(PBCInt2eBounds &bounds)
{
/*
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    ioff = np.arange(0, nfi, 3, dtype=np.int8)
    joff = np.arange(0, nfj, 3, dtype=np.int8)
    koff = np.arange(0, nfk, 3, dtype=np.int8)
    loff = np.arange(0, nfl, 3, dtype=np.int8)
    gxyz_offset = lib.cartesian_prod([ioff, joff, koff, loff])
    copy = 256 // len(gxyz_offset) + 1
    return cp.vstack([cp.asarray(gxyz_offset)]*copy, dtype=np.int8)
*/
    GXYZOffset goff[625];
    int nfi = bounds.nfi;
    int nfj = bounds.nfj;
    int nfk = bounds.nfk;
    int nfl = bounds.nfl;
    int nf = 0;
    for (int i = 0; i < nfi; i += 3) {
    for (int j = 0; j < nfj; j += 3) {
    for (int k = 0; k < nfk; k += 3) {
    for (int l = 0; l < nfl; l += 3) {
        goff[nf].ioff = i;
        goff[nf].joff = j;
        goff[nf].koff = k;
        goff[nf].loff = l;
        ++nf;
    } } } }
    for (int n = nf; n < 256; n += nf) {
        for (int m = 0; m < nf; ++m) {
            goff[n+m] = goff[m];
        }
    }
    checkCudaErrors(
        cudaMemcpyToSymbol(c_gxyz_offset, goff, max(nf, 256)*sizeof(GXYZOffset),
                           0, cudaMemcpyHostToDevice));
    GXYZOffset *p_gxyz_offset;
    cudaGetSymbolAddress((void**)&p_gxyz_offset, c_gxyz_offset);
    return p_gxyz_offset;
}

static size_t threads_scheme_for_k(dim3& threads, PBCInt2eBounds &bounds,
                                   int shm_size, int gout_stride_max)
{
/*
    order = li + lj + lk + ll
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    ntiles_i = (nfi + 2) // 3
    ntiles_j = (nfj + 2) // 3
    ntiles_k = (nfk + 2) // 3
    ntiles_l = (nfl + 2) // 3
    ldi = ntiles_i * 3
    ldj = ntiles_j * 3
    ldk = ntiles_k * 3
    ldl = ntiles_l * 3
    cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9
    g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
    nroots = order // 2 + 1
    if omega < 0: # SR
        nroots *= 2
    vk_cache_size = max(nfi, nfj) * max(nfk, nfl)
    dm_cache_size = max(ldi, ldj) * max(ldk, ldl)
    root_g_cache_size = nroots*2 + g_size*3 + 9
    unit = max(root_g_cache_size, vk_cache_size+dm_cache_size)
    counts = (shm_size - cart_idx_size*4) // (unit*8)
    n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l
    gout_stride = min(n_tiles, THREADS)
    nsq_per_block = min(counts, THREADS // gout_stride)
    if nsq_per_block > 8:
        nsq_per_block = nsq_per_block // 8 * 8
    buflen = nsq_per_block * unit*8 + cart_idx_size*4
*/
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int ldi = ntiles_i * 3;
    int ldj = ntiles_j * 3;
    int ldk = ntiles_k * 3;
    int ldl = ntiles_l * 3;
    int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
    int g_size = bounds.g_size;
    int nroots = bounds.nroots;
    int dm_cache_size = max(ldi, ldj) * max(ldk, ldl);
    int root_g_cache_size = nroots*2 + g_size*3 + 14;
    int unit = max(root_g_cache_size, dm_cache_size);
    int counts = (shm_size - cart_idx_size*4) / (unit*8);
    int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
    int THREADS = 256;
    int gout_stride = min(n_tiles, gout_stride_max);
    int nsq_per_block = min(counts, THREADS / gout_stride);
    if (nsq_per_block > 8) {
        nsq_per_block = nsq_per_block / 8 * 8;
    }
    threads.x = nsq_per_block;
    threads.y = gout_stride;
    int buflen = nsq_per_block * unit*8 + cart_idx_size*4;
    return buflen;
}

extern int rys_k_unrolled(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds, int *pool);

extern "C" {
int PBC_build_k(double *vk, double *dm, int n_dm, int nao,
                PBCIntEnvVars envs, int *shls_slice, int shm_size,
                int npairs_ij, int npairs_kl, int *pair_ij_mapping, int *pair_kl_mapping,
                float *q_cond, float *s_estimator, float *dm_cond,
                int *img_idx, uint32_t *img_offsets, float cutoff,
                ImgIdxPage *pool, int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    int jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int ntiles_i = (nfi + 2) / 3;
    int ntiles_j = (nfj + 2) / 3;
    int ntiles_k = (nfk + 2) / 3;
    int ntiles_l = (nfl + 2) / 3;
    int order = li + lj + lk + ll;
    int nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size = stride_l * (ll + 1);
    PBCInt2eBounds bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff,
        ntiles_i, ntiles_j, ntiles_k, ntiles_l, img_offsets, img_idx};

    JKMatrix kmat = {NULL, vk, dm, n_dm, 0, omega};
    if (omega >= 0) {
        kmat.lr_factor = 1;
        kmat.sr_factor = 0;
    } else {
        kmat.lr_factor = 0;
        kmat.sr_factor = 1;
    }

    if (1){//!rys_k_unrolled(&envs, &kmat, &bounds, pool)) {
        GXYZOffset* p_gxyz_offset = RYS_make_gxyz_offset(bounds);
        int gout_pattern = (((li == 0) >> 3) |
                            ((lj == 0) >> 2) |
                            ((lk == 0) >> 1) |
                            ( ll == 0));
        dim3 threads;
        int buflen = threads_scheme_for_k(threads, bounds, shm_size, 256);
        int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;

        rys_k_kernel<<<npairs_ij, threads, buflen>>>(
            envs, kmat, bounds, pool, p_gxyz_offset,
            gout_pattern, reserved_shm_size);

        int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
        if (n_tiles > 256) { // fffg, ffgg, fggg, gggg
            buflen = threads_scheme_for_k(threads, bounds, shm_size,
                                          min(256, n_tiles-256));
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;
            rys_k_kernel<<<npairs_ij, threads, buflen>>>(
                envs, kmat, bounds, pool, p_gxyz_offset+256,
                gout_pattern, reserved_shm_size);
        }

        if (n_tiles > 512) { // gggg
            buflen = threads_scheme_for_k(threads, bounds, shm_size,
                                          min(256, n_tiles-512));
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;
            rys_k_kernel<<<npairs_ij, threads, buflen>>>(
                envs, kmat, bounds, pool, p_gxyz_offset+512,
                gout_pattern, reserved_shm_size);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in PBC_build_k, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int PBC_build_k_init(int shm_size)
{
    cudaFuncSetAttribute(rys_k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
