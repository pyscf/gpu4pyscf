/*
 * Copyright 2024 The PySCF Developers. All Rights Reserved.
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

#include "gvhf-rys/vhf.cuh"
#include "rys_roots.cu"
#include "int3c2e.cuh"

#define THREADS         (WARP_SIZE*WARPS)
// TODO: benchmark performance for 32, 40, 45, 54
#define GOUT_WIDTH      40

__global__
void pbc_int3c2e_kernel(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int nksh_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int nsp_per_block = blockDim.z;
    int ksh_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;

    int nksp_per_block = nksh_per_block * nsp_per_block;
    int ksp_id = nksh_per_block * sp_id + ksh_id;
    int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * nsp_per_block * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * nksh_per_block;
    int nksh = MIN(bounds.nksh - ksh0_this_block, nksh_per_block);
    int ksh0 = ksh0_this_block + bounds.ksh0;

    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int lij = li + lj;
    int nroots = bounds.nroots;
    int nfi = bounds.nfi;
    int nfij = bounds.nfij;
    int nfk = bounds.nfk;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int g_size = bounds.g_size;
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int lk_offset = lk * (lk + 1) * (lk + 2) / 2;
    int *idx_k = c_g_cart_idx + lk_offset;
    int *idy_k = idx_k + nfk;
    int *idz_k = idy_k + nfk;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    float log_cutoff = envs.log_cutoff;

    int gx_len = g_size * nksp_per_block;
    extern __shared__ int img_counts_in_warp[];
    double *rw = (double *)(img_counts_in_warp + WARPS);
    rw += ksp_id;
    double *g = rw + nksp_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    int8_t *img_mask = (int8_t *)(img_counts_in_warp + WARPS);
    double rjri[3], Rpq[3];
    double gout[GOUT_WIDTH];

    int ntasks = nksh * nsp_per_block * SPTAKS_PER_BLOCK;
    for (int task_id = 0; task_id < ntasks; task_id += nksp_per_block) {
        // convert task_id to ish, jsh, ksh
        int ijk_idx = task_id + ksp_id;
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1;
        if (pair_ij_idx >= bounds.npairs_ij) {
            pair_ij_idx = sp0_this_block;
            img1 = 0;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij_idx];
        int img0 = sp_img_offsets[pair_ij_idx];
        if (thread_id < WARPS) {
            img_counts_in_warp[thread_id] = 0;
        }
        __syncthreads();
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();
        int img_counts = img_counts_in_warp[warp_id];

        int ish = bas_ij / envs.nbas;
        int jsh = bas_ij % envs.nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

        for (int gout_start = 0; gout_start < nfij*nfk;
             gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp % kprim;
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double aij = ai + aj;
                double cijk = ci[ip] * cj[jp] * ck[kp];
                __syncthreads();
                if (gout_id == 0) {
                    double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                    gy[0] = fac;
                }
                for (int img = 0; img < img_counts; ++img) {
                    int img_id = img0 + img;
                    __syncthreads();
                    if (img_id > img1) {
                        // ensure threads in the warp processing the same number of images
                        img_id = img0;
                        if (gout_id == 0) {
                            gy[0] = 0.;
                        }
                    }
                    int img_ij = img_idx[img_id];
                    int iL = img_ij / nimgs;
                    int jL = img_ij % nimgs;
                    double xi = ri[0] + img_coords[iL*3+0];
                    double yi = ri[1] + img_coords[iL*3+1];
                    double zi = ri[2] + img_coords[iL*3+2];
                    double xj = rj[0] + img_coords[jL*3+0];
                    double yj = rj[1] + img_coords[jL*3+1];
                    double zj = rj[2] + img_coords[jL*3+2];
                    double xjxi = xj - xi;
                    double yjyi = yj - yi;
                    double zjzi = zj - zi;
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double aj_aij = aj / aij;
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rr_ij;

                    double xij = xjxi * aj_aij + xi;
                    double yij = yjyi * aj_aij + yi;
                    double zij = zjzi * aj_aij + zi;
                    double xpq = xij - rk[0];
                    double ypq = yij - rk[1];
                    double zpq = zij - rk[2];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double theta = aij * ak / (aij + ak);
                    double omega2 = omega * omega;
                    double theta_fac = omega2 / (omega2 + theta);
#if 1
                    if (thread_id < WARP_SIZE == 0) {
                        img_mask[warp_id] = 0;
                    }
                    __syncthreads();
                    float Kab_f32 = Kab;
                    if (gout_id == 0 && img+img0 < img1 && Kab_f32-5.f*lij < log_cutoff) {
                        // check any not vanished integrals
                        float ai_f32 = ai;
                        float aj_f32 = aj;
                        float aij_f32 = aij;
                        float ak_f32 = ak;
                        float fi = ai_f32 / aij_f32;
                        float fj = aj_f32 / aij_f32;
                        // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
                        //           ~ between [0, 2]
                        float fac_guess = 1.3f;
                        // fac in Eq 63 of arXiv:2302.11307 ~ log(ci*cj*ck * (pi^2/(aij*ak))**1.5)
                        float log_fac = logf(cijk) + 1.5f*logf(9.87f/(aij_f32*ak_f32)) + fac_guess;
                        float theta_fac_rr = (float)theta_fac * (float)rr;
                        float rt_rpq = sqrtf((float)rr) * ak_f32/(aij_f32+ak_f32);
                        float u = .5f / aij_f32;
                        float r = sqrtf((float)rr_ij);
                        float ti = fj * r + rt_rpq;
                        float tj = fi * r + rt_rpq;
                        float ti_fac = .5f*li * logf(ti*ti + li*u + 1.f);
                        float tj_fac = .5f*lj * logf(tj*tj + lj*u + 1.f);
                        float tk_fac = .5f*lk * logf(rt_rpq*rt_rpq + lk*.5f/ak_f32 + 1.f);
                        float estimator = log_fac + ti_fac + tj_fac + tk_fac - Kab_f32 - theta_fac_rr;
                        if (estimator > log_cutoff) {
                            img_mask[warp_id] = 1;
                        }
                    }
                    __syncthreads();
                    if (img_mask[warp_id] == 0) {
                        continue;
                    }
#endif
                    rjri[0] = xjxi;
                    rjri[1] = yjyi;
                    rjri[2] = zjzi;
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    int _nroots = nroots/2;
                    double theta_rr = theta * rr;
                    gx[0] = exp(-Kab);
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
                                double *_gx = gx + n * g_size * nksp_per_block;
                                double xpa = rjri[n] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n];
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
                                double cpx = rt_ak * Rpq[_ix];
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
                            if (task_id < ntasks) {
                                int lk3 = (lk+1)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix];
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
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            int ijk = (gout_start + n*gout_stride+gout_id);
                            int k  = ijk / nfij;
                            int ij = ijk % nfij;
                            if (k >= nfk) break;
                            int addrx = (idx_ij[ij] + idx_k[k] * stride_k) * nksp_per_block;
                            int addry = (idy_ij[ij] + idy_k[k] * stride_k) * nksp_per_block;
                            int addrz = (idz_ij[ij] + idz_k[k] * stride_k) * nksp_per_block;
                            gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                        }
                    }
                }
            }

            int ijk_idx = task_id + ksp_id;
            int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
            if (pair_ij_idx < bounds.npairs_ij) {
                int *ao_loc = envs.ao_loc;
                int ncells = envs.bvk_ncells;
                int nbasp = envs.nbas / ncells;
                size_t ncol = bounds.ncol * ncells;
                size_t naux = bounds.naux;
                int cell_i = ish / nbasp;
                int cell0_ish = ish % nbasp;
                int cell_j = jsh / nbasp;
                int cell0_jsh = jsh % nbasp;
                size_t i0 = cell_i * (ao_loc[cell0_ish] - ao_loc[bounds.ish0]);
                size_t j0 = cell_j * (ao_loc[cell0_jsh] - ao_loc[bounds.jsh0]);
                size_t k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
                double *eri_tensor = out + (i0*ncol+j0) * naux + k0;
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijk = n*gout_stride + gout_id;
                    size_t k  = ijk / nfij;
                    size_t ij = ijk % nfij;
                    if (k >= nfk) break;
                    size_t i = ij % nfi;
                    size_t j = ij / nfi;
                    size_t addr = (i*ncol+j)*naux + k;
                    eri_tensor[addr] = gout[n];
                }
            }
        }
    }
}
