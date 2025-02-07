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

#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "int3c2e.cuh"

#define THREADS         (WARP_SIZE*WARPS)
// TODO: benchmark performance for 32, 38, 40, 45, 54
#define GOUT_WIDTH      45

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

    int gx_len = g_size * nksp_per_block;
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + ksp_id;
    double *g = rw + nksp_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    double *rjri = gz + gx_len;
    double *Rpq = rjri + nksp_per_block * 3;
    __shared__ int img_counts_in_warp[WARPS];
    double gout[GOUT_WIDTH];

    int ntasks = nksh * nsp_per_block * SPTAKS_PER_BLOCK;
    for (int task_id = 0; task_id < ntasks; task_id += nksp_per_block) {
        // convert task_id to ish, jsh, ksh
        int ijk_idx = task_id + ksp_id;
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.npairs_ij) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
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
                int img_counts = img_counts_in_warp[warp_id];
                for (int img = 0; img < img_counts; ++img) {
                    int img_id = img0 + img;
                    __syncthreads();
                    if (img_id >= img1) {
                        // ensure the same number of images processed in the same warp
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
                    double theta_rr = theta * rr;
// Somehow, this screening test does not filter out many integrals.
// More benchmarks are needed
#if 0
                    __shared__ int8_t img_mask[WARPS];
                    if (thread_id_in_warp == 0) {
                        img_mask[warp_id] = 0;
                    }
                    float Kab_f32 = Kab;
                    // IMPORTANT: run the screening test on each warp.
                    // When nksh_per_block*gout_stride>32, gout is evaluated across warps.
                    // If tests are skipped for some warps, g[xyz] vectors and
                    // gout on these warps will never be evaluated. These warps
                    // may proceeed to a wrong __syncthreads() barrier and
                    // produce wrong g[xyz].
                    float log_cutoff = envs.log_cutoff;
                    if ((thread_id_in_warp / nksh_per_block == 0) &&
                        img0+img < img1 && 5.f+2.f*lij-Kab_f32 > log_cutoff) {
                        // check any not vanished integrals
                        float ai_f32 = ai;
                        float aj_f32 = aj;
                        float aij_f32 = aij;
                        float ak_f32 = ak;
                        float fi = ai_f32 / aij_f32;
                        float fj = aj_f32 / aij_f32;
                        // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
                        //           ~ between [0, 2]
                        float fac_guess = 1.f;
                        // fac in Eq 63 of arXiv:2302.11307 ~ log(ci*cj*ck * (pi^2/(aij*ak))**1.5)
                        float log_fac = logf(fabs(cijk)) + 3.434f - 1.5f*logf(aij_f32*ak_f32) + fac_guess;
                        float theta_fac_rr = (float)theta_fac * (float)theta_rr;
                        float rt_aa = sqrtf((float)rr) / (aij_f32+ak_f32) + 1e-9f;
                        float rt_aij = rt_aa * ak_f32;
                        float rt_akl = rt_aa * aij_f32;
                        float r = sqrtf((float)rr_ij);
                        float ti = fj * r + rt_aij;
                        float tj = fi * r + rt_aij;
                        float ti_fac = .5f*li * logf(ti*ti + .5f*li/aij_f32);
                        float tj_fac = .5f*lj * logf(tj*tj + .5f*lj/aij_f32);
                        float tk_fac = .5f*lk * logf(rt_akl*rt_akl + .5f*lk/ak_f32);
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
                    if (gout_id == 0) {
                        rjri[0*nksp_per_block] = xjxi;
                        rjri[1*nksp_per_block] = yjyi;
                        rjri[2*nksp_per_block] = zjzi;
                        Rpq[0*nksp_per_block] = xpq;
                        Rpq[1*nksp_per_block] = ypq;
                        Rpq[2*nksp_per_block] = zpq;
                        gx[0] = exp(-Kab);
                    }
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
                            if (task_id < ntasks) {
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
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            int ijk = gout_start + n*gout_stride+gout_id;
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

            if (pair_ij_idx < bounds.npairs_ij) {
                int *ao_loc = envs.ao_loc;
                int nbasp = envs.cell0_nbas;
                int ncells = envs.bvk_ncells;
                int cell_i = ish / nbasp;
                int cell0_ish = ish % nbasp;
                int cell_j = jsh / nbasp;
                int cell0_jsh = jsh % nbasp;
                int nrow = bounds.nrow;
                int ncol = bounds.ncol;
                size_t naux = bounds.naux;
                int i0 = ao_loc[cell0_ish] - ao_loc[bounds.ish0];
                int j0 = ao_loc[cell0_jsh] - ao_loc[bounds.jsh0];
                int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
                double *eri_tensor = out + (((cell_i * nrow + i0) * ncells +
                                              cell_j) * ncol + j0) * naux + k0;
                int nKj = ncells * ncol;
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijk = gout_start + n*gout_stride+gout_id;
                    size_t k  = ijk / nfij;
                    size_t ij = ijk % nfij;
                    if (k >= nfk) break;
                    size_t i = ij % nfi;
                    size_t j = ij / nfi;
                    size_t addr = (i*nKj+j)*naux + k;
                    eri_tensor[addr] = gout[n];
                }
            }
        }
    }
}

__global__
void sr_int3c2e_img_counts_kernel(int *img_counts, PBCInt3c2eEnvVars envs,
                                  float *exps, float *log_coeff, float *aux_exps,
                                  int ish0, int jsh0, int nish, int njsh)
{
    int Ki = blockIdx.x;
    int Kj = blockIdx.y;
    int cell_i = Ki / nish;
    int cell_j = Kj / njsh;
    int cell0_ish = Ki % nish + ish0;
    int cell0_jsh = Kj % njsh + jsh0;
    int nbasp = envs.cell0_nbas;
    int ish = cell_i * nbasp + cell0_ish;
    int jsh = cell_j * nbasp + cell0_jsh;
    int ncells = envs.bvk_ncells;
    int nKj = ncells * njsh;
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int nimgs = envs.nimgs;
    int nimgs2 = nimgs * nimgs;
    int cell0_natm = envs.cell0_natm;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    extern __shared__ float x_cache[];
    float *y_cache = x_cache + cell0_natm;
    float *z_cache = y_cache + cell0_natm;
    for (int k = thread_id; k < cell0_natm; k += threads) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        x_cache[k] = rk[0];
        y_cache[k] = rk[1];
        z_cache[k] = rk[2];
    }
    __syncthreads();

    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float aij = ai + aj;
    float u = .5f / aij;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float log_cutoff = envs.log_cutoff;

    // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
    //           ~ between [0, 2]
    float fac_guess = .5f - logf(omega2)/4;
    float log_fac = log_ci + log_cj + 1.717f - 1.5f*logf(aij) + fac_guess;

    int count = 0;
    for (int ijL = thread_id; ijL < nimgs2; ijL += threads) {
        int iL = ijL / nimgs;
        int jL = ijL % nimgs;
        float xiL = xi + img_coords[iL*3+0];
        float yiL = yi + img_coords[iL*3+1];
        float ziL = zi + img_coords[iL*3+2];
        float xjL = xj + img_coords[jL*3+0];
        float yjL = yj + img_coords[jL*3+1];
        float zjL = zj + img_coords[jL*3+2];
        float xjxi = xjL - xiL;
        float yjyi = yjL - yiL;
        float zjzi = zjL - ziL;
        float xij = xjxi * fj + xiL;
        float yij = yjyi * fj + yiL;
        float zij = zjzi * fj + ziL;
        float theta = (omega2 * aij) / (omega2 + aij);
        float rr_min = 1e3f;
        float theta_rr_min = 1e6f;
        for (int k = 0; k < cell0_natm; ++k) {
            float dx = xij - x_cache[k];
            float dy = yij - y_cache[k];
            float dz = zij - z_cache[k];
            float rr = dx * dx + dy * dy + dz * dz;
            float ak = aux_exps[k];
            float theta_k = theta * ak / (theta + ak);
            float theta_rr = theta_k * rr;
            if (theta_rr < theta_rr_min) {
                theta_rr_min = theta_rr;
                rr_min = rr;
            }
        }

        // exp(- 1/(1/aij+1/ak+1/omega^2) * r_guess^2) < 1e-9
        // => ~ exp(- omega^2 * r_guess^2) < 1e-9
        // => r_guess > 5/omega
        // 1/(1/aij+1/ak+1/omega^2)*r_guess/aij in Eq 64 of arXiv:2302.11307
        //     ~ omega^2*r_guess/aij ~ omega/aij * 5.f
        //float rt_aij = fabs(omega)/aij * 5.;
        float rt_aij = omega2 * sqrtf(rr_min) / aij + 1e-9f;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float dr = sqrtf(rr_ij);
        float dri = fj * dr + rt_aij;
        float drj = fi * dr + rt_aij;
        float dri_fac = .5f*li * logf(dri*dri + li*u);
        float drj_fac = .5f*lj * logf(drj*drj + lj*u);
        float estimator = log_fac + dri_fac + drj_fac - theta_ij*rr_ij - theta_rr_min;
        if (estimator > log_cutoff) {
            count += 1;
        }
    }

    extern __shared__ int counts[];
    counts[thread_id] = count;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            counts[thread_id] += counts[thread_id + stride];
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        img_counts[Ki*nKj+Kj] = counts[0];
    }
}

__global__
void sr_int3c2e_img_idx_kernel(int *img_idx, int *img_offsets, int *bas_mapping,
                               PBCInt3c2eEnvVars envs,
                               float *exps, float *log_coeff, float *aux_exps,
                               int ish0, int jsh0, int nish, int njsh)
{
    int thread_id = threadIdx.x;
    int threads = blockDim.x;
    int ncells = envs.bvk_ncells;
    int nKj = ncells * njsh;
    int row_id = blockIdx.x;
    int bas_ij = bas_mapping[row_id];
    int Ki = bas_ij / nKj;
    int Kj = bas_ij % nKj;
    int cell_i = Ki / nish;
    int cell_j = Kj / njsh;
    int cell0_ish = Ki % nish + ish0;
    int cell0_jsh = Kj % njsh + jsh0;
    int nbasp = envs.cell0_nbas;
    int ish = cell_i * nbasp + cell0_ish;
    int jsh = cell_j * nbasp + cell0_jsh;
    int nimgs = envs.nimgs;
    int nimgs2 = nimgs * nimgs;
    int cell0_natm = envs.cell0_natm;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    extern __shared__ int8_t mask[];
    uint16_t* cum_count = (uint16_t *)(mask + IMG_BLOCK);
    float *x_cache = (float *)(cum_count + threads);
    float *y_cache = x_cache + cell0_natm;
    float *z_cache = y_cache + cell0_natm;
    for (int k = thread_id; k < cell0_natm; k += threads) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        x_cache[k] = rk[0];
        y_cache[k] = rk[1];
        z_cache[k] = rk[2];
    }
    for (int i = thread_id; i < IMG_BLOCK; i += threads) {
        mask[i] = 0;
    }
    __syncthreads();

    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float aij = ai + aj;
    float u = .5f / aij;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float log_cutoff = envs.log_cutoff;

    // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
    //           ~ between [0, 2]
    float fac_guess = .5f - logf(omega2)/4;
    float log_fac = log_ci + log_cj + 1.717f - 1.5f*logf(aij) + fac_guess;
    int offset_start = img_offsets[row_id];

    for (int img_start = 0; img_start < nimgs2; img_start += IMG_BLOCK) {
        int block_nimgs2 = MIN(IMG_BLOCK, nimgs2-img_start);
        int bacth_size = (block_nimgs2 + threads - 1) / threads;
        int ij0 = img_start + thread_id * bacth_size;
        int ij1 = MIN(ij0 + bacth_size, nimgs2);

        int count = 0;
        for (int ijL = ij0; ijL < ij1; ++ijL) {
            int iL = ijL / nimgs;
            int jL = ijL % nimgs;
            float xiL = xi + img_coords[iL*3+0];
            float yiL = yi + img_coords[iL*3+1];
            float ziL = zi + img_coords[iL*3+2];
            float xjL = xj + img_coords[jL*3+0];
            float yjL = yj + img_coords[jL*3+1];
            float zjL = zj + img_coords[jL*3+2];
            float xjxi = xjL - xiL;
            float yjyi = yjL - yiL;
            float zjzi = zjL - ziL;
            float xij = xjxi * fj + xiL;
            float yij = yjyi * fj + yiL;
            float zij = zjzi * fj + ziL;
            float theta = (omega2 * aij) / (omega2 + aij);
            float rr_min = 1e3f;
            float theta_rr_min = 1e6f;
            for (int k = 0; k < cell0_natm; ++k) {
                float dx = xij - x_cache[k];
                float dy = yij - y_cache[k];
                float dz = zij - z_cache[k];
                float rr = dx * dx + dy * dy + dz * dz;
                float ak = aux_exps[k];
                float theta_k = theta * ak / (theta + ak);
                float theta_rr = theta_k * rr;
                if (theta_rr < theta_rr_min) {
                    theta_rr_min = theta_rr;
                    rr_min = rr;
                }
            }

            // exp(- 1/(1/aij+1/ak+1/omega^2) * r_guess^2) < 1e-9
            // => ~ exp(- omega^2 * r_guess^2) < 1e-9
            // => r_guess > 5/omega
            // 1/(1/aij+1/ak+1/omega^2)*r_guess/aij in Eq 64 of arXiv:2302.11307
            //     ~ omega^2*r_guess/aij ~ omega/aij * 5.f
            //float rt_aij = fabs(omega)/aij * 5.;
            float rt_aij = omega2 * sqrtf(rr_min) / aij + 1e-9f;
            float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
            float dr = sqrtf(rr_ij);
            float dri = fj * dr + rt_aij;
            float drj = fi * dr + rt_aij;
            float dri_fac = .5f*li * logf(dri*dri + li*u);
            float drj_fac = .5f*lj * logf(drj*drj + lj*u);
            float estimator = log_fac + dri_fac + drj_fac - theta_ij*rr_ij - theta_rr_min;
            if (estimator > log_cutoff) {
                mask[ijL - img_start] = 1;
                count += 1;
            }
        }

        cum_count[thread_id] = count;
        // Up-sweep phase
        for (int stride = 1; stride < threads; stride *= 2) {
            __syncthreads();
            int index = (thread_id + 1) * stride * 2 - 1;
            if (index < threads) {
                cum_count[index] += cum_count[index-stride];
            }
        }
        __syncthreads();
        // Down-sweep phase
        for (int stride = threads/4; stride > 0; stride /= 2) {
            __syncthreads();
            int index = (thread_id + 1) * stride * 2 - 1;
            if (index + stride < threads) {
                cum_count[index + stride] += cum_count[index];
            }
        }
        __syncthreads();

        int offset = offset_start;
        if (thread_id > 0) {
            offset += cum_count[thread_id-1];
        }
        for (int ijL = ij0; ijL < ij1; ++ijL) {
            if (mask[ijL-img_start]) {
                img_idx[offset] = ijL;
                mask[ijL-img_start] = 0;
                ++offset;
            }
        }
        offset_start += cum_count[threads-1];
        __syncthreads();
    }
}
