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
#include "gvhf-rys/rys_roots.cu"
#include "int3c2e.cuh"

#define WARP_SIZE       32
// corresponding to 256 threads
#define WARPS           8
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

    extern __shared__ double rw[];
    double *g = rw + nksh_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nksh_per_block * g_size;
    double *gz = gy + nksh_per_block * g_size;
    float *xk_cache = (float *)(g + 3 * nksh_per_block * g_size);
    float *yk_cache = xk_cache + nksh_per_block;
    float *zk_cache = yk_cache + nksh_per_block;
    int8_t *img_mask = (int8_t *)(zk_cache + nksh_per_block);

    for (int k = thread_id; k < nksh; k += THREADS) {
        double *rk = env + bas[(ksh0+k)*BAS_SLOTS+PTR_BAS_COORD];
        xk_cache[k] = rk[0];
        yk_cache[k] = rk[1];
        zk_cache[k] = rk[2];
    }

    double rjri[3], Rpq[3];
    double gout[GOUT_WIDTH];

    int ntasks = nksh * nsp_per_block * SPTAKS_PER_BLOCK;
    for (int task_id = 0; task_id < ntasks;
         task_id += nksh_per_block*nsp_per_block) {
        // convert task_id to ish, jsh, ksh
        int ijk_idx = task_id + sp_id * nksh_per_block + ksh_id;
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        if (pair_ij_idx >= bounds.npairs_ij) {
            pair_ij_idx = sp0_this_block;
        }
        int ish = bounds.ish_in_pair[pair_ij_idx];
        int jsh = bounds.jsh_in_pair[pair_ij_idx];
        int img0 = sp_img_offsets[pair_ij_idx];
        int img1 = sp_img_offsets[pair_ij_idx+1];
        extern __shared__ int img_counts_in_warp[];
        if (thread_id < WARPS) {
            img_counts_in_warp[thread_id] = 0;
        }
        __syncthreads();
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();
        int img_counts = img_counts_in_warp[warp_id];
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
                double aj_aij = aj / aij;
                double cijk = ci[ip] * cj[jp] * ck[kp];

                float ai_f32 = ai;
                float aj_f32 = aj;
                float aij_f32 = aij;
                float ak_f32 = ak;
                float fi = ai_f32 / aij_f32;
                float fj = aj_f32 / aij_f32;
                float theta_ij_f32 = ai_f32 * fj;
                float omega_f32 = omega;
                float omega2 = omega_f32 * omega_f32;
                float theta_ij_omega = aij_f32 * omega2 / (aij_f32 + omega2);
                // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
                //           ~ between [0, 2]
                float fac_guess = 1.f;
                // fac in Eq 63 of arXiv:2302.11307 ~ log(ci*cj*ck * (pi^2/(aij*ak))**1.5)
                float log_fac = logf(cijk) + 1.5f*logf(9.87f/(aij_f32*ak_f32)) + fac_guess;
                // theta = 1/(1/aij+1/ak+1/omega^2);
                // theta*r_guess^2 ~= omega^2*r_guess^2 ~ 1e-9 => r_guess ~= 5/omega
                // theta*r_guess/aij in Eq 64 of arXiv:2302.11307 ~ omega^2*r_guess/aij ~ omega/aij * 5.f
                float r_omega_aij = omega_f32/aij_f32 * 5.f;
                float u = .5f / aij_f32;
                __syncthreads();
                for (int img = thread_id; img < img_counts; img += THREADS) {
FIXME warp_id * img_counts
                    img_mask[img] = 0;
                }
FIXME img_mask for each warp
                int nimgs_aligned = (img_counts + THREADS - 1) & (0x100000 - THREADS);
                for (int n = thread_id; n < nimgs_aligned; n += THREADS) {
                    int k = n / img_counts;
                    int img = n % img_counts;
                    __syncthreads();
                    if (ksh >= nksh || img_mask[img] != 0) { continue; }
                    int img_id = img_idx[img];
                    int iL = img_id / nimgs;
                    int jL = img_id % nimgs;
                    float xi = ri[0] + img_coords[iL];
                    float yi = ri[1] + img_coords[iL];
                    float zi = ri[2] + img_coords[iL];
                    float xj = rj[0] + img_coords[jL];
                    float yj = rj[1] + img_coords[jL];
                    float zj = rj[2] + img_coords[jL];
                    float xjxi = xj - xi;
                    float yjyi = yj - yi;
                    float zjzi = zj - zi;
                    float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
                    float theta_ij_rr = theta_ij_f32 * rr_ij;
                    float penalty = 7;
                    // Remaining terms in the estimator almost never exceed 1e3.
                    // A penalty ~ log(1e3) to capture other factors in the overlap
                    // of orbital basis.
                    if (log_fac + penalty - theta_ij_rr < log_cutoff) {
                        continue;
                    }
                    float r = sqrtf(rr_ij);
                    float ti = fj * r + r_omega_aij;
                    float tj = fi * r + r_omega_aij;
                    float ti_fac = .5f*li * logf(ti*ti + li*u);
                    float tj_fac = .5f*lj * logf(tj*tj + lj*u);
FIXME ensure (pi/ak)^1.5 * (lk/ak)^(lk/2) properly cancels the normalization factors in ck[kp]
                    float tk_fac = .5f*lk * logf(lk*.5f/ak_f32);
                    float xij = xjxi * fj + xi;
                    float yij = yjyi * fj + yi;
                    float zij = zjzi * fj + zi;
                    float dx = xij - xk_cache[k];
                    float dy = yij - yk_cache[k];
                    float dz = zij - zk_cache[k];
                    float rr = dx * dx + dy * dy + dz * dz;
                    // theta = 1/(1/aij+1/ak+1/omega2);
                    float theta = (theta_ij_omega * ak_f32) / (theta_ij_omega + ak_f32);
                    float estimator = log_fac + ti_fac + tj_fac + tk_fac - theta_ij_rr - theta*rr;
                    if (estimator > log_cutoff) {
                        img_mask[img] = 1;
                    }
                }

                __syncthreads();
                if (gout_id == 0) {
                    double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                    gy[ksh_id] = fac;
                }
                for (int img = 0; img < img_counts; ++img) {
                    if (img_mask[img] == 0) {
                        continue;
                    }
                    int img_id = img0 + img;
                    if (img_id > img1) {
                        // ensure threads in the warp processing the same number of images
                        img_id = img0;
                        gy[ksh_id] = 0.;
                    }
                    img_id = img_idx[img_id];
                    int iL = img_id / nimgs;
                    int jL = img_id % nimgs;
                    double xi = ri[0] + img_coords[iL];
                    double yi = ri[1] + img_coords[iL];
                    double zi = ri[2] + img_coords[iL];
                    double xj = rj[0] + img_coords[jL];
                    double yj = rj[1] + img_coords[jL];
                    double zj = rj[2] + img_coords[jL];
                    double xjxi = xj - xi;
                    double yjyi = yj - yi;
                    double zjzi = zj - zi;
                    rjri[0] = xjxi;
                    rjri[1] = yjyi;
                    rjri[2] = zjzi;
                    __syncthreads();
                    double theta_ij = ai * aj_aij;
                    double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
                    gx[ksh_id] = Kab;

                    double xij = xjxi * aj_aij + xi;
                    double yij = yjyi * aj_aij + yi;
                    double zij = zjzi * aj_aij + zi;
                    double xpq = xij - rk[0];
                    double ypq = yij - rk[1];
                    double zpq = zij - rk[2];
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double theta = aij * ak / (aij + ak);
                    double theta_rr = theta * rr;

                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nksh_per_block);
                    double omega2 = omega * omega;
                    double theta_fac = omega2 / (omega2 + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[ksh_id+ irys*2   *nksh_per_block] *= theta_fac;
                        rw[ksh_id+(irys*2+1)*nksh_per_block] *= sqrt_theta_fac;
                    }

                    double s0x, s1x, s2x;
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gz[ksh_id] = rw[ksh_id+(irys*2+1)*nksh_per_block];
                        }
                        double rt = rw[ksh_id + irys*2*nksh_per_block];
                        double rt_aa = rt / (aij + ak);

                        if (lij > 0) {
                            __syncthreads();
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = g + n * g_size * nksh_per_block;
                                int ir = ksh_id + n * nksh_per_block;
                                double xpa = rjri[ir] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n];
                                s0x = _gx[ksh_id];
                                s1x = c0x * s0x;
                                _gx[ksh_id + nksh_per_block] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    s2x = c0x * s1x + i * b10 * s0x;
                                    _gx[ksh_id + (i+1)*nksh_per_block] = s2x;
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
                                double *_gx = g + (i + _ix * g_size) * nksh_per_block;
                                double cpx = rt_ak * Rpq[_ix];
                                //for i in range(lij+1):
                                //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                                if (n < lij3) {
                                    s0x = _gx[ksh_id];
                                    s1x = cpx * s0x;
                                    if (i > 0) {
                                        s1x += i * b00 * _gx[ksh_id-nksh_per_block];
                                    }
                                    _gx[ksh_id + stride_k*nksh_per_block] = s1x;
                                }
                                //for k in range(1, lk):
                                //    for i in range(lij+1):
                                //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                                for (int k = 1; k < lk; ++k) {
                                    __syncthreads();
                                    if (n < lij3) {
                                        s2x = cpx*s1x + k*b01*s0x;
                                        if (i > 0) {
                                            s2x += i * b00 * _gx[ksh_id + (k*stride_k-1)*nksh_per_block];
                                        }
                                        _gx[ksh_id + (k*stride_k+stride_k)*nksh_per_block] = s2x;
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
                                    double *_gx = g + (_ix*g_size + k*stride_k) * nksh_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ksh_id + ij*nksh_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ksh_id + ij*nksh_per_block];
                                            _gx[ksh_id + (ij+stride_j)*nksh_per_block] = s1x - xjxi * s0x;
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
                            int addrx = ksh_id + (idx_ij[ij] + idx_k[k] * stride_k) * nksh_per_block;
                            int addry = ksh_id + (idy_ij[ij] + idy_k[k] * stride_k) * nksh_per_block;
                            int addrz = ksh_id + (idz_ij[ij] + idz_k[k] * stride_k) * nksh_per_block;
                            gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                        }
                    }
                }
            }

            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int ncells = envs.bvk_ncells;
                int nbasp = nbas / ncells;
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
