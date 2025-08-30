/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

#include "vhf1.cuh"
#include "rys_roots.cu"
#include "create_tasks_o1.cu"

#define GOUT_WIDTH0     60
#define GOUT_WIDTH1     81

__constant__ int _c_cartesian_lexical_xyz[] = {
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    2, 0, 0,
    1, 1, 0,
    1, 0, 1,
    0, 2, 0,
    0, 1, 1,
    0, 0, 2,
    3, 0, 0,
    2, 1, 0,
    2, 0, 1,
    1, 2, 0,
    1, 1, 1,
    1, 0, 2,
    0, 3, 0,
    0, 2, 1,
    0, 1, 2,
    0, 0, 3,
    4, 0, 0,
    3, 1, 0,
    3, 0, 1,
    2, 2, 0,
    2, 1, 1,
    2, 0, 2,
    1, 3, 0,
    1, 2, 1,
    1, 1, 2,
    1, 0, 3,
    0, 4, 0,
    0, 3, 1,
    0, 2, 2,
    0, 1, 3,
    0, 0, 4,
};

__device__ __forceinline__ unsigned get_smid()
{
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

template <int LIJ>
__device__ __forceinline__
void vrr(double *g, double *rjri, double *Rpq, double aj_aij, double rt_aij,
         double b10, int g_size)
{
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    for (int n = gout_id; n < 3; n += gout_stride) {
        double *_gx = g + n * g_size * nsq_per_block;
        double Rpa = rjri[n*nsq_per_block] * aj_aij;
        double c0x = Rpa - rt_aij * Rpq[n*nsq_per_block];
        double s0x, s1x, s2x;
        s0x = _gx[0];
        s1x = c0x * s0x;
        _gx[nsq_per_block] = s1x;
        for (int i = 1; i < LIJ; ++i) {
            s2x = c0x * s1x + i * b10 * s0x;
            _gx[(i+1)*nsq_per_block] = s2x;
            s0x = s1x;
            s1x = s2x;
        }
    }
}

template <int LKL>
__device__ __forceinline__
void trr(double *g, double *rlrk, double *Rpq, double al_akl, double rt_akl,
         double b00, double b01, int lij3, int stride_k, int g_size)
{
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
        __syncthreads();
        int i = n / 3; //for i in range(lij+1):
        int _ix = n % 3; // TODO: remove _ix for nroots > 2
        double *_gx = g + (i + _ix * g_size) * nsq_per_block;
        double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
        double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
        //for i in range(lij+1):
        //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
        double s0x, s1x, s2x;
        if (n < lij3) {
            s0x = _gx[0];
            s1x = cpx * s0x;
            s1x += i * b00 * _gx[-nsq_per_block];
            _gx[stride_k*nsq_per_block] = s1x;
        }

        //for k in range(1, lkl):
        //    for i in range(lij+1):
        //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
        for (int k = 1; k < LKL; ++k) {
            __syncthreads();
            if (n < lij3) {
                s2x = cpx*s1x + k*b01*s0x;
                s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
                _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
    }
}

template <int LI, int LJ>
__device__ __forceinline__
void hrr_ij(double *g, double *rjri, int count, int g_size)
{
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    constexpr int lij = LI + LJ;
    constexpr int stride_j = LI + 1;
    constexpr int stride_k = stride_j * (LJ + 1);
    for (int m = gout_id; m < count; m += gout_stride) {
        int k = m / 3;
        int _ix = m % 3;
        double xjxi = rjri[_ix*nsq_per_block];
        double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
        for (int j = 0; j < LJ; ++j) {
            int ij = (lij-j) + j*stride_j;
            double s0x, s1x;
            s1x = _gx[ij*nsq_per_block];
            for (--ij; ij >= j*stride_j; --ij) {
                s0x = _gx[ij*nsq_per_block];
                _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                s1x = s0x;
            }
        }
    }
}
template <int LK, int LL>
__device__ __forceinline__
void hrr_kl(double *g, double *rlrk, int stride_k)
{
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    constexpr int lkl = LK + LL;
    for (int n = gout_id; n < stride_k*3; n += gout_stride) {
        int i = n / 3;
        int _ix = n % 3;
        double xlxk = rlrk[_ix*nsq_per_block];
        double *_gx = g + (_ix*stride_k*(LK+1)*(LL+1) + i) * nsq_per_block;
        for (int l = 0; l < LL; ++l) {
            int kl = (lkl-l + LK+1)*stride_k;
            double s0x, s1x;
            s1x = _gx[kl*nsq_per_block];
            for (int k = lkl-l-1; k >= 0; --k) {
                int kl = (k + LK+1) * stride_k;
                s0x = _gx[kl*nsq_per_block];
                _gx[(kl+stride_k*(LK+1))*nsq_per_block] = s1x - xlxk * s0x;
                s1x = s0x;
            }
        }
    }
}

template <int I, int J, int K, int L>
__device__ __forceinline__
void inner_dot(double *gout, double *g,
               uint16_t *idx_i, uint16_t *idx_j,
               uint16_t *idx_k, uint16_t *idx_l)
{
#pragma unroll
    for (int l = 0; l < L; ++l) {
        int lxoff = idx_l[l*3+0];
        int lyoff = idx_l[l*3+1];
        int lzoff = idx_l[l*3+2];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        int kxoff = idx_k[k*3+0] + lxoff;
        int kyoff = idx_k[k*3+1] + lyoff;
        int kzoff = idx_k[k*3+2] + lzoff;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        int jxoff = idx_j[j*3+0] + kxoff;;
        int jyoff = idx_j[j*3+1] + kyoff;;
        int jzoff = idx_j[j*3+2] + kzoff;;
#pragma unroll
    for (int i = 0; i < I; ++i) {
        int n = i + j * 3 + k * 9 + l * 27;
        int addrx = idx_i[i*3+0] + jxoff;
        int addry = idx_i[i*3+1] + jyoff;
        int addrz = idx_i[i*3+2] + jzoff;
        gout[n] += g[addrx] * g[addry] * g[addrz];
    } } } }
}

__device__ __forceinline__
void load_dm(double *dm, double *dm_cache, int nao, int nfi, int nfj)
{
    int t_id = threadIdx.y;
    int t_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
    int ni = nfi + 2;
    int nj = nfj + 2;
    for (int m = t_id; m < ni*nj; m += t_stride) {
        int i = m / ni;
        int j = m % nj;
        if (i < nfi && j < nfj) {
            dm_cache[m*nsq_per_block] = dm[i*nao+j];
        } else {
            dm_cache[m*nsq_per_block] = 0;
        }
    }
}

__device__ __forceinline__
void store_vk(double *vk, double *vk_cache, int nao, int nfi, int nfj)
{
    int t_id = threadIdx.y;
    int t_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
    for (int m = t_id; m < nfi*nfj; m += t_stride) {
        int i = m / nfi;
        int j = m % nfj;
        atomicAdd(vk+i*nao+j, vk_cache[m*nsq_per_block]);
    }
}

template <int I, int J, int K, int L>
__device__ __forceinline__
void dot_dm(double *vk, double *dm, double *gout,
            int di, int dl, int nk, int nfl)
{
    int nsq_per_block = blockDim.x;
#pragma unroll
    for (int l = 0; l < 3; ++l) {
        if (l >= dl) break;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (i >= di) break;
        double v = 0;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            int n = i * I + j * J + k * K + l * L;
            v += gout[n] * dm[(j*nk+k)*nsq_per_block];
        } }
        atomicAdd(vk+(i*nfl+l)*nsq_per_block, v);
    } }
}

__global__
void rys_k_kernel_o0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, int *pool)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int *bas_kl_idx = pool + get_smid() * QUEUE_DEPTH;
    __shared__ int ntasks;
    __shared__ double omega;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
        omega = envs.env[PTR_RANGE_OMEGA];
    }
    __syncthreads();
    if (omega >= 0) {
        _fill_k_tasks(&ntasks, bas_kl_idx, envs, bounds);
    } else {
        _fill_sr_k_tasks(&ntasks, bas_kl_idx, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfj = bounds.nfj;
    int nfk = bounds.nfk;
    int nfl = bounds.nfl;
    int nfij = nfi * nfj;
    int nfkl = nfk * nfl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = bounds.g_size;
    int nroots = bounds.nroots;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sq_id;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *aij_cache = Rpq + nsq_per_block * 3;
    double *akl_cache = aij_cache + nsq_per_block * 2;
    double *cicj_cache = akl_cache + nsq_per_block * 2;
    double *rw = cicj_cache + nsq_per_block * iprim * jprim;
    double *gx = rw + nsq_per_block * nroots * 2;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gx + nsq_per_block * g_size * 2;
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int nbas = envs.nbas;
        int *bas = envs.bas;
        double *env = envs.env;
        int li = bounds.li;
        int lj = bounds.lj;
        int lk = bounds.lk;
        int ll = bounds.ll;
        int lij = li + lj;
        int lkl = lk + ll;

        int bas_kl = bas_kl_idx[task_id];
        double fac_sym = PI_FAC;
        int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            rjri[0*nsq_per_block] = xjxi;
            rjri[1*nsq_per_block] = yjyi;
            rjri[2*nsq_per_block] = zjzi;
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double rr_ij = rjri[3*nsq_per_block];
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }

        for (int gout_start = 0; gout_start < nfij*nfkl;
             gout_start+=gout_stride*GOUT_WIDTH0) {
        double gout[GOUT_WIDTH0];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH0; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            if (gout_id == 0) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * rr_kl);
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double xij = ri[0] + rjri[0*nsq_per_block] * aj_aij;
                double yij = ri[1] + rjri[1*nsq_per_block] * aj_aij;
                double zij = ri[2] + rjri[2*nsq_per_block] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp*nsq_per_block];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                    aij_cache[0] = aij;
                    aij_cache[nsq_per_block] = aj_aij;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                int nroots = bounds.nroots;
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
                    double rt_aa = rt / (aij + akl);
                    double s0x, s1x, s2x;

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        double aj_aij = aij_cache[nsq_per_block];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        //switch (lij) {
                        //case 0: vrr<0>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 1: vrr<1>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 2: vrr<2>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 3: vrr<3>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 4: vrr<4>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 5: vrr<5>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 6: vrr<6>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 7: vrr<7>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 8: vrr<8>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //}
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * g_size * nsq_per_block;
                            double Rpa = rjri[n*nsq_per_block] * aj_aij;
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
                        double al_akl = akl_cache[nsq_per_block];
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        int lij3 = (lij+1)*3;
                        //switch (lkl) {
                        //case 0: trr<0>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 1: trr<1>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 2: trr<2>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 3: trr<3>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 4: trr<4>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 5: trr<5>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 6: trr<6>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 7: trr<7>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 8: trr<8>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //}
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                s1x += i * b00 * _gx[-nsq_per_block];
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
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
                        if (task_id < ntasks) {
                            int lkl3 = (lkl+1)*3;
                            //switch (li*5+lj) {
                            //case 0 : hrr_ij<0,0>(gx, rjri, lkl3, g_size); break;
                            //case 1 : hrr_ij<0,1>(gx, rjri, lkl3, g_size); break;
                            //case 2 : hrr_ij<0,2>(gx, rjri, lkl3, g_size); break;
                            //case 3 : hrr_ij<0,3>(gx, rjri, lkl3, g_size); break;
                            //case 4 : hrr_ij<0,4>(gx, rjri, lkl3, g_size); break;
                            //case 5 : hrr_ij<1,0>(gx, rjri, lkl3, g_size); break;
                            //case 6 : hrr_ij<1,1>(gx, rjri, lkl3, g_size); break;
                            //case 7 : hrr_ij<1,2>(gx, rjri, lkl3, g_size); break;
                            //case 8 : hrr_ij<1,3>(gx, rjri, lkl3, g_size); break;
                            //case 9 : hrr_ij<1,4>(gx, rjri, lkl3, g_size); break;
                            //case 10: hrr_ij<2,0>(gx, rjri, lkl3, g_size); break;
                            //case 11: hrr_ij<2,1>(gx, rjri, lkl3, g_size); break;
                            //case 12: hrr_ij<2,2>(gx, rjri, lkl3, g_size); break;
                            //case 13: hrr_ij<2,3>(gx, rjri, lkl3, g_size); break;
                            //case 14: hrr_ij<2,4>(gx, rjri, lkl3, g_size); break;
                            //case 15: hrr_ij<3,0>(gx, rjri, lkl3, g_size); break;
                            //case 16: hrr_ij<3,1>(gx, rjri, lkl3, g_size); break;
                            //case 17: hrr_ij<3,2>(gx, rjri, lkl3, g_size); break;
                            //case 18: hrr_ij<3,3>(gx, rjri, lkl3, g_size); break;
                            //case 19: hrr_ij<3,4>(gx, rjri, lkl3, g_size); break;
                            //case 20: hrr_ij<4,0>(gx, rjri, lkl3, g_size); break;
                            //case 21: hrr_ij<4,1>(gx, rjri, lkl3, g_size); break;
                            //case 22: hrr_ij<4,2>(gx, rjri, lkl3, g_size); break;
                            //case 23: hrr_ij<4,3>(gx, rjri, lkl3, g_size); break;
                            //case 24: hrr_ij<4,4>(gx, rjri, lkl3, g_size); break;
                            //}
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xjxi = rjri[_ix*nsq_per_block];
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
                        if (task_id < ntasks) {
                            //switch (lk*5+ll) {
                            //case 0 : hrr_kl<0,0>(gx, rlrk, stride_k); break;
                            //case 1 : hrr_kl<0,1>(gx, rlrk, stride_k); break;
                            //case 2 : hrr_kl<0,2>(gx, rlrk, stride_k); break;
                            //case 3 : hrr_kl<0,3>(gx, rlrk, stride_k); break;
                            //case 4 : hrr_kl<0,4>(gx, rlrk, stride_k); break;
                            //case 5 : hrr_kl<1,0>(gx, rlrk, stride_k); break;
                            //case 6 : hrr_kl<1,1>(gx, rlrk, stride_k); break;
                            //case 7 : hrr_kl<1,2>(gx, rlrk, stride_k); break;
                            //case 8 : hrr_kl<1,3>(gx, rlrk, stride_k); break;
                            //case 9 : hrr_kl<1,4>(gx, rlrk, stride_k); break;
                            //case 10: hrr_kl<2,0>(gx, rlrk, stride_k); break;
                            //case 11: hrr_kl<2,1>(gx, rlrk, stride_k); break;
                            //case 12: hrr_kl<2,2>(gx, rlrk, stride_k); break;
                            //case 13: hrr_kl<2,3>(gx, rlrk, stride_k); break;
                            //case 14: hrr_kl<2,4>(gx, rlrk, stride_k); break;
                            //case 15: hrr_kl<3,0>(gx, rlrk, stride_k); break;
                            //case 16: hrr_kl<3,1>(gx, rlrk, stride_k); break;
                            //case 17: hrr_kl<3,2>(gx, rlrk, stride_k); break;
                            //case 18: hrr_kl<3,3>(gx, rlrk, stride_k); break;
                            //case 19: hrr_kl<3,4>(gx, rlrk, stride_k); break;
                            //case 20: hrr_kl<4,0>(gx, rlrk, stride_k); break;
                            //case 21: hrr_kl<4,1>(gx, rlrk, stride_k); break;
                            //case 22: hrr_kl<4,2>(gx, rlrk, stride_k); break;
                            //case 23: hrr_kl<4,3>(gx, rlrk, stride_k); break;
                            //case 24: hrr_kl<4,4>(gx, rlrk, stride_k); break;
                            //}
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xlxk = rlrk[_ix*nsq_per_block];
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
                    if (task_id >= ntasks) {
                        continue;
                    }
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH0; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int addrx = (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                        int addry = (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                        int addrz = (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }
        __syncthreads();
        if (task_id >= ntasks) {
            continue;
        }

        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *dm = jk.dm;
        double *vk = jk.vk;
        double* k_cache = shared_memory;
        double* k_ik = k_cache;
        double* k_il = k_ik + nfi * nfk * nsq_per_block;
        double* k_jk = k_il + nfi * nfl * nsq_per_block;
        double* k_jl = k_jk + nfj * nfk * nsq_per_block;
        int jk_cache_size = nfi * nfk + nfi * nfl + nfj * nfk + nfj * nfl;

        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            __syncthreads();
            for (int i = gout_id; i < jk_cache_size; i += gout_stride)
                k_cache[i * nsq_per_block] = 0;
            __syncthreads();
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH0; ++n) {
                int ijkl = (gout_start + n*gout_stride+gout_id);
                int kl = ijkl / nfij;
                int ij = ijkl % nfij;
                if (kl >= nfkl) break;
                double s = gout[n];
                int i = ij % nfi;
                int j = ij / nfi;
                int k = kl % nfk;
                int l = kl / nfk;
                int _i = i + i0;
                int _j = j + j0;
                int _k = k + k0;
                int _l = l + l0;
                // The order of ik,il,jk,jl is consistent with ij,kl
                // which is (j * nfi + i) and (l * nfk + k)
                const int ik = k * nfi + i;
                const int il = l * nfi + i;
                const int jk = k * nfj + j;
                const int jl = l * nfj + j;
                const int _jl = _j*nao+_l;
                const int _jk = _j*nao+_k;
                const int _il = _i*nao+_l;
                const int _ik = _i*nao+_k;
                atomicAdd(k_ik + ik * nsq_per_block, s * dm[_jl]);
                atomicAdd(k_il + il * nsq_per_block, s * dm[_jk]);
                atomicAdd(k_jk + jk * nsq_per_block, s * dm[_il]);
                atomicAdd(k_jl + jl * nsq_per_block, s * dm[_ik]);
            }
            __syncthreads();
            for (int ik = gout_id; ik < nfi * nfk; ik += gout_stride) {
                const int i = ik % nfi;
                const int k = ik / nfi;
                const int _i = i + i0;
                const int _k = k + k0;
                const int _ik = _i*nao+_k;
                atomicAdd(vk + _ik, k_ik[ik * nsq_per_block]);
            }
            for (int il = gout_id; il < nfi * nfl; il += gout_stride) {
                const int i = il % nfi;
                const int l = il / nfi;
                const int _i = i + i0;
                const int _l = l + l0;
                const int _il = _i*nao+_l;
                atomicAdd(vk + _il, k_il[il * nsq_per_block]);
            }
            for (int jk = gout_id; jk < nfj * nfk; jk += gout_stride) {
                const int j = jk % nfj;
                const int k = jk / nfj;
                const int _j = j + j0;
                const int _k = k + k0;
                const int _jk = _j*nao+_k;
                atomicAdd(vk + _jk, k_jk[jk * nsq_per_block]);
            }
            for (int jl = gout_id; jl < nfj * nfl; jl += gout_stride) {
                const int j = jl % nfj;
                const int l = jl / nfj;
                const int _j = j + j0;
                const int _l = l + l0;
                const int _jl = _j*nao+_l;
                atomicAdd(vk + _jl, k_jl[jl * nsq_per_block]);
            }
            __syncthreads();

            vk += nao * nao;
            dm += nao * nao;
        }
    } }
}

__global__
void rys_k_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                  int *pool, GXYZOffsets *gxyz_offsets)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int *bas_kl_idx = pool + get_smid() * QUEUE_DEPTH;
    __shared__ int ntasks;
    __shared__ double omega;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
        omega = envs.env[PTR_RANGE_OMEGA];
    }
    __syncthreads();
    if (omega >= 0) {
        _fill_k_tasks(&ntasks, bas_kl_idx, envs, bounds);
    } else {
        _fill_sr_k_tasks(&ntasks, bas_kl_idx, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfj = bounds.nfj;
    int nfk = bounds.nfk;
    int nfl = bounds.nfl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = bounds.g_size;

    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sq_id;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *aij_cache = Rpq + nsq_per_block * 3;
    double *akl_cache = aij_cache + nsq_per_block * 2;
    double *cicj_cache = akl_cache + nsq_per_block * 2;
    double *rw = cicj_cache + nsq_per_block * iprim * jprim;
    double *gx = rw + nsq_per_block * nroots * 2;
    uint16_t *idx_i = (uint16_t *)(gx + nsq_per_block * g_size * 3);
    uint16_t *idx_j = idx_i + nfi * 3;
    uint16_t *idx_k = idx_i + nfj * 3;
    uint16_t *idx_l = idx_k + nfk * 3;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    if (t_id < nfi * (li+3)) {
        idx_i[t_id] = _c_cartesian_lexical_xyz[nfi*li+t_id] * nsq_per_block;
        idx_i[t_id] += (t_id % 3) * nsq_per_block * g_size;
    }
    if (t_id < nfj * (lj+3)) {
        idx_j[t_id] = _c_cartesian_lexical_xyz[nfj*lj+t_id] * stride_j * nsq_per_block;
    }
    if (t_id < nfk * (lk+3)) {
        idx_k[t_id] = _c_cartesian_lexical_xyz[nfk*lk+t_id] * stride_k * nsq_per_block;
    }
    if (t_id < nfl * (ll+3)) {
        idx_l[t_id] = _c_cartesian_lexical_xyz[nfl*ll+t_id] * stride_l * nsq_per_block;
    }
    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int nbas = envs.nbas;
        int *bas = envs.bas;
        double *env = envs.env;
        int li = bounds.li;
        int lj = bounds.lj;
        int lk = bounds.lk;
        int ll = bounds.ll;
        int iprim = bounds.iprim;
        int jprim = bounds.jprim;
        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        int stride_j = bounds.stride_j;
        int stride_k = bounds.stride_k;
        int stride_l = bounds.stride_l;
        int g_size = bounds.g_size;

        int bas_kl = bas_kl_idx[task_id];
        double fac_sym = PI_FAC;
        int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            rjri[0*nsq_per_block] = xjxi;
            rjri[1*nsq_per_block] = yjyi;
            rjri[2*nsq_per_block] = zjzi;
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double rr_ij = rjri[3*nsq_per_block];
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }

        double gout[GOUT_WIDTH1];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH1; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            if (gout_id == 0) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * rr_kl);
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double xij = ri[0] + rjri[0*nsq_per_block] * aj_aij;
                double yij = ri[1] + rjri[1*nsq_per_block] * aj_aij;
                double zij = ri[2] + rjri[2*nsq_per_block] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp*nsq_per_block];
                    gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                    aij_cache[0] = aij;
                    aij_cache[nsq_per_block] = aj_aij;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                int nroots = bounds.nroots;
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                int lij = li + lj;
                int lkl = lk + ll;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[nsq_per_block*g_size*2] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
                    double rt_aa = rt / (aij + akl);
                    double s0x, s1x, s2x;

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        double aj_aij = aij_cache[nsq_per_block];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        //switch (lij) {
                        //case 0: vrr<0>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 1: vrr<1>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 2: vrr<2>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 3: vrr<3>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 4: vrr<4>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 5: vrr<5>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 6: vrr<6>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 7: vrr<7>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //case 8: vrr<8>(gx, rjri, Rpq, aj_aij, rt_aij, b10, g_size); break;
                        //}
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * g_size * nsq_per_block;
                            double Rpa = rjri[n*nsq_per_block] * aj_aij;
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
                        double al_akl = akl_cache[nsq_per_block];
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        int lij3 = (lij+1)*3;
                        //switch (lkl) {
                        //case 0: trr<0>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 1: trr<1>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 2: trr<2>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 3: trr<3>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 4: trr<4>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 5: trr<5>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 6: trr<6>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 7: trr<7>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //case 8: trr<8>(gx, rlrk, Rpq, al_akl, rt_akl, b00, b01, lij3, stride_k, g_size); break;
                        //}
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                s1x += i * b00 * _gx[-nsq_per_block];
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
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
                        if (task_id < ntasks) {
                            int lkl3 = (lkl+1)*3;
                            //switch (li*5+lj) {
                            //case 0 : hrr_ij<0,0>(gx, rjri, lkl3, g_size); break;
                            //case 1 : hrr_ij<0,1>(gx, rjri, lkl3, g_size); break;
                            //case 2 : hrr_ij<0,2>(gx, rjri, lkl3, g_size); break;
                            //case 3 : hrr_ij<0,3>(gx, rjri, lkl3, g_size); break;
                            //case 4 : hrr_ij<0,4>(gx, rjri, lkl3, g_size); break;
                            //case 5 : hrr_ij<1,0>(gx, rjri, lkl3, g_size); break;
                            //case 6 : hrr_ij<1,1>(gx, rjri, lkl3, g_size); break;
                            //case 7 : hrr_ij<1,2>(gx, rjri, lkl3, g_size); break;
                            //case 8 : hrr_ij<1,3>(gx, rjri, lkl3, g_size); break;
                            //case 9 : hrr_ij<1,4>(gx, rjri, lkl3, g_size); break;
                            //case 10: hrr_ij<2,0>(gx, rjri, lkl3, g_size); break;
                            //case 11: hrr_ij<2,1>(gx, rjri, lkl3, g_size); break;
                            //case 12: hrr_ij<2,2>(gx, rjri, lkl3, g_size); break;
                            //case 13: hrr_ij<2,3>(gx, rjri, lkl3, g_size); break;
                            //case 14: hrr_ij<2,4>(gx, rjri, lkl3, g_size); break;
                            //case 15: hrr_ij<3,0>(gx, rjri, lkl3, g_size); break;
                            //case 16: hrr_ij<3,1>(gx, rjri, lkl3, g_size); break;
                            //case 17: hrr_ij<3,2>(gx, rjri, lkl3, g_size); break;
                            //case 18: hrr_ij<3,3>(gx, rjri, lkl3, g_size); break;
                            //case 19: hrr_ij<3,4>(gx, rjri, lkl3, g_size); break;
                            //case 20: hrr_ij<4,0>(gx, rjri, lkl3, g_size); break;
                            //case 21: hrr_ij<4,1>(gx, rjri, lkl3, g_size); break;
                            //case 22: hrr_ij<4,2>(gx, rjri, lkl3, g_size); break;
                            //case 23: hrr_ij<4,3>(gx, rjri, lkl3, g_size); break;
                            //case 24: hrr_ij<4,4>(gx, rjri, lkl3, g_size); break;
                            //}
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xjxi = rjri[_ix*nsq_per_block];
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
                        if (task_id < ntasks) {
                            //switch (lk*5+ll) {
                            //case 0 : hrr_kl<0,0>(gx, rlrk, stride_k); break;
                            //case 1 : hrr_kl<0,1>(gx, rlrk, stride_k); break;
                            //case 2 : hrr_kl<0,2>(gx, rlrk, stride_k); break;
                            //case 3 : hrr_kl<0,3>(gx, rlrk, stride_k); break;
                            //case 4 : hrr_kl<0,4>(gx, rlrk, stride_k); break;
                            //case 5 : hrr_kl<1,0>(gx, rlrk, stride_k); break;
                            //case 6 : hrr_kl<1,1>(gx, rlrk, stride_k); break;
                            //case 7 : hrr_kl<1,2>(gx, rlrk, stride_k); break;
                            //case 8 : hrr_kl<1,3>(gx, rlrk, stride_k); break;
                            //case 9 : hrr_kl<1,4>(gx, rlrk, stride_k); break;
                            //case 10: hrr_kl<2,0>(gx, rlrk, stride_k); break;
                            //case 11: hrr_kl<2,1>(gx, rlrk, stride_k); break;
                            //case 12: hrr_kl<2,2>(gx, rlrk, stride_k); break;
                            //case 13: hrr_kl<2,3>(gx, rlrk, stride_k); break;
                            //case 14: hrr_kl<2,4>(gx, rlrk, stride_k); break;
                            //case 15: hrr_kl<3,0>(gx, rlrk, stride_k); break;
                            //case 16: hrr_kl<3,1>(gx, rlrk, stride_k); break;
                            //case 17: hrr_kl<3,2>(gx, rlrk, stride_k); break;
                            //case 18: hrr_kl<3,3>(gx, rlrk, stride_k); break;
                            //case 19: hrr_kl<3,4>(gx, rlrk, stride_k); break;
                            //case 20: hrr_kl<4,0>(gx, rlrk, stride_k); break;
                            //case 21: hrr_kl<4,1>(gx, rlrk, stride_k); break;
                            //case 22: hrr_kl<4,2>(gx, rlrk, stride_k); break;
                            //case 23: hrr_kl<4,3>(gx, rlrk, stride_k); break;
                            //case 24: hrr_kl<4,4>(gx, rlrk, stride_k); break;
                            //}
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xlxk = rlrk[_ix*nsq_per_block];
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
                    int nfi = bounds.nfi;
                    int nfj = bounds.nfj;
                    int nfk = bounds.nfk;
                    uint16_t *idx_i = (uint16_t *)(gx + nsq_per_block * g_size * 3);
                    uint16_t *idx_j = idx_i + nfi * 3;
                    uint16_t *idx_k = idx_j + nfj * 3;
                    uint16_t *idx_l = idx_k + nfk * 3;
                    GXYZOffsets goff = gxyz_offsets[gout_id];
                    idx_i += goff.ioff;
                    idx_j += goff.joff;
                    idx_k += goff.koff;
                    idx_l += goff.loff;
                    switch (((li == 0) >> 3) |
                            ((lj == 0) >> 2) |
                            ((lk == 0) >> 1) |
                            ( ll == 0)) {
                    case 0 : inner_dot<3, 3, 3, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 1 : inner_dot<3, 3, 3, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 2 : inner_dot<3, 3, 1, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 3 : inner_dot<3, 3, 1, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 4 : inner_dot<3, 1, 3, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 5 : inner_dot<3, 1, 3, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 6 : inner_dot<3, 1, 1, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 7 : inner_dot<3, 1, 1, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 8 : inner_dot<1, 3, 3, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 9 : inner_dot<1, 3, 3, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 10: inner_dot<1, 3, 1, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 11: inner_dot<1, 3, 1, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 12: inner_dot<1, 1, 3, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 13: inner_dot<1, 1, 3, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 14: inner_dot<1, 1, 1, 3>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    case 15: inner_dot<1, 1, 1, 1>(gout, gx, idx_i, idx_j, idx_k, idx_l); break;
                    }
                }
            }
        }
        __syncthreads();
        if (task_id >= ntasks) {
            continue;
        }

        GXYZOffsets goff = gxyz_offsets[gout_id];
        int ioff = goff.ioff;
        int joff = goff.joff;
        int koff = goff.koff;
        int loff = goff.loff;
        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        int i0 = ao_loc[ish] + ioff;
        int j0 = ao_loc[jsh] + joff;
        int k0 = ao_loc[ksh] + koff;
        int l0 = ao_loc[lsh] + loff;
        int nfi = bounds.nfi;
        int nfj = bounds.nfj;
        int nfk = bounds.nfk;
        int nfl = bounds.nfl;
        int ni = nfi; if (li == 3) ni = 12;
        int nj = nfj; if (lj == 3) nj = 12;
        int nk = nfk; if (lk == 3) nk = 12;
        int nl = nfl; if (ll == 3) nl = 12;
        double *dm_cache = shared_memory + sq_id;
        double *vk_cache = dm_cache + max(ni, nj) * max(nk, nl) * nsq_per_block;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            double *dm = jk.dm + i_dm * nao * nao;
            double *vk = jk.vk + i_dm * nao * nao;
            load_dm(dm+j0*nao+k0, dm_cache, nao, nfj, nfk);
            __syncthreads();
            dot_dm<1, 3, 9, 27>(vk_cache, dm_cache, gout, nfi-ioff, nfl-loff, nk, nfl);
            __syncthreads();
            store_vk(vk+i0*nao+l0, vk_cache, nao, nfi, nfl);

            load_dm(dm+j0*nao+l0, dm_cache, nao, nfj, nfl);
            __syncthreads();
            dot_dm<1, 3, 27, 9>(vk_cache, dm_cache, gout, nfi-ioff, nfk-koff, nl, nfk);
            __syncthreads();
            store_vk(vk+i0*nao+k0, vk_cache, nao, nfi, nfk);

            load_dm(dm+i0*nao+k0, dm_cache, nao, nfi, nfk);
            __syncthreads();
            dot_dm<3, 1, 9, 27>(vk_cache, dm_cache, gout, nfj-joff, nfl-loff, nk, nfl);
            __syncthreads();
            store_vk(vk+j0*nao+l0, vk_cache, nao, nfj, nfl);

            load_dm(dm+i0*nao+l0, dm_cache, nao, nfi, nfl);
            __syncthreads();
            dot_dm<3, 1, 27, 9>(vk_cache, dm_cache, gout, nfj-joff, nfk-koff, nl, nfk);
            __syncthreads();
            store_vk(vk+j0*nao+k0, vk_cache, nao, nfj, nfk);
        }
    }
}
