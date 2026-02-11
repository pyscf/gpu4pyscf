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

#include "gint/cuda_alloc.cuh"
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
//#include "gvhf-rys/create_tasks.cu"
#include "gvhf-rys/rys_contract_k.cuh"
//#include "pbc.cuh"
#include "create_tasks.cu"

#ifdef USE_SYCL
SYCL_EXTERNAL sycl_device_global<GXYZOffset[625]> s_gxyz_offset;
#endif

#define GOUT_WIDTH1     81

// gout_pattern = ((li == 0) >> 3) | ((lj == 0) >> 2) | ((lk == 0) >> 1) | (ll == 0);
template <int OFFSET>
__global__ static
void rys_k_kernel(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                  int *bas_mask_idx, int *Ts_ij_lookup,
                  int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                  uint32_t *pool, int *head, GXYZOffset *p_gxyz_offsets,
                  int gout_pattern, int reserved_shm_size
                  #ifdef USE_SYCL
                  , sycl::nd_item<2> &item, char *shm_mem
                  #endif
                  )
{
    // sq is short for shl_quartet
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    int gout_id = item.get_local_id(0);
    int gout_stride = item.get_local_range(0);
    int thread_id = item.get_local_id(1) + item.get_local_range(1) * item.get_local_id(0);

    int t_id = item.get_local_id(0) * item.get_local_range(1) + item.get_local_id(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    double *shared_memory = reinterpret_cast<double *>(shm_mem);

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &cell_j = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish_cell0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh_cell0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &i0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &j0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

    auto gxyz_offsets = s_gxyz_offset.get() + OFFSET;
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;

    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];
    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;

    const GXYZOffset *gxyz_offsets = p_gxyz_offsets + OFFSET;
    #endif

    // sq is short for shl_quartet
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = bounds.g_size;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *fac_ijkl = shared_memory + nsq_per_block * 8 + sq_id;
    double *gx = shared_memory + nsq_per_block * 9 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+9) + sq_id;
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    double *cicj_cache = shared_memory + reserved_shm_size - iprim*jprim;
    int *idx_i = (int*)(shared_memory + reserved_shm_size);
    int *idx_j = idx_i + ntiles_i * 9;
    int *idx_k = idx_j + ntiles_j * 9;
    int *idx_l = idx_k + ntiles_k * 9;
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

while (1) {
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int threads = nsq_per_block * gout_stride;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (thread_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, bas_ij, bas_mask_idx,
                          Ts_ij_lookup, nimgs, nbas_cell0, envs, bounds);
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
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

            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0*nsq_per_block] = xlxk;
                rlrk[1*nsq_per_block] = ylyk;
                rlrk[2*nsq_per_block] = zlzk;
                fac_ijkl[0] = fac_sym;
            }

            double gout[GOUT_WIDTH1];
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH1; ++n) { gout[n] = 0; }

            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0*nsq_per_block];
                    double ylyk = rlrk[1*nsq_per_block];
                    double zlzk = rlrk[2*nsq_per_block];
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = fac_ijkl[0];
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + (rjri[0]) * aj_aij;
                    double yij = ri[1] + (rjri[1]) * aj_aij;
                    double zij = ri[2] + (rjri[2]) * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
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
                            double aj_aij = aij_cache[1];
                            double rt_aij = rt_aa * akl;
                            double b10 = .5/aij * (1 - rt_aij);
                            __syncthreads();
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * g_size * nsq_per_block;
                                double Rpa = (rjri[n]) * aj_aij;
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
                            if (task_id < ntasks) {
                                int lkl3 = (lkl+1)*3;
                                for (int m = gout_id; m < lkl3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix];
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
            }
            __syncthreads();

            GXYZOffset goff = gxyz_offsets[gout_id];
            int ioff = goff.ioff;
            int joff = goff.joff;
            int koff = goff.koff;
            int loff = goff.loff;
            int *ao_loc = envs.ao_loc;
            int k0 = ao_loc[ksh_cell0];
            int l0 = ao_loc[lsh_cell0];
            int nao2 = nao * nao;
            int dm_size = nao2 * nimgs_uniq_pair;
            int nfi = bounds.nfi;
            int nfj = bounds.nfj;
            int nfk = bounds.nfk;
            int nfl = bounds.nfl;
            double *dm_cache = shared_memory + sq_id;
            int active = task_id < ntasks;
            for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                int ldi = bounds.ntiles_i * 3;
                int ldj = bounds.ntiles_j * 3;
                int ldk = bounds.ntiles_k * 3;
                int ldl = bounds.ntiles_l * 3;
                double *vk = kmat.vk + i_dm * dm_size;
                double *dm = kmat.dm + i_dm * dm_size;
                double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                double *vk_il = vk + Ts_ij_lookup[cell_l] * nao2;
                double *vk_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                load_dm(dm_jk+j0*nao+k0, dm_cache, nao, nfj, nfk, ldj, ldk);
                dot_dm<1, 3, 9, 27>(vk_il, dm_cache, gout, nao, i0, l0,
                                    ioff, joff, koff, loff, ldk, nfi, nfl, active);
                load_dm(dm_jl+j0*nao+l0, dm_cache, nao, nfj, nfl, ldj, ldl);
                dot_dm<1, 3, 27, 9>(vk_ik, dm_cache, gout, nao, i0, k0,
                                    ioff, joff, loff, koff, ldl, nfi, nfk, active);
                if (ish_cell0 != jsh_cell0) {
                    double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                    double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                    double *vk_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                    double *vk_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                    load_dm(dm_ik+i0*nao+k0, dm_cache, nao, nfi, nfk, ldi, ldk);
                    dot_dm<3, 1, 9, 27>(vk_jl, dm_cache, gout, nao, j0, l0,
                                        joff, ioff, koff, loff, ldk, nfj, nfl, active);
                    load_dm(dm_il+i0*nao+l0, dm_cache, nao, nfi, nfl, ldi, ldl);
                    dot_dm<3, 1, 27, 9>(vk_jk, dm_cache, gout, nao, j0, k0,
                                        joff, ioff, loff, koff, ldl, nfj, nfk, active);
                }
            }
        }
    }
}
}

GXYZOffset *RYS_make_gxyz_offset(BoundsInfo &bounds)
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
    #ifdef USE_SYCL
    sycl_get_queue()->memcpy(s_gxyz_offset, goff, max(nf, 256)*sizeof(GXYZOffset)).wait();
    return nullptr;
    #else
    checkCudaErrors(
        cudaMemcpyToSymbol(c_gxyz_offset, goff, max(nf, 256)*sizeof(GXYZOffset),
                           0, cudaMemcpyHostToDevice));
    GXYZOffset *p_gxyz_offset;
    cudaGetSymbolAddress((void**)&p_gxyz_offset, c_gxyz_offset);
    return p_gxyz_offset;
    #endif
}

static size_t threads_scheme_for_k(int (&threads)[2], BoundsInfo &bounds,
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
    int ijprim = bounds.iprim * bounds.jprim;
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
    int root_g_cache_size = nroots*2 + g_size*3 + 9;
    int unit = max(root_g_cache_size, dm_cache_size);
    int counts = (shm_size - cart_idx_size*4 - ijprim*8) / (unit*8);
    int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
    int THREADS = 256;
    int gout_stride = min(n_tiles, gout_stride_max);
    int nsq_per_block = min(counts, THREADS / gout_stride);
    if (nsq_per_block > 8) {
        nsq_per_block = nsq_per_block / 8 * 8;
    }
    threads[0] = nsq_per_block;
    threads[1] = gout_stride;
    int buflen = nsq_per_block * unit*8 + cart_idx_size*4 + ijprim*8;
    return buflen;
}

//extern int rys_k_unrolled(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds, int *pool);

extern "C" {
int PBC_build_k(double *vk, double *dm, int n_dm, int nao,
                RysIntEnvVars *envs, int *shls_slice, int shm_size,
                int npairs_ij, int npairs_kl,
                uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup, int nimgs, int nimgs_uniq_pair,
                float *q_cond, float *s_estimator, float *dm_cond,
                float cutoff, uint32_t *pool, int nbas_cell0,
                int *atm, int natm, int *bas, int nbas, double *env)
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
    nroots *= 2; // SR ERIs
    double omega = -fabs(env[PTR_RANGE_OMEGA]);
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size = stride_l * (ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff,
        ntiles_i, ntiles_j, ntiles_k, ntiles_l};

    JKMatrix kmat = {NULL, vk, dm, n_dm, 0, omega};
    kmat.lr_factor = 0;
    kmat.sr_factor = 1;

    #ifdef USE_SYCL
    int workers = sycl_get_queue()->get_device().get_info<sycl::info::device::max_compute_units>();
    #else
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    #endif
    int *head = (int *)(pool + workers * QUEUE_DEPTH);
    cudaMemset(head, 0, sizeof(int));

    if (1){//!rys_k_unrolled(envs, &kmat, &bounds, pool)) {
      GXYZOffset* p_gxyz_offset = RYS_make_gxyz_offset(bounds);
      int gout_pattern = (((li == 0) >> 3) |
                          ((lj == 0) >> 2) |
                          ((lk == 0) >> 1) |
                          ( ll == 0));
      int threads[2];
      int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
      int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;

      auto launch = [&](auto offset, int tile_chunk) {
        constexpr int OFF = decltype(offset)::value;
        int buflen = threads_scheme_for_k(threads, bounds, shm_size, tile_chunk);
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;

        #ifdef USE_SYCL
        sycl::range<2> blocks(1, workers);
        sycl::range<2> cuda_threads(threads[1], threads[0]);
        auto dev_envs = *envs;
        sycl_get_queue()->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<char, 1> local_acc(sycl::range<1>(buflen), cgh);
          cgh.parallel_for(sycl::nd_range<2>(blocks * cuda_threads, cuda_threads), [=](auto item) {
            rys_k_kernel<OFF>(dev_envs, kmat, bounds, bas_mask_idx, Ts_ij_lookup,
                              nimgs, nimgs_uniq_pair, nbas_cell0, nao,
                              pool, head, p_gxyz_offset, gout_pattern, reserved_shm_size,
                              item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
          });
        });
        #else
        dim3 cuda_threads(threads[0], threads[1]);
        rys_k_kernel<OFF><<<workers, cuda_threads, buflen>>>(
            *envs, kmat, bounds, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao,
            pool, head, p_gxyz_offset, gout_pattern, reserved_shm_size);
        #endif
      };

      launch(std::integral_constant<int, 0>{}, 256);
      if (n_tiles > 256)  launch(std::integral_constant<int, 256>{}, std::min(256, n_tiles - 256));
      if (n_tiles > 512)  launch(std::integral_constant<int, 512>{}, std::min(256, n_tiles - 512));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in PBC_build_k, li,lj,lk,ll = %d,%d,%d,%d, "
                "device_id = %d, error message = %s\n",
                li,lj,lk,ll, device_id, cudaGetErrorString(err));
        fflush(stderr);
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
