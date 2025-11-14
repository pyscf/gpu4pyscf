/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

#include "gint/cuda_alloc.cuh"
#include "vhf.cuh"
#include "rys_roots.cu"
#include "rys_contract_k.cuh"
#include "create_tasks.cu"

#define GOUT_WIDTH1     81

__global__ static
void rys_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   int *pool, GXYZOffset *gxyz_offsets,
                   int gout_pattern, int reserved_shm_size)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int smid = get_smid();
    int *bas_kl_idx = pool + smid * QUEUE_DEPTH;
    __shared__ int ntasks;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
    if (jk.omega >= 0) {
        _fill_vjk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_vjk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }

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

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    __shared__ int i0, j0, nao;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ double *expi;
    __shared__ double *expj;
    if (t_id == 0) {
        int *ao_loc = envs.ao_loc;
        nao = ao_loc[nbas];
        i0 = ao_loc[ish];
        j0 = ao_loc[jsh];
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
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
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        double cicj = ci[ip] * cj[jp];
        if (ish == jsh) {
            cicj *= .5;
        }
        cicj_cache[ij] = cicj * Kab;
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
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ksh == lsh) fac_sym *= .5;
            if (bas_ij == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
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
                double fac_sym = fac_ijkl[0];
                gx[0] = fac_sym * ckcl;
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
                double xij = ri[0] + (rjri[0]) * aj_aij;
                double yij = ri[1] + (rjri[1]) * aj_aij;
                double zij = ri[2] + (rjri[2]) * aj_aij;
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
                rys_roots_rs(nroots, theta, rr, jk.omega, rw, nsq_per_block,
                             gout_id, gout_stride);
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

        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            GXYZOffset goff = gxyz_offsets[gout_id];
            int ioff = goff.ioff;
            int joff = goff.joff;
            int koff = goff.koff;
            int loff = goff.loff;
            int *ao_loc = envs.ao_loc;
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
            int active = task_id < ntasks;
            double *dm = jk.dm + i_dm * nao * nao;
            double *vk = jk.vk + i_dm * nao * nao;
            double *vj = jk.vj + i_dm * nao * nao;
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

            load_dm(dm+i0*nao+j0, dm_cache, nao, nfi, nfj, ldi, ldj, active);
            dot_dm<9, 1, 3, 27>(vj, dm_cache, gout, nao, k0, l0,
                                koff, ioff, joff, loff, ldj, nfk, nfl, active);
            load_dm(dm+k0*nao+l0, dm_cache, nao, nfk, nfl, ldk, ldl, active);
            dot_dm<1, 9, 27, 3>(vj, dm_cache, gout, nao, i0, j0,
                                ioff, koff, loff, joff, ldl, nfi, nfj, active);
        }
    }
}

static size_t threads_scheme_for_jk(dim3& threads, BoundsInfo &bounds,
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
    dm_cache_size = max(dm_cache_size, ldi*ldj);
    dm_cache_size = max(dm_cache_size, ldk*ldl);
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
    threads.x = nsq_per_block;
    threads.y = gout_stride;
    int buflen = nsq_per_block * unit*8 + cart_idx_size*4 + ijprim*8;
    return buflen;
}

extern GXYZOffset *RYS_make_gxyz_offset(BoundsInfo &bounds);
extern int rys_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int *pool);

extern "C" {
int RYS_build_jk(double *vj, double *vk, double *dm, int n_dm, int nao,
                 RysIntEnvVars *envs, int *shls_slice, int shm_size,
                 int npairs_ij, int npairs_kl,
                 uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
                 float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                 int *pool, int *atm, int natm, int *bas, int nbas, double *env)
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
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff,
        ntiles_i, ntiles_j, ntiles_k, ntiles_l};

    JKMatrix jk = {vj, vk, dm, n_dm, 0, omega};
    if (!rys_jk_unrolled(envs, &jk, &bounds, pool)) {
        GXYZOffset* p_gxyz_offset = RYS_make_gxyz_offset(bounds);
        int gout_pattern = (((li == 0) >> 3) |
                            ((lj == 0) >> 2) |
                            ((lk == 0) >> 1) |
                            ( ll == 0));
        dim3 threads;
        int buflen = threads_scheme_for_jk(threads, bounds, shm_size, 256);
        int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;

        rys_jk_kernel<<<npairs_ij, threads, buflen>>>(
            *envs, jk, bounds, pool, p_gxyz_offset,
            gout_pattern, reserved_shm_size);

        int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
        if (n_tiles > 256) { // fffg, ffgg, fggg, gggg
            buflen = threads_scheme_for_jk(threads, bounds, shm_size,
                                           min(256, n_tiles-256));
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;
            rys_jk_kernel<<<npairs_ij, threads, buflen>>>(
                *envs, jk, bounds, pool, p_gxyz_offset+256,
                gout_pattern, reserved_shm_size);
        }

        if (n_tiles > 512) { // gggg
            buflen = threads_scheme_for_jk(threads, bounds, shm_size,
                                           min(256, n_tiles-512));
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;
            rys_jk_kernel<<<npairs_ij, threads, buflen>>>(
                *envs, jk, bounds, pool, p_gxyz_offset+512,
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
        fprintf(stderr, "CUDA Error in RYS_build_jk, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_build_jk_init(int shm_size)
{
    cudaFuncSetAttribute(rys_jk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
