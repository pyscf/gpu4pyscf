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
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots_for_k.cu"
#include "rys_contract_k.cuh"
#include "create_tasks.cu"

#define GWIDTH_IP1 27

__global__ static
void rys_vjk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                       int *pool, int reserved_shm_size, int nfij, int nfkl)
{
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
    if (jk.lr_factor != 0) {
        _fill_vjk_tasks_nosym(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_vjk_tasks_nosym(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }

    int g_size = bounds.g_size;
    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *fac_ijkl = shared_memory + nsq_per_block * 8 + sq_id;
    double *gx = shared_memory + nsq_per_block * 9 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+9) + sq_id;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    double *cicj_cache = shared_memory + reserved_shm_size;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lk);
    int *idx_l = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.ll);

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[3];
    __shared__ double *expi;
    __shared__ double *expj;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = nsq_per_block * gout_stride;

    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
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
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
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
        } else {
            fac_sym = 0;
        }
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

        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GWIDTH_IP1) {
            double goutx[GWIDTH_IP1];
            double gouty[GWIDTH_IP1];
            double goutz[GWIDTH_IP1];
#pragma unroll
            for (int n = 0; n < GWIDTH_IP1; ++n) {
                goutx[n] = 0;
                gouty[n] = 0;
                goutz[n] = 0;
            }

        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            __syncthreads();
            if (gout_id == 0) {
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                double fac_sym = fac_ijkl[0];
                gx[0] = fac_sym * ckcl;
                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
            }
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
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
                        aij_cache[2] = ai * 2;
                    }
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                int nroots = bounds.nroots;
                rys_roots_rs(nroots, theta, rr, jk.omega, rw, nsq_per_block, gout_id, gout_stride);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[nsq_per_block*g_size*2] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
                    double rt = rw[irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);
                    double s0x, s1x, s2x;
                    int lij = li + lj + 1;
                    int lkl = lk + ll;

                    __syncthreads();
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = gx + n * g_size * nsq_per_block;
                        double Rpa = rjri[n] * aij_cache[1];
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

                    if (lkl > 0) {
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3;
                            double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                            double al_akl = akl_cache[nsq_per_block];
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
                                    int ij = (lij-j) + j*stride_j;
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
                                    int kl = (lkl-l)*stride_k + l*stride_l;
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
                    int nfi = bounds.nfi;
                    int nfk = bounds.nfk;
#pragma unroll
                    for (int n = 0; n < GWIDTH_IP1; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int ix = idx_i[i*3+0];
                        int iy = idx_i[i*3+1];
                        int iz = idx_i[i*3+2];
                        int jx = idx_j[j*3+0];
                        int jy = idx_j[j*3+1];
                        int jz = idx_j[j*3+2];
                        int kx = idx_k[k*3+0];
                        int ky = idx_k[k*3+1];
                        int kz = idx_k[k*3+2];
                        int lx = idx_l[l*3+0];
                        int ly = idx_l[l*3+1];
                        int lz = idx_l[l*3+2];
                        int addrx = (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = (iy + jy*stride_j + ky*stride_k + ly*stride_l + g_size) * nsq_per_block;
                        int addrz = (iz + jz*stride_j + kz*stride_k + lz*stride_l + g_size*2) * nsq_per_block;
                        double ai2 = aij_cache[2];
                        double fx = ai2 * gx[addrx+nsq_per_block];
                        double fy = ai2 * gx[addry+nsq_per_block];
                        double fz = ai2 * gx[addrz+nsq_per_block];
                        if (ix > 0) { fx -= ix * gx[addrx-nsq_per_block]; }
                        if (iy > 0) { fy -= iy * gx[addry-nsq_per_block]; }
                        if (iz > 0) { fz -= iz * gx[addrz-nsq_per_block]; }
                        goutx[n] += fx * gx[addry] * gx[addrz];
                        gouty[n] += fy * gx[addrx] * gx[addrz];
                        goutz[n] += fz * gx[addrx] * gx[addry];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int ia = bas[ish*BAS_SLOTS+ATOM_OF] - jk.atom_offset;
        double *dm = jk.dm;
        int do_j = jk.vj != NULL;
        int do_k = jk.vk != NULL;
        double *vj_x = jk.vj + (ia*3+0)*nao*nao;
        double *vj_y = jk.vj + (ia*3+1)*nao*nao;
        double *vj_z = jk.vj + (ia*3+2)*nao*nao;
        double *vk_x = jk.vk + (ia*3+0)*nao*nao;
        double *vk_y = jk.vk + (ia*3+1)*nao*nao;
        double *vk_z = jk.vk + (ia*3+2)*nao*nao;
        int nfi = bounds.nfi;
        int nfk = bounds.nfk;
#pragma unroll
        for (int n = 0; n < GWIDTH_IP1; ++n) {
            int ijkl = (gout_start + n*gout_stride+gout_id);
            int kl = ijkl / nfij;
            int ij = ijkl % nfij;
            if (kl >= nfkl) break;
            double sx = goutx[n];
            double sy = gouty[n];
            double sz = goutz[n];
            int i = ij % nfi;
            int j = ij / nfi;
            int k = kl % nfk;
            int l = kl / nfk;
            int _i = i + i0;
            int _j = j + j0;
            int _k = k + k0;
            int _l = l + l0;
            if (do_j) {
                int _ji = _j*nao+_i;
                int _lk = _l*nao+_k;
                atomicAdd(vj_x+_lk, sx * dm[_ji]);
                atomicAdd(vj_y+_lk, sy * dm[_ji]);
                atomicAdd(vj_z+_lk, sz * dm[_ji]);
                atomicAdd(vj_x+_ji, sx * dm[_lk]);
                atomicAdd(vj_y+_ji, sy * dm[_lk]);
                atomicAdd(vj_z+_ji, sz * dm[_lk]);
            }
            if (do_k) {
                int _jl = _j*nao+_l;
                int _jk = _j*nao+_k;
                int _il = _i*nao+_l;
                int _ik = _i*nao+_k;
                atomicAdd(vk_x+_ik, sx * dm[_jl]);
                atomicAdd(vk_y+_ik, sy * dm[_jl]);
                atomicAdd(vk_z+_ik, sz * dm[_jl]);
                atomicAdd(vk_x+_il, sx * dm[_jk]);
                atomicAdd(vk_y+_il, sy * dm[_jk]);
                atomicAdd(vk_z+_il, sz * dm[_jk]);
                atomicAdd(vk_x+_jk, sx * dm[_il]);
                atomicAdd(vk_y+_jk, sy * dm[_il]);
                atomicAdd(vk_z+_jk, sz * dm[_il]);
                atomicAdd(vk_x+_jl, sx * dm[_ik]);
                atomicAdd(vk_y+_jl, sy * dm[_ik]);
                atomicAdd(vk_z+_jl, sz * dm[_ik]);
            }
        }
    } }
}

__global__ static
void rys_ejk_ip1_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                        int *pool, double *dd_pool, int reserved_shm_size)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int smid = get_smid();
    int *bas_kl_idx = pool + smid * QUEUE_DEPTH;
    int nf = bounds.nfi * bounds.nfj * bounds.nfk * bounds.nfl;
    double *dd_cache = dd_pool + smid * nf * blockDim.x + sq_id;
    __shared__ int ntasks;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
    if (jk.lr_factor != 0) {
        _fill_ejk_tasks(&ntasks, bas_kl_idx, bas_ij, jk, envs, bounds);
    } else {
        _fill_sr_ejk_tasks(&ntasks, bas_kl_idx, bas_ij, jk, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }

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
    int nfi = bounds.nfi;
    int nfj = bounds.nfj;
    int nfk = bounds.nfk;
    int nfl = bounds.nfl;
    int nfij = nfi * nfj;
    int nfkl = nfk * nfl;
    int lij = li + lj + 1;
    int lkl = lk + ll + 1;
    int i_1 =          nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *gx = shared_memory + nsq_per_block * 6 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+6) + sq_id;
    double *cicj_cache = shared_memory + reserved_shm_size;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.lk);
    int *idx_l = _c_cartesian_lexical_xyz + lex_xyz_offset(bounds.ll);

    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[nbas];
    double *dm = jk.dm;
    double v_ix = 0;
    double v_iy = 0;
    double v_iz = 0;
    double v_jx = 0;
    double v_jy = 0;
    double v_jz = 0;

    __shared__ int ish, jsh, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    if (thread_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        i0 = ao_loc[ish];
        j0 = ao_loc[jsh];
    }
    __syncthreads();
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
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
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
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
        }

        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        if (jk.n_dm == 1) {
            for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                int kl = n / nfij;
                int ij = n % nfij;
                int i = ij % nfi;
                int j = ij / nfi;
                int k = kl % nfk;
                int l = kl / nfk;
                int _i = i + i0;
                int _j = j + j0;
                int _k = k + k0;
                int _l = l + l0;
                int _jl = _j*nao+_l;
                int _jk = _j*nao+_k;
                int _il = _i*nao+_l;
                int _ik = _i*nao+_k;
                int _ji = _j*nao+_i;
                int _lk = _l*nao+_k;
                double dd = 0;
                if (do_k) {
                    dd += jk.k_factor * (dm[_jk] * dm[_il] + dm[_jl] * dm[_ik]);
                }
                if (do_j) {
                    dd += jk.j_factor * dm[_ji] * dm[_lk];
                }
                dd_cache[n*nsq_per_block] = fac_sym * dd;
            }
        } else {
            double *dmb = dm + nao * nao;
            for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                int kl = n / nfij;
                int ij = n % nfij;
                int i = ij % nfi;
                int j = ij / nfi;
                int k = kl % nfk;
                int l = kl / nfk;
                int _i = i + i0;
                int _j = j + j0;
                int _k = k + k0;
                int _l = l + l0;
                int _jl = _j*nao+_l;
                int _jk = _j*nao+_k;
                int _il = _i*nao+_l;
                int _ik = _i*nao+_k;
                int _ji = _j*nao+_i;
                int _lk = _l*nao+_k;
                double dd = 0;
                if (do_k) {
                    dd += dm [_jk] * dm [_il] + dm [_jl] * dm [_ik];
                    dd += dmb[_jk] * dmb[_il] + dmb[_jl] * dmb[_ik];
                    dd *= jk.k_factor;
                }
                if (do_j) {
                    dd += jk.j_factor * (dm[_ji] + dmb[_ji]) * (dm[_lk] + dmb[_lk]);
                }
                dd_cache[n*nsq_per_block] = fac_sym * dd;
            }
        }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double ak2 = ak * 2;
            double al2 = al * 2;
            if (gout_id == 0) {
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * rr_kl);
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
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
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                int nroots = bounds.nroots;
                rys_roots_for_k(nroots, theta, rr, rw, jk.omega, jk.lr_factor, jk.sr_factor);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[nsq_per_block*g_size*2] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);
                    double s0x, s1x, s2x;
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
                                    int ij = (lij-j) + j*stride_j;
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
                                    int kl = (lkl-l)*stride_k + l*stride_l;
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
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                        int kl = n / nfij;
                        int ij = n % nfij;
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int ix = idx_i[i*3+0];
                        int iy = idx_i[i*3+1];
                        int iz = idx_i[i*3+2];
                        int jx = idx_j[j*3+0];
                        int jy = idx_j[j*3+1];
                        int jz = idx_j[j*3+2];
                        int kx = idx_k[k*3+0];
                        int ky = idx_k[k*3+1];
                        int kz = idx_k[k*3+2];
                        int lx = idx_l[l*3+0];
                        int ly = idx_l[l*3+1];
                        int lz = idx_l[l*3+2];
                        double dd = dd_cache[n*nsq_per_block];
                        int addrx = (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = (iy + jy*stride_j + ky*stride_k + ly*stride_l + g_size) * nsq_per_block;
                        int addrz = (iz + jz*stride_j + kz*stride_k + lz*stride_l + g_size*2) * nsq_per_block;
                        double Ix = gx[addrx];
                        double Iy = gx[addry];
                        double Iz = gx[addrz];
                        double prod_xy = Ix * Iy * dd;
                        double prod_xz = Ix * Iz * dd;
                        double prod_yz = Iy * Iz * dd;
                        double gix = gx[addrx+i_1];
                        double giy = gx[addry+i_1];
                        double giz = gx[addrz+i_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gx[addry+k_1];
                        double gkz = gx[addrz+k_1];
                        double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                        double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                        double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                        double fkx = ak2 * gkx; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; } v_kx += fkx * prod_yz;
                        double fky = ak2 * gky; if (ky > 0) { fky -= ky * gx[addry-k_1]; } v_ky += fky * prod_xz;
                        double fkz = ak2 * gkz; if (kz > 0) { fkz -= kz * gx[addrz-k_1]; } v_kz += fkz * prod_xy;
                        double fjx = aj2 * (gix - rjri[0] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                        double fjy = aj2 * (giy - rjri[1] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                        double fjz = aj2 * (giz - rjri[2] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                        double flx = al2 * (gkx - rlrk[0*nsq_per_block] * Ix); if (lx > 0) { flx -= lx * gx[addrx-l_1]; } v_lx += flx * prod_yz;
                        double fly = al2 * (gky - rlrk[1*nsq_per_block] * Iy); if (ly > 0) { fly -= ly * gx[addry-l_1]; } v_ly += fly * prod_xz;
                        double flz = al2 * (gkz - rlrk[2*nsq_per_block] * Iz); if (lz > 0) { flz -= lz * gx[addrz-l_1]; } v_lz += flz * prod_xy;
                    }
                }
            }
        }
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *reduce = shared_memory + thread_id;
        __syncthreads();
        if (task_id < ntasks) {
            reduce[0*threads] = v_kx;
            reduce[1*threads] = v_ky;
            reduce[2*threads] = v_kz;
            reduce[3*threads] = v_lx;
            reduce[4*threads] = v_ly;
            reduce[5*threads] = v_lz;
        }

        for (int i = gout_stride/2; i > 0; i >>= 1) {
            __syncthreads();
            if (gout_id < i && task_id < ntasks) {
#pragma unroll
                for (int n = 0; n < 6; ++n) {
                    reduce[n*threads] += reduce[n*threads+i*nsq_per_block];
                }
            }
        }
        if (gout_id == 0 && task_id < ntasks) {
            double *ejk = jk.ejk;
            atomicAdd(ejk+ka*3+0, reduce[0*threads]);
            atomicAdd(ejk+ka*3+1, reduce[1*threads]);
            atomicAdd(ejk+ka*3+2, reduce[2*threads]);
            atomicAdd(ejk+la*3+0, reduce[3*threads]);
            atomicAdd(ejk+la*3+1, reduce[4*threads]);
            atomicAdd(ejk+la*3+2, reduce[5*threads]);
        }
    }
    int ia = bas[ish*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
    double *reduce = shared_memory + thread_id;
    __syncthreads();
    reduce[0*threads] = v_ix;
    reduce[1*threads] = v_iy;
    reduce[2*threads] = v_iz;
    reduce[3*threads] = v_jx;
    reduce[4*threads] = v_jy;
    reduce[5*threads] = v_jz;

    for (int i = gout_stride/2; i > 0; i >>= 1) {
        __syncthreads();
        if (gout_id < i) {
#pragma unroll
            for (int n = 0; n < 6; ++n) {
                reduce[n*threads] += reduce[n*threads+i*nsq_per_block];
            }
        }
    }
    if (gout_id == 0) {
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, reduce[0*threads]);
        atomicAdd(ejk+ia*3+1, reduce[1*threads]);
        atomicAdd(ejk+ia*3+2, reduce[2*threads]);
        atomicAdd(ejk+ja*3+0, reduce[3*threads]);
        atomicAdd(ejk+ja*3+1, reduce[4*threads]);
        atomicAdd(ejk+ja*3+2, reduce[5*threads]);
    }
}

extern int rys_vjk_ip1_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int *pool);

extern int rys_ejk_ip1_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                        int *pool, double *dd_pool);

extern "C" {
int RYS_build_jk_ip1(double *vj, double *vk, double *dm, int n_dm, int nao, int atom_offset,
                     RysIntEnvVars envs, int *scheme, int *shls_slice,
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
    int nfij = nfi * nfj;
    int nfkl = nfk * nfl;
    int order = li + lj + lk + ll;
    int nroots = (order + 1) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 1);
    uint16_t g_size = stride_l * (ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, vk, dm, n_dm, atom_offset, omega};
    if (omega >= 0) {
        jk.lr_factor = 1;
        jk.sr_factor = 0;
    } else {
        jk.lr_factor = 0;
        jk.sr_factor = 1;
    }

    if (!rys_vjk_ip1_unrolled(&envs, &jk, &bounds, pool)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int reserved_shm_size = (nroots*2 + g_size*3 + 9) * quartets_per_block;
        int buflen = reserved_shm_size + ij_prims;

        rys_vjk_ip1_kernel<<<npairs_ij, threads, buflen*sizeof(double)>>>(
            envs, jk, bounds, pool, reserved_shm_size, nfij, nfkl);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in RYS_build_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_per_atom_jk_ip1(double *ejk, double j_factor, double k_factor,
                        double *dm, int n_dm, int nao,
                        RysIntEnvVars envs, int *scheme, int *shls_slice,
                        int npairs_ij, int npairs_kl,
                        uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
                        float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                        int *pool, double *dd_pool,
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
    int order = li + lj + lk + ll;
    int nroots = (order + 1) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 2);
    int g_size = stride_l * (ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff};

    if (n_dm == 1) { // RHF
        k_factor *= .5;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 2.*j_factor, -k_factor, n_dm, omega};
    if (omega >= 0) {
        jk.lr_factor = 1;
        jk.sr_factor = 0;
    } else {
        jk.lr_factor = 0;
        jk.sr_factor = 1;
    }

    if (!rys_ejk_ip1_unrolled(&envs, &jk, &bounds, pool, dd_pool)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + 6) * quartets_per_block;
        int reserved_shm_size = MAX(buflen, 6*gout_stride*quartets_per_block);
        buflen = (reserved_shm_size + ij_prims)*sizeof(double);

        rys_ejk_ip1_kernel<<<npairs_ij, threads, buflen>>>(
                envs, jk, bounds, pool, dd_pool, reserved_shm_size);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_build_vjk_ip1_init(int shm_size)
{
    cudaFuncSetAttribute(rys_vjk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_ejk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
