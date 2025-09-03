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

#include "vhf1.cuh"
#include "rys_roots_for_k.cu"
#include "rys_contract_k.cuh"
#include "create_tasks_ip1_o1.cu"

#define GWIDTH_IP1 27

extern __constant__ int _c_cartesian_lexical_xyz[];

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
        _fill_jk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_jk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
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

extern int rys_vjk_ip1_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int *pool);

extern "C" {
int RYS_build_jk_ip1_o0(double *vj, double *vk, double *dm, int n_dm, int nao, int atom_offset,
                     RysIntEnvVars envs, int *scheme, int *shls_slice,
                     int npairs_ij, int npairs_kl, int *pair_ij_mapping, int *pair_kl_mapping,
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

int RYS_build_vjk_ip1_init(int shm_size)
{
    cudaFuncSetAttribute(rys_vjk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
