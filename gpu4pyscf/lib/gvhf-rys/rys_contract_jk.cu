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

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"

__device__
static void rys_jk_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                           ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfk = bounds.nfk;
    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    //double *env = c_env;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];

    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3];
    double gout[GWIDTH];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        //int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
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
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xjxi * aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yjyi * aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zjzi * aj_aij;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GWIDTH) {
#pragma unroll
        for (int n = 0; n < GWIDTH; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            Rqc[0] = xlxk * al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ylyk * al_akl;
            Rqc[2] = zlzk * al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp*4*nsq_per_block;
                double xij = ri[0] + Rpa[sq_id+0*nsq_per_block];
                double yij = ri[1] + Rpa[sq_id+1*nsq_per_block];
                double zij = ri[2] + Rpa[sq_id+2*nsq_per_block];
                double xkl = rk[0] + Rqc[0];
                double ykl = rk[1] + Rqc[1];
                double zkl = rk[2] + Rqc[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = Rpa[sq_id+3*nsq_per_block];
                    g[sq_id + g_size * nsq_per_block] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(nroots, theta_rr, rw);
                } else {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        g[sq_id + 2*g_size*nsq_per_block] = rw[sq_id+(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[sq_id + irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = g + n * g_size * nsq_per_block;
                            int ir = sq_id + n * nsq_per_block;
                            double c0x = Rpa[ir] - rt_aij * Rpq[n];
                            s0x = _gx[sq_id];
                            s1x = c0x * s0x;
                            _gx[sq_id + nsq_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[sq_id + (i+1)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lkl > 0) {
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            double cpx = Rqc[_ix] + rt_akl * Rpq[_ix];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[sq_id];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[sq_id-nsq_per_block];
                                }
                                _gx[sq_id + stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[sq_id + (k*stride_k-1)*nsq_per_block];
                                    }
                                    _gx[sq_id + (k*stride_k+stride_k)*nsq_per_block] = s2x;
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
                                double xixj = ri[_ix] - rj[_ix];
                                double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = (lij-j) + j*stride_j;
                                    s1x = _gx[sq_id + ij*nsq_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[sq_id + ij*nsq_per_block];
                                        _gx[sq_id + (ij+stride_j)*nsq_per_block] = xixj * s0x + s1x;
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
                                double xkxl = rk[_ix] - rl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
                                    int kl = (lkl-l)*stride_k + l*stride_l;
                                    s1x = _gx[sq_id + kl*nsq_per_block];
                                    for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                        s0x = _gx[sq_id + kl*nsq_per_block];
                                        _gx[sq_id + (kl+stride_l)*nsq_per_block] = xkxl * s0x + s1x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
#pragma unroll
                    for (int n = 0; n < GWIDTH; ++n) {
                        int ijkl = (gout_start + n*gout_stride+gout_id);
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int addrx = sq_id + (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                        int addry = sq_id + (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                        int addrz = sq_id + (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *dm = jk.dm;
        double *vj = jk.vj;
        double *vk = jk.vk;
        int do_j = vj != NULL;
        int do_k = vk != NULL;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
#pragma unroll
            for (int n = 0; n < GWIDTH; ++n) {
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
                if (do_j) {
                    int _ji = _j*nao+_i;
                    int _lk = _l*nao+_k;
                    atomicAdd(vj+_lk, s * dm[_ji]);
                    atomicAdd(vj+_ji, s * dm[_lk]);
                }
                if (do_k) {
                    int _jl = _j*nao+_l;
                    int _jk = _j*nao+_k;
                    int _il = _i*nao+_l;
                    int _ik = _i*nao+_k;
                    atomicAdd(vk+_ik, s * dm[_jl]);
                    atomicAdd(vk+_il, s * dm[_jk]);
                    atomicAdd(vk+_jk, s * dm[_il]);
                    atomicAdd(vk+_jl, s * dm[_ik]);
                }
            }
            vj += nao * nao;
            vk += nao * nao;
            dm += nao * nao;
        }
    } }
}

__device__
static void rys_sr_jk_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                              ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfk = bounds.nfk;
    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    //double *env = c_env;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];

    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3];
    double gout[GWIDTH];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        //int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
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
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xjxi * aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yjyi * aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zjzi * aj_aij;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GWIDTH) {
#pragma unroll
        for (int n = 0; n < GWIDTH; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            Rqc[0] = xlxk * al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ylyk * al_akl;
            Rqc[2] = zlzk * al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp*4*nsq_per_block;
                double xij = ri[0] + Rpa[sq_id+0*nsq_per_block];
                double yij = ri[1] + Rpa[sq_id+1*nsq_per_block];
                double zij = ri[2] + Rpa[sq_id+2*nsq_per_block];
                double xkl = rk[0] + Rqc[0];
                double ykl = rk[1] + Rqc[1];
                double zkl = rk[2] + Rqc[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = Rpa[sq_id+3*nsq_per_block];
                    g[sq_id + g_size * nsq_per_block] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double theta_rr = theta * rr;
                int _nroots = nroots/2;
                rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                double theta_fac = omega * omega / (omega * omega + theta);
                rys_roots(_nroots, theta_fac*theta_rr, rw);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                    rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                }
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        g[sq_id + 2*g_size*nsq_per_block] = rw[sq_id+(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[sq_id + irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = g + n * g_size * nsq_per_block;
                            int ir = sq_id + n * nsq_per_block;
                            double c0x = Rpa[ir] - rt_aij * Rpq[n];
                            s0x = _gx[sq_id];
                            s1x = c0x * s0x;
                            _gx[sq_id + nsq_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[sq_id + (i+1)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lkl > 0) {
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            double cpx = Rqc[_ix] + rt_akl * Rpq[_ix];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[sq_id];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[sq_id-nsq_per_block];
                                }
                                _gx[sq_id + stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[sq_id + (k*stride_k-1)*nsq_per_block];
                                    }
                                    _gx[sq_id + (k*stride_k+stride_k)*nsq_per_block] = s2x;
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
                                double xixj = ri[_ix] - rj[_ix];
                                double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = (lij-j) + j*stride_j;
                                    s1x = _gx[sq_id + ij*nsq_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[sq_id + ij*nsq_per_block];
                                        _gx[sq_id + (ij+stride_j)*nsq_per_block] = xixj * s0x + s1x;
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
                                double xkxl = rk[_ix] - rl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
                                    int kl = (lkl-l)*stride_k + l*stride_l;
                                    s1x = _gx[sq_id + kl*nsq_per_block];
                                    for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                        s0x = _gx[sq_id + kl*nsq_per_block];
                                        _gx[sq_id + (kl+stride_l)*nsq_per_block] = xkxl * s0x + s1x;
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
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
#pragma unroll
                    for (int n = 0; n < GWIDTH; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int addrx = sq_id + (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                        int addry = sq_id + (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                        int addrz = sq_id + (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *dm = jk.dm;
        double *vj = jk.vj;
        double *vk = jk.vk;
        int do_j = vj != NULL;
        int do_k = vk != NULL;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
#pragma unroll
            for (int n = 0; n < GWIDTH; ++n) {
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
                if (do_j) {
                    int _ji = _j*nao+_i;
                    int _lk = _l*nao+_k;
                    atomicAdd(vj+_lk, s * dm[_ji]);
                    atomicAdd(vj+_ji, s * dm[_lk]);
                }
                if (do_k) {
                    int _jl = _j*nao+_l;
                    int _jk = _j*nao+_k;
                    int _il = _i*nao+_l;
                    int _ik = _i*nao+_k;
                    atomicAdd(vk+_ik, s * dm[_jl]);
                    atomicAdd(vk+_il, s * dm[_jk]);
                    atomicAdd(vk+_jk, s * dm[_il]);
                    atomicAdd(vk+_jl, s * dm[_ik]);
                }
            }
            vj += nao * nao;
            vk += nao * nao;
            dm += nao * nao;
        }
    } }
}

__global__
void rys_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        int ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        if (ntasks > 0) {
            rys_jk_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_sr_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        int ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_sr_jk_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
