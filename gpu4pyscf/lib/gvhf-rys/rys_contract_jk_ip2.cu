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

// type 1: (d^2i j | k l)
// type 2: (di dj | k l)
// type 3: (di j | dk l)

__global__ static
void rys_ejk_ip2_type12_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                               int *pool, double *dd_pool, int lij, int lkl)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
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
    int nroots = bounds.nroots;
    //int lij = li + lj + 2;
    //int lkl = lk + ll + 2;
    int i_1 =          nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+nroots*2+8);
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);
    int *idx_l = _c_cartesian_lexical_xyz + lex_xyz_offset(ll);
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[4];
    __shared__ double *expi;
    __shared__ double *expj;
    if (thread_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
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
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        double *dm = jk.dm;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
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
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;
        int nfi = bounds.nfi;
        int nfj = bounds.nfj;
        int nfk = bounds.nfk;
        int nfl = bounds.nfl;
        int nfij = nfi * nfj;
        int nfkl = nfk * nfl;
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
                    dd += jk.j_factor * (dm[_ji] + dm[_ji]) * (dmb[_lk] + dmb[_lk]);
                }
                dd_cache[n*nsq_per_block] = fac_sym * dd;
            }
        }

        __shared__ int klp;
        if (sq_id == 0 && gout_id == 0) {
            klp = 0;
        }
        __syncthreads();
        while (klp < kprim*lprim) {
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
            __syncthreads();
            if (gout_id == 0) {
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
                if (sq_id == 0) {
                    klp++;
                }
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
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
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
                        aij_cache[3] = aj * 2;
                    }
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
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
                        double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                        double Rqc = rlrk[_ix*nsq_per_block] * akl_cache[nsq_per_block];
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
                    __syncthreads();
                    if (task_id < ntasks) {
                        int lkl3 = (lkl+1)*3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xjxi = rjri[_ix];
                            double *_gx = gx + (_ix*g_size + k*stride_k) * nsq_per_block;
                            for (int j = 0; j <= lj; ++j) {
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
                    __syncthreads();
                    if (task_id < ntasks) {
                        for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                            int i = n / 3;
                            int _ix = n % 3;
                            double xlxk = rlrk[_ix*nsq_per_block];
                            double *_gx = gx + (_ix*g_size + i) * nsq_per_block;
                            for (int l = 0; l <= ll; ++l) {
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
                        double Ixdd = Ix * dd;
                        double Iydd = Iy * dd;
                        double Izdd = Iz * dd;
                        double prod_yz = Iy * Izdd;
                        double prod_xz = Ix * Izdd;
                        double prod_xy = Ix * Iydd;
                        double gix = gx[addrx+i_1];
                        double giy = gx[addry+i_1];
                        double giz = gx[addrz+i_1];
                        double gjx = gx[addrx+j_1];
                        double gjy = gx[addry+j_1];
                        double gjz = gx[addrz+j_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gx[addry+k_1];
                        double gkz = gx[addrz+k_1];
                        double glx = gx[addrx+l_1];
                        double gly = gx[addry+l_1];
                        double glz = gx[addrz+l_1];

                        double f1x, f1y, f1z;
                        double f2x, f2y, f2z;
                        double f3x, f3y, f3z;
                        double _gx_inc2, _gy_inc2, _gz_inc2;
                        double ai2 = aij_cache[2];
                        double aj2 = aij_cache[3];
                        f1x = aj2 * gjx;
                        f1y = aj2 * gjy;
                        f1z = aj2 * gjz;
                        if (jx > 0) { f1x -= jx * gx[addrx-j_1]; }
                        if (jy > 0) { f1y -= jy * gx[addry-j_1]; }
                        if (jz > 0) { f1z -= jz * gx[addrz-j_1]; }

                        f2x = ai2 * gix;
                        f2y = ai2 * giy;
                        f2z = ai2 * giz;
                        if (ix > 0) { f2x -= ix * gx[addrx-i_1]; }
                        if (iy > 0) { f2y -= iy * gx[addry-i_1]; }
                        if (iz > 0) { f2z -= iz * gx[addrz-i_1]; }

                        double gijx = gx[addrx+i_1+j_1];
                        double gijy = gx[addry+i_1+j_1];
                        double gijz = gx[addrz+i_1+j_1];
                        f3x = ai2 * gijx;
                        f3y = ai2 * gijy;
                        f3z = ai2 * gijz;
                        if (ix > 0) { f3x -= ix * gx[addrx-i_1+j_1]; }
                        if (iy > 0) { f3y -= iy * gx[addry-i_1+j_1]; }
                        if (iz > 0) { f3z -= iz * gx[addrz-i_1+j_1]; }
                        f3x *= aj2;
                        f3y *= aj2;
                        f3z *= aj2;
                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+i_1-j_1];
                            if (ix > 0) { fx -= ix * gx[addrx-i_1-j_1]; }
                            f3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gx[addry+i_1-j_1];
                            if (iy > 0) { fy -= iy * gx[addry-i_1-j_1]; }
                            f3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gx[addrz+i_1-j_1];
                            if (iz > 0) { fz -= iz * gx[addrz-i_1-j_1]; }
                            f3z -= jz * fz;
                        }
                        v1xx += f3x * prod_yz;
                        v1yy += f3y * prod_xz;
                        v1zz += f3z * prod_xy;
                        v1xy += f2x * f1y * Izdd;
                        v1xz += f2x * f1z * Iydd;
                        v1yx += f2y * f1x * Izdd;
                        v1yz += f2y * f1z * Ixdd;
                        v1zx += f2z * f1x * Iydd;
                        v1zy += f2z * f1y * Ixdd;

                        double xjxi = rjri[0];
                        double yjyi = rjri[1];
                        double zjzi = rjri[2];
                        _gx_inc2 = gijx - gjx * xjxi;
                        _gy_inc2 = gijy - gjy * yjyi;
                        _gz_inc2 = gijz - gjz * zjzi;
                        f3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * Ix);
                        f3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * Iy);
                        f3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * Iz);
                        if (jx > 1) { f3x += jx*(jx-1) * gx[addrx-j_1*2]; }
                        if (jy > 1) { f3y += jy*(jy-1) * gx[addry-j_1*2]; }
                        if (jz > 1) { f3z += jz*(jz-1) * gx[addrz-j_1*2]; }
                        v_jxx += f3x * prod_yz;
                        v_jyy += f3y * prod_xz;
                        v_jzz += f3z * prod_xy;
                        v_jxy += f1x * f1y * Izdd;
                        v_jxz += f1x * f1z * Iydd;
                        v_jyz += f1y * f1z * Ixdd;

                        _gx_inc2 = gijx + gix * xjxi;
                        _gy_inc2 = gijy + giy * yjyi;
                        _gz_inc2 = gijz + giz * zjzi;
                        f3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * Ix);
                        f3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * Iy);
                        f3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * Iz);
                        if (ix > 1) { f3x += ix*(ix-1) * gx[addrx-i_1*2]; }
                        if (iy > 1) { f3y += iy*(iy-1) * gx[addry-i_1*2]; }
                        if (iz > 1) { f3z += iz*(iz-1) * gx[addrz-i_1*2]; }
                        v_ixx += f3x * prod_yz;
                        v_iyy += f3y * prod_xz;
                        v_izz += f3z * prod_xy;
                        v_ixy += f2x * f2y * Izdd;
                        v_ixz += f2x * f2z * Iydd;
                        v_iyz += f2y * f2z * Ixdd;

                        f1x = al2 * glx;
                        f1y = al2 * gly;
                        f1z = al2 * glz;
                        if (lx > 0) { f1x -= lx * gx[addrx-l_1]; }
                        if (ly > 0) { f1y -= ly * gx[addry-l_1]; }
                        if (lz > 0) { f1z -= lz * gx[addrz-l_1]; }

                        f2x = ak2 * gkx;
                        f2y = ak2 * gky;
                        f2z = ak2 * gkz;
                        if (kx > 0) { f2x -= kx * gx[addrx-k_1]; }
                        if (ky > 0) { f2y -= ky * gx[addry-k_1]; }
                        if (kz > 0) { f2z -= kz * gx[addrz-k_1]; }

                        double gklx = gx[addrx+k_1+l_1];
                        double gkly = gx[addry+k_1+l_1];
                        double gklz = gx[addrz+k_1+l_1];
                        f3x = ak2 * gklx;
                        f3y = ak2 * gkly;
                        f3z = ak2 * gklz;
                        if (kx > 0) { f3x -= kx * gx[addrx-k_1+l_1]; }
                        if (ky > 0) { f3y -= ky * gx[addry-k_1+l_1]; }
                        if (kz > 0) { f3z -= kz * gx[addrz-k_1+l_1]; }
                        f3x *= al2;
                        f3y *= al2;
                        f3z *= al2;
                        if (lx > 0) {
                            double fx = ak2 * gx[addrx+k_1-l_1];
                            if (kx > 0) { fx -= kx * gx[addrx-k_1-l_1]; }
                            f3x -= lx * fx;
                        }
                        if (ly > 0) {
                            double fy = ak2 * gx[addry+k_1-l_1];
                            if (ky > 0) { fy -= ky * gx[addry-k_1-l_1]; }
                            f3y -= ly * fy;
                        }
                        if (lz > 0) {
                            double fz = ak2 * gx[addrz+k_1-l_1];
                            if (kz > 0) { fz -= kz * gx[addrz-k_1-l_1]; }
                            f3z -= lz * fz;
                        }
                        v2xx += f3x * prod_yz;
                        v2yy += f3y * prod_xz;
                        v2zz += f3z * prod_xy;
                        v2xy += f2x * f1y * Izdd;
                        v2xz += f2x * f1z * Iydd;
                        v2yx += f2y * f1x * Izdd;
                        v2yz += f2y * f1z * Ixdd;
                        v2zx += f2z * f1x * Iydd;
                        v2zy += f2z * f1y * Ixdd;

                        double xlxk = rlrk[0*nsq_per_block];
                        double ylyk = rlrk[1*nsq_per_block];
                        double zlzk = rlrk[2*nsq_per_block];
                        _gx_inc2 = gklx - glx * xlxk;
                        _gy_inc2 = gkly - gly * ylyk;
                        _gz_inc2 = gklz - glz * zlzk;
                        f3x = al2 * (al2 * _gx_inc2 - (2*lx+1) * Ix);
                        f3y = al2 * (al2 * _gy_inc2 - (2*ly+1) * Iy);
                        f3z = al2 * (al2 * _gz_inc2 - (2*lz+1) * Iz);
                        if (lx > 1) { f3x += lx*(lx-1) * gx[addrx-l_1*2]; }
                        if (ly > 1) { f3y += ly*(ly-1) * gx[addry-l_1*2]; }
                        if (lz > 1) { f3z += lz*(lz-1) * gx[addrz-l_1*2]; }
                        v_lxx += f3x * prod_yz;
                        v_lyy += f3y * prod_xz;
                        v_lzz += f3z * prod_xy;
                        v_lxy += f1x * f1y * Izdd;
                        v_lxz += f1x * f1z * Iydd;
                        v_lyz += f1y * f1z * Ixdd;

                        _gx_inc2 = gklx + gkx * xlxk;
                        _gy_inc2 = gkly + gky * ylyk;
                        _gz_inc2 = gklz + gkz * zlzk;
                        f3x = ak2 * (ak2 * _gx_inc2 - (2*kx+1) * Ix);
                        f3y = ak2 * (ak2 * _gy_inc2 - (2*ky+1) * Iy);
                        f3z = ak2 * (ak2 * _gz_inc2 - (2*kz+1) * Iz);
                        if (kx > 1) { f3x += kx*(kx-1) * gx[addrx-k_1*2]; }
                        if (ky > 1) { f3y += ky*(ky-1) * gx[addry-k_1*2]; }
                        if (kz > 1) { f3z += kz*(kz-1) * gx[addrz-k_1*2]; }
                        v_kxx += f3x * prod_yz;
                        v_kyy += f3y * prod_xz;
                        v_kzz += f3z * prod_xy;
                        v_kxy += f2x * f2y * Izdd;
                        v_kxz += f2x * f2z * Iydd;
                        v_kyz += f2y * f2z * Ixdd;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
            int la = bas[lsh*BAS_SLOTS+ATOM_OF];
            int natm = envs.natm;
            double *ejk = jk.ejk;
            atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
            atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
            atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
            atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
            atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
            atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
            atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
            atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
            atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
            atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
            atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
            atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
            atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
            atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
            atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
            atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
            atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
            atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
            atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
            atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
            atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
            atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
            atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
            atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
        }
    }
}

__global__ static
void rys_ejk_ip2_type3_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                              int *pool, double *dd_pool)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
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
    int nroots = bounds.nroots;
    int lij = li + lj + 1;
    int lkl = lk + ll + 1;
    int i_1 =          nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+nroots*2+8);
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);
    int *idx_l = _c_cartesian_lexical_xyz + lex_xyz_offset(ll);
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[4];
    __shared__ double *expi;
    __shared__ double *expj;
    if (thread_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
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
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        int *ao_loc = envs.ao_loc;
        int nao = ao_loc[nbas];
        double *dm = jk.dm;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
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
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;
        int nfi = bounds.nfi;
        int nfj = bounds.nfj;
        int nfk = bounds.nfk;
        int nfl = bounds.nfl;
        int nfij = nfi * nfj;
        int nfkl = nfk * nfl;
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
                    dd += jk.j_factor * (dm[_ji] + dm[_ji]) * (dmb[_lk] + dmb[_lk]);
                }
                dd_cache[n*nsq_per_block] = fac_sym * dd;
            }
        }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
            __syncthreads();
            if (gout_id == 0) {
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
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
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
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
                        aij_cache[3] = aj * 2;
                    }
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
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
                        double *_gx = gx + (i + _ix * g_size) * nsq_per_block;
                        double Rqc = rlrk[_ix*nsq_per_block] * akl_cache[nsq_per_block];
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
                        double Ixdd = Ix * dd;
                        double Iydd = Iy * dd;
                        double Izdd = Iz * dd;
                        double prod_yz = Iy * Izdd;
                        double prod_xz = Ix * Izdd;
                        double prod_xy = Ix * Iydd;
                        double gix = gx[addrx+i_1];
                        double giy = gx[addry+i_1];
                        double giz = gx[addrz+i_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gx[addry+k_1];
                        double gkz = gx[addrz+k_1];

                        double ai2 = aij_cache[2];
                        double aj2 = aij_cache[3];
                        double fix = ai2 * gix;
                        double fiy = ai2 * giy;
                        double fiz = ai2 * giz;
                        if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                        if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                        if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }

                        double xjxi = rjri[0];
                        double yjyi = rjri[1];
                        double zjzi = rjri[2];
                        double xlxk = rlrk[0*nsq_per_block];
                        double ylyk = rlrk[1*nsq_per_block];
                        double zlzk = rlrk[2*nsq_per_block];
                        double fjx = aj2 * (gix - xjxi * Ix);
                        double fjy = aj2 * (giy - yjyi * Iy);
                        double fjz = aj2 * (giz - zjzi * Iz);
                        if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                        if (jy > 0) { fjy -= jy * gx[addry-j_1]; }
                        if (jz > 0) { fjz -= jz * gx[addrz-j_1]; }

                        double fkx = ak2 * gkx;
                        double fky = ak2 * gky;
                        double fkz = ak2 * gkz;
                        if (kx > 0) { fkx -= kx * gx[addrx-k_1]; }
                        if (ky > 0) { fky -= ky * gx[addry-k_1]; }
                        if (kz > 0) { fkz -= kz * gx[addrz-k_1]; }

                        double flx = al2 * (gkx - xlxk * Ix);
                        double fly = al2 * (gky - ylyk * Iy);
                        double flz = al2 * (gkz - zlzk * Iz);
                        if (lx > 0) { flx -= lx * gx[addrx-l_1]; }
                        if (ly > 0) { fly -= ly * gx[addry-l_1]; }
                        if (lz > 0) { flz -= lz * gx[addrz-l_1]; }

                        v_ixky += fix * fky * Izdd;
                        v_ixkz += fix * fkz * Iydd;
                        v_iykx += fiy * fkx * Izdd;
                        v_iykz += fiy * fkz * Ixdd;
                        v_izkx += fiz * fkx * Iydd;
                        v_izky += fiz * fky * Ixdd;
                        v_jxky += fjx * fky * Izdd;
                        v_jxkz += fjx * fkz * Iydd;
                        v_jykx += fjy * fkx * Izdd;
                        v_jykz += fjy * fkz * Ixdd;
                        v_jzkx += fjz * fkx * Iydd;
                        v_jzky += fjz * fky * Ixdd;
                        v_ixly += fix * fly * Izdd;
                        v_ixlz += fix * flz * Iydd;
                        v_iylx += fiy * flx * Izdd;
                        v_iylz += fiy * flz * Ixdd;
                        v_izlx += fiz * flx * Iydd;
                        v_izly += fiz * fly * Ixdd;
                        v_jxly += fjx * fly * Izdd;
                        v_jxlz += fjx * flz * Iydd;
                        v_jylx += fjy * flx * Izdd;
                        v_jylz += fjy * flz * Ixdd;
                        v_jzlx += fjz * flx * Iydd;
                        v_jzly += fjz * fly * Ixdd;

                        double gikx = gx[addrx+i_1+k_1];
                        double giky = gx[addry+i_1+k_1];
                        double gikz = gx[addrz+i_1+k_1];
                        double fikx = ai2 * gikx;
                        double fiky = ai2 * giky;
                        double fikz = ai2 * gikz;
                        if (ix > 0) { fikx -= ix * gx[addrx-i_1+k_1]; }
                        if (iy > 0) { fiky -= iy * gx[addry-i_1+k_1]; }
                        if (iz > 0) { fikz -= iz * gx[addrz-i_1+k_1]; }
                        fikx *= ak2;
                        fiky *= ak2;
                        fikz *= ak2;

                        double fjkx = aj2 * (gikx - xjxi * gkx);
                        double fjky = aj2 * (giky - yjyi * gky);
                        double fjkz = aj2 * (gikz - zjzi * gkz);
                        if (jx > 0) { fjkx -= jx * gx[addrx-j_1+k_1]; }
                        if (jy > 0) { fjky -= jy * gx[addry-j_1+k_1]; }
                        if (jz > 0) { fjkz -= jz * gx[addrz-j_1+k_1]; }
                        fjkx *= ak2;
                        fjky *= ak2;
                        fjkz *= ak2;

                        if (kx > 0) {
                            double gixk = gx[addrx+i_1-k_1];
                            double fx = ai2 * gixk;
                            if (ix > 0) { fx -= ix * gx[addrx-i_1-k_1]; }
                            fikx -= kx * fx;
                            fx = aj2 * (gixk - xjxi * gx[addrx-k_1]);
                            if (jx > 0) { fx -= jx * gx[addrx-j_1-k_1]; }
                            fjkx -= kx * fx;
                        }
                        if (ky > 0) {
                            double giyk = gx[addry+i_1-k_1];
                            double fy = ai2 * giyk;
                            if (iy > 0) { fy -= iy * gx[addry-i_1-k_1]; }
                            fiky -= ky * fy;
                            fy = aj2 * (giyk - yjyi * gx[addry-k_1]);
                            if (jy > 0) { fy -= jy * gx[addry-j_1-k_1]; }
                            fjky -= ky * fy;
                        }
                        if (kz > 0) {
                            double gizk = gx[addrz+i_1-k_1];
                            double fz = ai2 * gizk;
                            if (iz > 0) { fz -= iz * gx[addrz-i_1-k_1]; }
                            fikz -= kz * fz;
                            fz = aj2 * (gizk - zjzi * gx[addrz-k_1]);
                            if (jz > 0) { fz -= jz * gx[addrz-j_1-k_1]; }
                            fjkz -= kz * fz;
                        }

                        v_ixkx += fikx * prod_yz;
                        v_iyky += fiky * prod_xz;
                        v_izkz += fikz * prod_xy;
                        v_jxkx += fjkx * prod_yz;
                        v_jyky += fjky * prod_xz;
                        v_jzkz += fjkz * prod_xy;

                        double filx = ai2 * (gikx - xlxk * gix);
                        double fily = ai2 * (giky - ylyk * giy);
                        double filz = ai2 * (gikz - zlzk * giz);
                        if (ix > 0) { filx -= ix * (gx[addrx-i_1+k_1] - xlxk * gx[addrx-i_1]); }
                        if (iy > 0) { fily -= iy * (gx[addry-i_1+k_1] - ylyk * gx[addry-i_1]); }
                        if (iz > 0) { filz -= iz * (gx[addrz-i_1+k_1] - zlzk * gx[addrz-i_1]); }
                        filx *= al2;
                        fily *= al2;
                        filz *= al2;

                        double fjlx = aj2 * (gikx - xjxi * gkx - xlxk * (gix - xjxi * Ix));
                        double fjly = aj2 * (giky - yjyi * gky - ylyk * (giy - yjyi * Iy));
                        double fjlz = aj2 * (gikz - zjzi * gkz - zlzk * (giz - zjzi * Iz));
                        if (jx > 0) { fjlx -= jx * (gx[addrx-j_1+k_1] - xlxk * gx[addrx-j_1]); }
                        if (jy > 0) { fjly -= jy * (gx[addry-j_1+k_1] - ylyk * gx[addry-j_1]); }
                        if (jz > 0) { fjlz -= jz * (gx[addrz-j_1+k_1] - zlzk * gx[addrz-j_1]); }
                        fjlx *= al2;
                        fjly *= al2;
                        fjlz *= al2;

                        if (lx > 0) {
                            double gixl = gx[addrx+i_1-l_1];
                            double fx = ai2 * gixl;
                            if (ix > 0) { fx -= ix * gx[addrx-i_1-l_1]; }
                            filx -= lx * fx;
                            fx = aj2 * (gixl - xjxi * gx[addrx-l_1]);
                            if (jx > 0) { fx -= jx * gx[addrx-j_1-l_1]; }
                            fjlx -= lx * fx;
                        }
                        if (ly > 0) {
                            double giyl = gx[addry+i_1-l_1];
                            double fy = ai2 * giyl;
                            if (iy > 0) { fy -= iy * gx[addry-i_1-l_1]; }
                            fily -= ly * fy;
                            fy = aj2 * (giyl - yjyi * gx[addry-l_1]);
                            if (jy > 0) { fy -= jy * gx[addry-j_1-l_1]; }
                            fjly -= ly * fy;
                        }
                        if (lz > 0) {
                            double gizl = gx[addrz+i_1-l_1];
                            double fz = ai2 * gizl;
                            if (iz > 0) { fz -= iz * gx[addrz-i_1-l_1]; }
                            filz -= lz * fz;
                            fz = aj2 * (gizl - zjzi * gx[addrz-l_1]);
                            if (jz > 0) { fz -= jz * gx[addrz-j_1-l_1]; }
                            fjlz -= lz * fz;
                        }
                        v_ixlx += filx * prod_yz;
                        v_iyly += fily * prod_xz;
                        v_izlz += filz * prod_xy;
                        v_jxlx += fjlx * prod_yz;
                        v_jyly += fjly * prod_xz;
                        v_jzlz += fjlz * prod_xy;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            int bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
            int la = bas[lsh*BAS_SLOTS+ATOM_OF];
            int natm = envs.natm;
            double *ejk = jk.ejk;
            atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
            atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
            atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
            atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
            atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
            atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
            atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
            atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
            atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
            atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
            atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
            atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
            atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
            atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
            atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
            atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
            atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
            atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
            atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
            atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
            atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
            atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
            atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
            atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
            atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
            atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
            atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
            atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
            atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
            atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
            atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
            atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
            atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
            atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
            atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
            atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
        }
    }
}

extern int rys_ejk_ip2_type12_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                        int *pool, double *dd_pool);
extern int rys_ejk_ip2_type3_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                        int *pool, double *dd_pool);

extern "C" {
int RYS_per_atom_jk_ip2_type12(double *ejk, double j_factor, double k_factor,
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
    int nroots = (order + 2) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 2);
    uint8_t stride_l = stride_k * (lk + 2);
    int g_size = stride_l * (ll + 2);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff};

    if (n_dm > 1) { // UHF
        k_factor *= 2.;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 4.*j_factor, -k_factor, n_dm, omega};
    if (omega >= 0) {
        jk.lr_factor = 1;
        jk.sr_factor = 0;
    } else {
        jk.lr_factor = 0;
        jk.sr_factor = 1;
    }

    if (!rys_ejk_ip2_type12_unrolled(&envs, &jk, &bounds, pool, dd_pool)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + 8) * quartets_per_block + ij_prims;
        int lij = li + lj + 2;
        int lkl = lk + ll + 2;

        rys_ejk_ip2_type12_kernel<<<npairs_ij, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, pool, dd_pool, lij, lkl);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip2_type12, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_per_atom_jk_ip2_type3(double *ejk, double j_factor, double k_factor,
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
    int nroots = (order + 2) / 2 + 1;
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

    if (n_dm > 1) { // UHF
        k_factor *= 2.;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 4.*j_factor, -k_factor, n_dm, omega};
    if (omega >= 0) {
        jk.lr_factor = 1;
        jk.sr_factor = 0;
    } else {
        jk.lr_factor = 0;
        jk.sr_factor = 1;
    }

    if (!rys_ejk_ip2_type3_unrolled(&envs, &jk, &bounds, pool, dd_pool)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + 8) * quartets_per_block + ij_prims;
        buflen = MAX(buflen, 9*gout_stride*quartets_per_block);

        rys_ejk_ip2_type3_kernel<<<npairs_ij, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, pool, dd_pool);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip2_type3, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_build_ejk_ip2_init(int shm_size)
{
    cudaFuncSetAttribute(rys_ejk_ip2_type12_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_ejk_ip2_type3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
