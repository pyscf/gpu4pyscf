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
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"
#include "create_tasks_ip2.cu"

// type 1: (d^2i j | k l)
// type 2: (di dj | k l)
// type 3: (di j | dk l)

__device__
static void rys_ejk_ip2_type12_general(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                              ShellQuartet *shl_quartet_idx, double *dd_cache, int ntasks)
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
    int lij = li + lj + 2;
    int lkl = lk + ll + 2;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 2);
    int i_1 =          nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;
    int nfj = nfij/nfi;
    int nfl = nfkl/nfk;
    int *idx_i = c_g_pair_idx + c_g_pair_offsets[li*LMAX1];
    int *idy_i = idx_i + nfi;
    int *idz_i = idy_i + nfi;
    int *idx_j = c_g_pair_idx + c_g_pair_offsets[lj*LMAX1];
    int *idy_j = idx_j + nfj;
    int *idz_j = idy_j + nfj;
    int *idx_k = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1];
    int *idy_k = idx_k + nfk;
    int *idz_k = idy_k + nfk;
    int *idx_l = c_g_pair_idx + c_g_pair_offsets[ll*LMAX1];
    int *idy_l = idx_l + nfl;
    int *idz_l = idy_l + nfl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    double *dm = jk.dm;
    dd_cache += sq_id;
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rjri = gz + nsq_per_block * g_size;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *cicj_cache = Rpq + nsq_per_block * 3;

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
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double dist_ij = xjxi*xjxi+yjyi*yjyi+zjzi*zjzi;
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
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * dist_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }

        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;
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
                dd_cache[n*nsq_per_block] = dd;
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
                dd_cache[n*nsq_per_block] = dd;
            }
        }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
            double akl = ak + al;
            double al_akl = al / akl;
            __syncthreads();
            if (gout_id == 0) {
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0*nsq_per_block] * aj_aij;
                double yij = ri[1] + rjri[1*nsq_per_block] * aj_aij;
                double zij = ri[2] + rjri[2*nsq_per_block] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                __syncthreads();
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp*nsq_per_block];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    __syncthreads();
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = g + n * g_size * nsq_per_block;
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
                        double *_gx = g + (i + _ix * g_size) * nsq_per_block;
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
                    __syncthreads();
                    if (task_id < ntasks) {
                        int lkl3 = (lkl+1)*3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xjxi = rjri[_ix*nsq_per_block];
                            double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
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
                            double *_gx = g + (_ix*g_size + i) * nsq_per_block;
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
                        int ix = idx_i[i];
                        int iy = idy_i[i];
                        int iz = idz_i[i];
                        int jx = idx_j[j];
                        int jy = idy_j[j];
                        int jz = idz_j[j];
                        int kx = idx_k[k];
                        int ky = idy_k[k];
                        int kz = idz_k[k];
                        int lx = idx_l[l];
                        int ly = idy_l[l];
                        int lz = idz_l[l];
                        double dd = dd_cache[n*nsq_per_block];
                        int addrx = (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double Ix = gx[addrx];
                        double Iy = gy[addry];
                        double Iz = gz[addrz];
                        double Ixdd = Ix * dd;
                        double Iydd = Iy * dd;
                        double Izdd = Iz * dd;
                        double prod_yz = Iy * Izdd;
                        double prod_xz = Ix * Izdd;
                        double prod_xy = Ix * Iydd;
                        double gix = gx[addrx+i_1];
                        double giy = gy[addry+i_1];
                        double giz = gz[addrz+i_1];
                        double gjx = gx[addrx+j_1];
                        double gjy = gy[addry+j_1];
                        double gjz = gz[addrz+j_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gy[addry+k_1];
                        double gkz = gz[addrz+k_1];
                        double glx = gx[addrx+l_1];
                        double gly = gy[addry+l_1];
                        double glz = gz[addrz+l_1];

                        double f1x, f1y, f1z;
                        double f2x, f2y, f2z;
                        double f3x, f3y, f3z;
                        double _gx_inc2, _gy_inc2, _gz_inc2;
                        f1x = aj2 * gjx;
                        f1y = aj2 * gjy;
                        f1z = aj2 * gjz;
                        if (jx > 0) { f1x -= jx * gx[addrx-j_1]; }
                        if (jy > 0) { f1y -= jy * gy[addry-j_1]; }
                        if (jz > 0) { f1z -= jz * gz[addrz-j_1]; }

                        f2x = ai2 * gix;
                        f2y = ai2 * giy;
                        f2z = ai2 * giz;
                        if (ix > 0) { f2x -= ix * gx[addrx-i_1]; }
                        if (iy > 0) { f2y -= iy * gy[addry-i_1]; }
                        if (iz > 0) { f2z -= iz * gz[addrz-i_1]; }

                        double gijx = gx[addrx+i_1+j_1];
                        double gijy = gy[addry+i_1+j_1];
                        double gijz = gz[addrz+i_1+j_1];
                        f3x = ai2 * gijx;
                        f3y = ai2 * gijy;
                        f3z = ai2 * gijz;
                        if (ix > 0) { f3x -= ix * gx[addrx-i_1+j_1]; }
                        if (iy > 0) { f3y -= iy * gy[addry-i_1+j_1]; }
                        if (iz > 0) { f3z -= iz * gz[addrz-i_1+j_1]; }
                        f3x *= aj2;
                        f3y *= aj2;
                        f3z *= aj2;
                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+i_1-j_1];
                            if (ix > 0) { fx -= ix * gx[addrx-i_1-j_1]; }
                            f3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gy[addry+i_1-j_1];
                            if (iy > 0) { fy -= iy * gy[addry-i_1-j_1]; }
                            f3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gz[addrz+i_1-j_1];
                            if (iz > 0) { fz -= iz * gz[addrz-i_1-j_1]; }
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

                        double xjxi = rjri[0*nsq_per_block];
                        double yjyi = rjri[1*nsq_per_block];
                        double zjzi = rjri[2*nsq_per_block];
                        _gx_inc2 = gijx - gjx * xjxi;
                        _gy_inc2 = gijy - gjy * yjyi;
                        _gz_inc2 = gijz - gjz * zjzi;
                        f3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * Ix);
                        f3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * Iy);
                        f3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * Iz);
                        if (jx > 1) { f3x += jx*(jx-1) * gx[addrx-j_1*2]; }
                        if (jy > 1) { f3y += jy*(jy-1) * gy[addry-j_1*2]; }
                        if (jz > 1) { f3z += jz*(jz-1) * gz[addrz-j_1*2]; }
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
                        if (iy > 1) { f3y += iy*(iy-1) * gy[addry-i_1*2]; }
                        if (iz > 1) { f3z += iz*(iz-1) * gz[addrz-i_1*2]; }
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
                        if (ly > 0) { f1y -= ly * gy[addry-l_1]; }
                        if (lz > 0) { f1z -= lz * gz[addrz-l_1]; }

                        f2x = ak2 * gkx;
                        f2y = ak2 * gky;
                        f2z = ak2 * gkz;
                        if (kx > 0) { f2x -= kx * gx[addrx-k_1]; }
                        if (ky > 0) { f2y -= ky * gy[addry-k_1]; }
                        if (kz > 0) { f2z -= kz * gz[addrz-k_1]; }

                        double gklx = gx[addrx+k_1+l_1];
                        double gkly = gy[addry+k_1+l_1];
                        double gklz = gz[addrz+k_1+l_1];
                        f3x = ak2 * gklx;
                        f3y = ak2 * gkly;
                        f3z = ak2 * gklz;
                        if (kx > 0) { f3x -= kx * gx[addrx-k_1+l_1]; }
                        if (ky > 0) { f3y -= ky * gy[addry-k_1+l_1]; }
                        if (kz > 0) { f3z -= kz * gz[addrz-k_1+l_1]; }
                        f3x *= al2;
                        f3y *= al2;
                        f3z *= al2;
                        if (lx > 0) {
                            double fx = ak2 * gx[addrx+k_1-l_1];
                            if (kx > 0) { fx -= kx * gx[addrx-k_1-l_1]; }
                            f3x -= lx * fx;
                        }
                        if (ly > 0) {
                            double fy = ak2 * gy[addry+k_1-l_1];
                            if (ky > 0) { fy -= ky * gy[addry-k_1-l_1]; }
                            f3y -= ly * fy;
                        }
                        if (lz > 0) {
                            double fz = ak2 * gz[addrz+k_1-l_1];
                            if (kz > 0) { fz -= kz * gz[addrz-k_1-l_1]; }
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
                        if (ly > 1) { f3y += ly*(ly-1) * gy[addry-l_1*2]; }
                        if (lz > 1) { f3z += lz*(lz-1) * gz[addrz-l_1*2]; }
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
                        if (ky > 1) { f3y += ky*(ky-1) * gy[addry-k_1*2]; }
                        if (kz > 1) { f3z += kz*(kz-1) * gz[addrz-k_1*2]; }
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
        if (task_id >= ntasks) {
            continue;
        }
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

__device__
static void rys_ejk_ip2_type3_general(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                             ShellQuartet *shl_quartet_idx, double *dd_cache, int ntasks)
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
    int lij = li + lj + 1;
    int lkl = lk + ll + 1;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int i_1 =          nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;
    int nfj = nfij/nfi;
    int nfl = nfkl/nfk;
    int *idx_i = c_g_pair_idx + c_g_pair_offsets[li*LMAX1];
    int *idy_i = idx_i + nfi;
    int *idz_i = idy_i + nfi;
    int *idx_j = c_g_pair_idx + c_g_pair_offsets[lj*LMAX1];
    int *idy_j = idx_j + nfj;
    int *idz_j = idy_j + nfj;
    int *idx_k = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1];
    int *idy_k = idx_k + nfk;
    int *idz_k = idy_k + nfk;
    int *idx_l = c_g_pair_idx + c_g_pair_offsets[ll*LMAX1];
    int *idy_l = idx_l + nfl;
    int *idz_l = idy_l + nfl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    double *dm = jk.dm;
    dd_cache += sq_id;
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rjri = gz + nsq_per_block * g_size;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *cicj_cache = Rpq + nsq_per_block * 3;

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
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double dist_ij = xjxi*xjxi+yjyi*yjyi+zjzi*zjzi;
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
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * dist_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }

        int do_j = jk.j_factor != 0.;
        int do_k = jk.k_factor != 0.;
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
                dd_cache[n*nsq_per_block] = dd;
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
                dd_cache[n*nsq_per_block] = dd;
            }
        }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double ak2 = ak * 2;
            double al2 = al * 2;
            double al_akl = al / akl;
            __syncthreads();
            if (gout_id == 0) {
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0*nsq_per_block] * aj_aij;
                double yij = ri[1] + rjri[1*nsq_per_block] * aj_aij;
                double zij = ri[2] + rjri[2*nsq_per_block] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                __syncthreads();
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp*nsq_per_block];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    __syncthreads();
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = g + n * g_size * nsq_per_block;
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
                        double *_gx = g + (i + _ix * g_size) * nsq_per_block;
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
                                double xjxi = rjri[_ix*nsq_per_block];
                                double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
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
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
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
                        int ix = idx_i[i];
                        int iy = idy_i[i];
                        int iz = idz_i[i];
                        int jx = idx_j[j];
                        int jy = idy_j[j];
                        int jz = idz_j[j];
                        int kx = idx_k[k];
                        int ky = idy_k[k];
                        int kz = idz_k[k];
                        int lx = idx_l[l];
                        int ly = idy_l[l];
                        int lz = idz_l[l];
                        double dd = dd_cache[n*nsq_per_block];
                        int addrx = (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double Ix = gx[addrx];
                        double Iy = gy[addry];
                        double Iz = gz[addrz];
                        double Ixdd = Ix * dd;
                        double Iydd = Iy * dd;
                        double Izdd = Iz * dd;
                        double prod_yz = Iy * Izdd;
                        double prod_xz = Ix * Izdd;
                        double prod_xy = Ix * Iydd;
                        double gix = gx[addrx+i_1];
                        double giy = gy[addry+i_1];
                        double giz = gz[addrz+i_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gy[addry+k_1];
                        double gkz = gz[addrz+k_1];

                        double fix = ai2 * gix;
                        double fiy = ai2 * giy;
                        double fiz = ai2 * giz;
                        if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                        if (iy > 0) { fiy -= iy * gy[addry-i_1]; }
                        if (iz > 0) { fiz -= iz * gz[addrz-i_1]; }

                        double xjxi = rjri[0*nsq_per_block];
                        double yjyi = rjri[1*nsq_per_block];
                        double zjzi = rjri[2*nsq_per_block];
                        double xlxk = rlrk[0*nsq_per_block];
                        double ylyk = rlrk[1*nsq_per_block];
                        double zlzk = rlrk[2*nsq_per_block];
                        double fjx = aj2 * (gix - xjxi * Ix);
                        double fjy = aj2 * (giy - yjyi * Iy);
                        double fjz = aj2 * (giz - zjzi * Iz);
                        if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                        if (jy > 0) { fjy -= jy * gy[addry-j_1]; }
                        if (jz > 0) { fjz -= jz * gz[addrz-j_1]; }

                        double fkx = ak2 * gkx;
                        double fky = ak2 * gky;
                        double fkz = ak2 * gkz;
                        if (kx > 0) { fkx -= kx * gx[addrx-k_1]; }
                        if (ky > 0) { fky -= ky * gy[addry-k_1]; }
                        if (kz > 0) { fkz -= kz * gz[addrz-k_1]; }

                        double flx = al2 * (gkx - xlxk * Ix);
                        double fly = al2 * (gky - ylyk * Iy);
                        double flz = al2 * (gkz - zlzk * Iz);
                        if (lx > 0) { flx -= lx * gx[addrx-l_1]; }
                        if (ly > 0) { fly -= ly * gy[addry-l_1]; }
                        if (lz > 0) { flz -= lz * gz[addrz-l_1]; }

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
                        double giky = gy[addry+i_1+k_1];
                        double gikz = gz[addrz+i_1+k_1];
                        double fikx = ai2 * gikx;
                        double fiky = ai2 * giky;
                        double fikz = ai2 * gikz;
                        if (ix > 0) { fikx -= ix * gx[addrx-i_1+k_1]; }
                        if (iy > 0) { fiky -= iy * gy[addry-i_1+k_1]; }
                        if (iz > 0) { fikz -= iz * gz[addrz-i_1+k_1]; }
                        fikx *= ak2;
                        fiky *= ak2;
                        fikz *= ak2;

                        double fjkx = aj2 * (gikx - xjxi * gkx);
                        double fjky = aj2 * (giky - yjyi * gky);
                        double fjkz = aj2 * (gikz - zjzi * gkz);
                        if (jx > 0) { fjkx -= jx * gx[addrx-j_1+k_1]; }
                        if (jy > 0) { fjky -= jy * gy[addry-j_1+k_1]; }
                        if (jz > 0) { fjkz -= jz * gz[addrz-j_1+k_1]; }
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
                            double giyk = gy[addry+i_1-k_1];
                            double fy = ai2 * giyk;
                            if (iy > 0) { fy -= iy * gy[addry-i_1-k_1]; }
                            fiky -= ky * fy;
                            fy = aj2 * (giyk - yjyi * gy[addry-k_1]);
                            if (jy > 0) { fy -= jy * gy[addry-j_1-k_1]; }
                            fjky -= ky * fy;
                        }
                        if (kz > 0) {
                            double gizk = gz[addrz+i_1-k_1];
                            double fz = ai2 * gizk;
                            if (iz > 0) { fz -= iz * gz[addrz-i_1-k_1]; }
                            fikz -= kz * fz;
                            fz = aj2 * (gizk - zjzi * gz[addrz-k_1]);
                            if (jz > 0) { fz -= jz * gz[addrz-j_1-k_1]; }
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
                        if (iy > 0) { fily -= iy * (gy[addry-i_1+k_1] - ylyk * gy[addry-i_1]); }
                        if (iz > 0) { filz -= iz * (gz[addrz-i_1+k_1] - zlzk * gz[addrz-i_1]); }
                        filx *= al2;
                        fily *= al2;
                        filz *= al2;

                        double fjlx = aj2 * (gikx - xjxi * gkx - xlxk * (gix - xjxi * Ix));
                        double fjly = aj2 * (giky - yjyi * gky - ylyk * (giy - yjyi * Iy));
                        double fjlz = aj2 * (gikz - zjzi * gkz - zlzk * (giz - zjzi * Iz));
                        if (jx > 0) { fjlx -= jx * (gx[addrx-j_1+k_1] - xlxk * gx[addrx-j_1]); }
                        if (jy > 0) { fjly -= jy * (gy[addry-j_1+k_1] - ylyk * gy[addry-j_1]); }
                        if (jz > 0) { fjlz -= jz * (gz[addrz-j_1+k_1] - zlzk * gz[addrz-j_1]); }
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
                            double giyl = gy[addry+i_1-l_1];
                            double fy = ai2 * giyl;
                            if (iy > 0) { fy -= iy * gy[addry-i_1-l_1]; }
                            fily -= ly * fy;
                            fy = aj2 * (giyl - yjyi * gy[addry-l_1]);
                            if (jy > 0) { fy -= jy * gy[addry-j_1-l_1]; }
                            fjly -= ly * fy;
                        }
                        if (lz > 0) {
                            double gizl = gz[addrz+i_1-l_1];
                            double fz = ai2 * gizl;
                            if (iz > 0) { fz -= iz * gz[addrz-i_1-l_1]; }
                            filz -= lz * fz;
                            fz = aj2 * (gizl - zjzi * gz[addrz-l_1]);
                            if (jz > 0) { fz -= jz * gz[addrz-j_1-l_1]; }
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
        if (task_id >= ntasks) {
            continue;
        }
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

__global__
void rys_ejk_ip2_type12_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                       ShellQuartet *pool, double *dd_pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;

    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int nf = nfij * nfkl;
    double *dd_cache = dd_pool + b_id * nf * blockDim.x;

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
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            rys_ejk_ip2_type12_general(envs, jk, bounds, shl_quartet_idx,
                                       dd_cache, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_ejk_ip2_type3_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                       ShellQuartet *pool, double *dd_pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;

    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int nf = nfij * nfkl;
    double *dd_cache = dd_pool + b_id * nf * blockDim.x;

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
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            rys_ejk_ip2_type3_general(envs, jk, bounds, shl_quartet_idx,
                                      dd_cache, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
