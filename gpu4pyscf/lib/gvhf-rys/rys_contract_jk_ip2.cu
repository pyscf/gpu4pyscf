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
    int lij = li + lj + 2;
    int lkl = lk + ll + 2;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 2);
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
    int g_stride_k = stride_k*nsq_per_block;
    int g_stride_l = stride_l*nsq_per_block;
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
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3];

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
            double aj_aij = aj / aij;
            double xixj = ri[0] - rj[0];
            double yiyj = ri[1] - rj[1];
            double zizj = ri[2] - rj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
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
            double xkxl = rk[0] - rl[0];
            double ykyl = rk[1] - rl[1];
            double zkzl = rk[2] - rl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
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

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    __syncthreads();
                    if (task_id < ntasks) {
                        int lkl3 = (lkl+1)*3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xixj = ri[_ix] - rj[_ix];
                            double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                            for (int j = 0; j <= lj; ++j) {
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
                    __syncthreads();
                    if (task_id < ntasks) {
                        for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                            int i = n / 3;
                            int _ix = n % 3;
                            double xkxl = rk[_ix] - rl[_ix];
                            double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                            for (int l = 0; l <= ll; ++l) {
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

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                        int kl = n / nfij;
                        int ij = n % nfij;
                        if (kl >= nfkl) break;
                        int ijx = idx_ij[ij];
                        int ijy = idy_ij[ij];
                        int ijz = idz_ij[ij];
                        int klx = idx_kl[kl];
                        int kly = idy_kl[kl];
                        int klz = idz_kl[kl];
                        int ix = ijx % (li + 1);
                        int jx = ijx / (li + 1);
                        int iy = ijy % (li + 1);
                        int jy = ijy / (li + 1);
                        int iz = ijz % (li + 1);
                        int jz = ijz / (li + 1);
                        int kx = klx % (lk + 1);
                        int lx = klx / (lk + 1);
                        int ky = kly % (lk + 1);
                        int ly = kly / (lk + 1);
                        int kz = klz % (lk + 1);
                        int lz = klz / (lk + 1);
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;
                        double dd = 0.;
                        if (do_k) {
                            int _jl = _j*nao+_l;
                            int _jk = _j*nao+_k;
                            int _il = _i*nao+_l;
                            int _ik = _i*nao+_k;
                            dd  = dm[_jk] * dm[_il];
                            dd += dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                int nao2 = nao * nao;
                                dd += dm[nao2+_jk] * dm[nao2+_il];
                                dd += dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            int _ji = _j*nao+_i;
                            int _lk = _l*nao+_k;
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[_ji] * dm[_lk];
                            } else {
                                int nao2 = nao * nao;
                                dd += jk.j_factor * (dm[_ji] + dm[nao2+_ji]) * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double Ix = gx[addrx] * dd;
                        double Iy = gy[addry] * dd;
                        double Iz = gz[addrz] * dd;
                        double prod_yz = gy[addry] * Iz;
                        double prod_xz = gx[addrx] * Iz;
                        double prod_xy = gx[addrx] * Iy;

                        double g1x, g1y, g1z;
                        double g2x, g2y, g2z;
                        double g3x, g3y, g3z;
                        double _gx_inc2, _gy_inc2, _gz_inc2;
                        g1x = aj2 * gx[addrx+g_stride_j];
                        g1y = aj2 * gy[addry+g_stride_j];
                        g1z = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { g1x -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { g1y -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { g1z -= jz * gz[addrz-g_stride_j]; }

                        g2x = ai2 * gx[addrx+g_stride_i];
                        g2y = ai2 * gy[addry+g_stride_i];
                        g2z = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { g2x -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { g2y -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { g2z -= iz * gz[addrz-g_stride_i]; }

                        g3x = ai2 * gx[addrx+g_stride_i+g_stride_j];
                        g3y = ai2 * gy[addry+g_stride_i+g_stride_j];
                        g3z = ai2 * gz[addrz+g_stride_i+g_stride_j];
                        if (ix > 0) { g3x -= ix * gx[addrx-g_stride_i+g_stride_j]; }
                        if (iy > 0) { g3y -= iy * gy[addry-g_stride_i+g_stride_j]; }
                        if (iz > 0) { g3z -= iz * gz[addrz-g_stride_i+g_stride_j]; }
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_j];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_j]; }
                            g3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_j];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_j]; }
                            g3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_j];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_j]; }
                            g3z -= jz * fz;
                        }
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;

                        double xixj = ri[0] - rj[0];
                        double yiyj = ri[1] - rj[1];
                        double zizj = ri[2] - rj[2];
                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] + gx[addrx+g_stride_j] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] + gy[addry+g_stride_j] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] + gz[addrz+g_stride_j] * zizj;
                        g3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * gx[addrx]);
                        g3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * gy[addry]);
                        g3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * gz[addrz]);
                        if (jx > 1) { g3x += jx*(jx-1) * gx[addrx-g_stride_j*2]; }
                        if (jy > 1) { g3y += jy*(jy-1) * gy[addry-g_stride_j*2]; }
                        if (jz > 1) { g3z += jz*(jz-1) * gz[addrz-g_stride_j*2]; }
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;

                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] - gx[addrx+g_stride_i] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] - gy[addry+g_stride_i] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] - gz[addrz+g_stride_i] * zizj;
                        g3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * gx[addrx]);
                        g3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * gy[addry]);
                        g3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * gz[addrz]);
                        if (ix > 1) { g3x += ix*(ix-1) * gx[addrx-g_stride_i*2]; }
                        if (iy > 1) { g3y += iy*(iy-1) * gy[addry-g_stride_i*2]; }
                        if (iz > 1) { g3z += iz*(iz-1) * gz[addrz-g_stride_i*2]; }
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;

                        g1x = al2 * gx[addrx+g_stride_l];
                        g1y = al2 * gy[addry+g_stride_l];
                        g1z = al2 * gz[addrz+g_stride_l];
                        if (lx > 0) { g1x -= lx * gx[addrx-g_stride_l]; }
                        if (ly > 0) { g1y -= ly * gy[addry-g_stride_l]; }
                        if (lz > 0) { g1z -= lz * gz[addrz-g_stride_l]; }

                        g2x = ak2 * gx[addrx+g_stride_k];
                        g2y = ak2 * gy[addry+g_stride_k];
                        g2z = ak2 * gz[addrz+g_stride_k];
                        if (kx > 0) { g2x -= kx * gx[addrx-g_stride_k]; }
                        if (ky > 0) { g2y -= ky * gy[addry-g_stride_k]; }
                        if (kz > 0) { g2z -= kz * gz[addrz-g_stride_k]; }

                        g3x = ak2 * gx[addrx+g_stride_k+g_stride_l];
                        g3y = ak2 * gy[addry+g_stride_k+g_stride_l];
                        g3z = ak2 * gz[addrz+g_stride_k+g_stride_l];
                        if (kx > 0) { g3x -= kx * gx[addrx-g_stride_k+g_stride_l]; }
                        if (ky > 0) { g3y -= ky * gy[addry-g_stride_k+g_stride_l]; }
                        if (kz > 0) { g3z -= kz * gz[addrz-g_stride_k+g_stride_l]; }
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        if (lx > 0) {
                            double fx = ak2 * gx[addrx+g_stride_k-g_stride_l];
                            if (kx > 0) { fx -= kx * gx[addrx-g_stride_k-g_stride_l]; }
                            g3x -= lx * fx;
                        }
                        if (ly > 0) {
                            double fy = ak2 * gy[addry+g_stride_k-g_stride_l];
                            if (ky > 0) { fy -= ky * gy[addry-g_stride_k-g_stride_l]; }
                            g3y -= ly * fy;
                        }
                        if (lz > 0) {
                            double fz = ak2 * gz[addrz+g_stride_k-g_stride_l];
                            if (kz > 0) { fz -= kz * gz[addrz-g_stride_k-g_stride_l]; }
                            g3z -= lz * fz;
                        }
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;

                        double xkxl = rk[0] - rl[0];
                        double ykyl = rk[1] - rl[1];
                        double zkzl = rk[2] - rl[2];
                        _gx_inc2 = gx[addrx+g_stride_k+g_stride_l] + gx[addrx+g_stride_l] * xkxl;
                        _gy_inc2 = gy[addry+g_stride_k+g_stride_l] + gy[addry+g_stride_l] * ykyl;
                        _gz_inc2 = gz[addrz+g_stride_k+g_stride_l] + gz[addrz+g_stride_l] * zkzl;
                        g3x = al2 * (al2 * _gx_inc2 - (2*lx+1) * gx[addrx]);
                        g3y = al2 * (al2 * _gy_inc2 - (2*ly+1) * gy[addry]);
                        g3z = al2 * (al2 * _gz_inc2 - (2*lz+1) * gz[addrz]);
                        if (lx > 1) { g3x += lx*(lx-1) * gx[addrx-g_stride_l*2]; }
                        if (ly > 1) { g3y += ly*(ly-1) * gy[addry-g_stride_l*2]; }
                        if (lz > 1) { g3z += lz*(lz-1) * gz[addrz-g_stride_l*2]; }
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;

                        _gx_inc2 = gx[addrx+g_stride_k+g_stride_l] - gx[addrx+g_stride_k] * xkxl;
                        _gy_inc2 = gy[addry+g_stride_k+g_stride_l] - gy[addry+g_stride_k] * ykyl;
                        _gz_inc2 = gz[addrz+g_stride_k+g_stride_l] - gz[addrz+g_stride_k] * zkzl;
                        g3x = ak2 * (ak2 * _gx_inc2 - (2*kx+1) * gx[addrx]);
                        g3y = ak2 * (ak2 * _gy_inc2 - (2*ky+1) * gy[addry]);
                        g3z = ak2 * (ak2 * _gz_inc2 - (2*kz+1) * gz[addrz]);
                        if (kx > 1) { g3x += kx*(kx-1) * gx[addrx-g_stride_k*2]; }
                        if (ky > 1) { g3y += ky*(ky-1) * gy[addry-g_stride_k*2]; }
                        if (kz > 1) { g3z += kz*(kz-1) * gz[addrz-g_stride_k*2]; }
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
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
    int lij = li + lj + 2;
    int lkl = lk + ll + 2;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
    int g_stride_k = stride_k*nsq_per_block;
    int g_stride_l = stride_l*nsq_per_block;
    int g_size = stride_l * (ll + 2);
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
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3];

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
            double aj_aij = aj / aij;
            double xixj = ri[0] - rj[0];
            double yiyj = ri[1] - rj[1];
            double zizj = ri[2] - rj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
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
            double xkxl = rk[0] - rl[0];
            double ykyl = rk[1] - rl[1];
            double zkzl = rk[2] - rl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
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

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
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

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    __syncthreads();
                    if (task_id < ntasks) {
                        int lkl3 = (lkl+1)*3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xixj = ri[_ix] - rj[_ix];
                            double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                            for (int j = 0; j <= lj; ++j) {
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
                    __syncthreads();
                    if (task_id < ntasks) {
                        for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                            int i = n / 3;
                            int _ix = n % 3;
                            double xkxl = rk[_ix] - rl[_ix];
                            double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                            for (int l = 0; l <= ll; ++l) {
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

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                        int kl = n / nfij;
                        int ij = n % nfij;
                        if (kl >= nfkl) break;
                        int ijx = idx_ij[ij];
                        int ijy = idy_ij[ij];
                        int ijz = idz_ij[ij];
                        int klx = idx_kl[kl];
                        int kly = idy_kl[kl];
                        int klz = idz_kl[kl];
                        int ix = ijx % (li + 1);
                        int jx = ijx / (li + 1);
                        int iy = ijy % (li + 1);
                        int jy = ijy / (li + 1);
                        int iz = ijz % (li + 1);
                        int jz = ijz / (li + 1);
                        int kx = klx % (lk + 1);
                        int lx = klx / (lk + 1);
                        int ky = kly % (lk + 1);
                        int ly = kly / (lk + 1);
                        int kz = klz % (lk + 1);
                        int lz = klz / (lk + 1);

                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;
                        double dd = 0.;
                        if (do_k) {
                            int _jl = _j*nao+_l;
                            int _jk = _j*nao+_k;
                            int _il = _i*nao+_l;
                            int _ik = _i*nao+_k;
                            dd  = dm[_jk] * dm[_il];
                            dd += dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                int nao2 = nao * nao;
                                dd += dm[nao2+_jk] * dm[nao2+_il];
                                dd += dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            int _ji = _j*nao+_i;
                            int _lk = _l*nao+_k;
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[_ji] * dm[_lk];
                            } else {
                                int nao2 = nao * nao;
                                dd += jk.j_factor * (dm[_ji] + dm[nao2+_ji]) * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }

                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double Ix = gx[addrx] * dd;
                        double Iy = gy[addry] * dd;
                        double Iz = gz[addrz] * dd;
                        double prod_yz = gy[addry] * Iz;
                        double prod_xz = gx[addrx] * Iz;
                        double prod_xy = gx[addrx] * Iy;

                        double gix, giy, giz;
                        double gjx, gjy, gjz;
                        double gkx, gky, gkz;
                        double glx, gly, glz;
                        double gikx, giky, gikz;
                        double gjkx, gjky, gjkz;
                        double gilx, gily, gilz;
                        double gjlx, gjly, gjlz;
                        gikx = ai2 * gx[addrx+g_stride_i+g_stride_k];
                        giky = ai2 * gy[addry+g_stride_i+g_stride_k];
                        gikz = ai2 * gz[addrz+g_stride_i+g_stride_k];
                        if (ix > 0) { gikx -= ix * gx[addrx-g_stride_i+g_stride_k]; }
                        if (iy > 0) { giky -= iy * gy[addry-g_stride_i+g_stride_k]; }
                        if (iz > 0) { gikz -= iz * gz[addrz-g_stride_i+g_stride_k]; }
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;

                        gjkx = aj2 * gx[addrx+g_stride_j+g_stride_k];
                        gjky = aj2 * gy[addry+g_stride_j+g_stride_k];
                        gjkz = aj2 * gz[addrz+g_stride_j+g_stride_k];
                        if (jx > 0) { gjkx -= jx * gx[addrx-g_stride_j+g_stride_k]; }
                        if (jy > 0) { gjky -= jy * gy[addry-g_stride_j+g_stride_k]; }
                        if (jz > 0) { gjkz -= jz * gz[addrz-g_stride_j+g_stride_k]; }
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;

                        if (kx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_k];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_k]; }
                            gikx -= kx * fx;
                            fx = aj2 * gx[addrx+g_stride_j-g_stride_k];
                            if (jx > 0) { fx -= jx * gx[addrx-g_stride_j-g_stride_k]; }
                            gjkx -= kx * fx;
                        }
                        if (ky > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_k];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_k]; }
                            giky -= ky * fy;
                            fy = aj2 * gy[addry+g_stride_j-g_stride_k];
                            if (jy > 0) { fy -= jy * gy[addry-g_stride_j-g_stride_k]; }
                            gjky -= ky * fy;
                        }
                        if (kz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_k];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_k]; }
                            gikz -= kz * fz;
                            fz = aj2 * gz[addrz+g_stride_j-g_stride_k];
                            if (jz > 0) { fz -= jz * gz[addrz-g_stride_j-g_stride_k]; }
                            gjkz -= kz * fz;
                        }

                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;

                        gilx = ai2 * gx[addrx+g_stride_i+g_stride_l];
                        gily = ai2 * gy[addry+g_stride_i+g_stride_l];
                        gilz = ai2 * gz[addrz+g_stride_i+g_stride_l];
                        if (ix > 0) { gilx -= ix * gx[addrx-g_stride_i+g_stride_l]; }
                        if (iy > 0) { gily -= iy * gy[addry-g_stride_i+g_stride_l]; }
                        if (iz > 0) { gilz -= iz * gz[addrz-g_stride_i+g_stride_l]; }
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;

                        gjlx = aj2 * gx[addrx+g_stride_j+g_stride_l];
                        gjly = aj2 * gy[addry+g_stride_j+g_stride_l];
                        gjlz = aj2 * gz[addrz+g_stride_j+g_stride_l];
                        if (jx > 0) { gjlx -= jx * gx[addrx-g_stride_j+g_stride_l]; }
                        if (jy > 0) { gjly -= jy * gy[addry-g_stride_j+g_stride_l]; }
                        if (jz > 0) { gjlz -= jz * gz[addrz-g_stride_j+g_stride_l]; }
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;

                        if (lx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_l];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_l]; }
                            gilx -= lx * fx;
                            fx = aj2 * gx[addrx+g_stride_j-g_stride_l];
                            if (jx > 0) { fx -= jx * gx[addrx-g_stride_j-g_stride_l]; }
                            gjlx -= lx * fx;
                        }
                        if (ly > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_l];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_l]; }
                            gily -= ly * fy;
                            fy = aj2 * gy[addry+g_stride_j-g_stride_l];
                            if (jy > 0) { fy -= jy * gy[addry-g_stride_j-g_stride_l]; }
                            gjly -= ly * fy;
                        }
                        if (lz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_l];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_l]; }
                            gilz -= lz * fz;
                            fz = aj2 * gz[addrz+g_stride_j-g_stride_l];
                            if (jz > 0) { fz -= jz * gz[addrz-g_stride_j-g_stride_l]; }
                            gjlz -= lz * fz;
                        }
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;

                        gix = ai2 * gx[addrx+g_stride_i];
                        giy = ai2 * gy[addry+g_stride_i];
                        giz = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { gix -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { giy -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { giz -= iz * gz[addrz-g_stride_i]; }

                        gjx = aj2 * gx[addrx+g_stride_j];
                        gjy = aj2 * gy[addry+g_stride_j];
                        gjz = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { gjx -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { gjy -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { gjz -= jz * gz[addrz-g_stride_j]; }

                        gkx = ak2 * gx[addrx+g_stride_k];
                        gky = ak2 * gy[addry+g_stride_k];
                        gkz = ak2 * gz[addrz+g_stride_k];
                        if (kx > 0) { gkx -= kx * gx[addrx-g_stride_k]; }
                        if (ky > 0) { gky -= ky * gy[addry-g_stride_k]; }
                        if (kz > 0) { gkz -= kz * gz[addrz-g_stride_k]; }

                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;

                        glx = al2 * gx[addrx+g_stride_l];
                        gly = al2 * gy[addry+g_stride_l];
                        glz = al2 * gz[addrz+g_stride_l];
                        if (lx > 0) { glx -= lx * gx[addrx-g_stride_l]; }
                        if (ly > 0) { gly -= ly * gy[addry-g_stride_l]; }
                        if (lz > 0) { glz -= lz * gz[addrz-g_stride_l]; }

                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
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
            rys_ejk_ip2_type12_general(envs, jk, bounds, shl_quartet_idx, ntasks);
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
            rys_ejk_ip2_type3_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
