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

#define GWIDTH_IP1 18

__device__
static void rys_jk_ip1_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                               ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int t_id = sq_id + gout_id * nsq_per_block;
    int threads = nsq_per_block * gout_stride;
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
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rjri_cache[];
    double *rw = rjri_cache + iprim*jprim*6 + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rlrk = gz + nsq_per_block * g_size;
    double *Rpq = rlrk + nsq_per_block * 3;
    double goutx[GWIDTH_IP1];
    double gouty[GWIDTH_IP1];
    double goutz[GWIDTH_IP1];

    __syncthreads();
    ShellQuartet sq = shl_quartet_idx[0];
    int ish = sq.i;
    int jsh = sq.j;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double *rjri = rjri_cache + ij*6;
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        rjri[3] = ci[ip] * cj[jp] * Kab;
        rjri[4] = aij;
        rjri[5] = ai * 2;
    }

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
        if (ksh == lsh) fac_sym *= .5;
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
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GWIDTH_IP1) {
#pragma unroll
        for (int n = 0; n < GWIDTH_IP1; ++n) { goutx[n] = 0; gouty[n] = 0; goutz[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
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
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                double *rjri = rjri_cache + ijp*6;
                double aij = rjri[4];
                double ai = rjri[5] * .5;
                double aj_aij = 1. - ai / aij;
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
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
                    double cicj = rjri[3];
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
                    double aij = rjri[4];
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
                        double Rpa = rjri[n] * aj_aij;
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
#pragma unroll
                    for (int n = 0; n < GWIDTH_IP1; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
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
                        int addrx = (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double ai2 = rjri[5];
                        double fx = ai2 * gx[addrx+nsq_per_block];
                        double fy = ai2 * gy[addry+nsq_per_block];
                        double fz = ai2 * gz[addrz+nsq_per_block];
                        if (ix > 0) { fx -= ix * gx[addrx-nsq_per_block]; }
                        if (iy > 0) { fy -= iy * gy[addry-nsq_per_block]; }
                        if (iz > 0) { fz -= iz * gz[addrz-nsq_per_block]; }
                        goutx[n] += fx * gy[addry] * gz[addrz];
                        gouty[n] += fy * gx[addrx] * gz[addrz];
                        goutz[n] += fz * gx[addrx] * gy[addry];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
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

__global__
void rys_jk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                       ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_jk_ip1_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__
static void rys_ejk_ip1_general(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
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
                    dd += jk.j_factor * (dm[_ji] + dmb[_ji]) * (dm[_lk] + dmb[_lk]);
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
                double theta_kl = ak * al_akl;
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
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
                        double prod_xy = Ix * Iy * dd;
                        double prod_xz = Ix * Iz * dd;
                        double prod_yz = Iy * Iz * dd;
                        double gix = gx[addrx+i_1];
                        double giy = gy[addry+i_1];
                        double giz = gz[addrz+i_1];
                        double gkx = gx[addrx+k_1];
                        double gky = gy[addry+k_1];
                        double gkz = gz[addrz+k_1];
                        double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                        double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gy[addry-i_1]; } v_iy += fiy * prod_xz;
                        double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gz[addrz-i_1]; } v_iz += fiz * prod_xy;
                        double fkx = ak2 * gkx; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; } v_kx += fkx * prod_yz;
                        double fky = ak2 * gky; if (ky > 0) { fky -= ky * gy[addry-k_1]; } v_ky += fky * prod_xz;
                        double fkz = ak2 * gkz; if (kz > 0) { fkz -= kz * gz[addrz-k_1]; } v_kz += fkz * prod_xy;
                        double fjx = aj2 * (gix - rjri[0*nsq_per_block] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                        double fjy = aj2 * (giy - rjri[1*nsq_per_block] * Iy); if (jy > 0) { fjy -= jy * gy[addry-j_1]; } v_jy += fjy * prod_xz;
                        double fjz = aj2 * (giz - rjri[2*nsq_per_block] * Iz); if (jz > 0) { fjz -= jz * gz[addrz-j_1]; } v_jz += fjz * prod_xy;
                        double flx = al2 * (gkx - rlrk[0*nsq_per_block] * Ix); if (lx > 0) { flx -= lx * gx[addrx-l_1]; } v_lx += flx * prod_yz;
                        double fly = al2 * (gky - rlrk[1*nsq_per_block] * Iy); if (ly > 0) { fly -= ly * gy[addry-l_1]; } v_ly += fly * prod_xz;
                        double flz = al2 * (gkz - rlrk[2*nsq_per_block] * Iz); if (lz > 0) { flz -= lz * gz[addrz-l_1]; } v_lz += flz * prod_xy;
                    }
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        int t_id = gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        double *reduce = rw_cache + sq_id;
        __syncthreads();
        reduce[t_id+0 *threads] = v_ix;
        reduce[t_id+1 *threads] = v_iy;
        reduce[t_id+2 *threads] = v_iz;
        reduce[t_id+3 *threads] = v_jx;
        reduce[t_id+4 *threads] = v_jy;
        reduce[t_id+5 *threads] = v_jz;
        reduce[t_id+6 *threads] = v_kx;
        reduce[t_id+7 *threads] = v_ky;
        reduce[t_id+8 *threads] = v_kz;
        reduce[t_id+9 *threads] = v_lx;
        reduce[t_id+10*threads] = v_ly;
        reduce[t_id+11*threads] = v_lz;
        for (int i = gout_stride/2; i > 0; i >>= 1) {
            __syncthreads();
            if (gout_id < i) {
#pragma unroll
                for (int n = 0; n < 12; ++n) {
                    reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                }
            }
        }
        if (gout_id == 0 && task_id < ntasks) {
            double *ejk = jk.ejk;
            atomicAdd(ejk+ia*3+0, reduce[0 *threads]);
            atomicAdd(ejk+ia*3+1, reduce[1 *threads]);
            atomicAdd(ejk+ia*3+2, reduce[2 *threads]);
            atomicAdd(ejk+ja*3+0, reduce[3 *threads]);
            atomicAdd(ejk+ja*3+1, reduce[4 *threads]);
            atomicAdd(ejk+ja*3+2, reduce[5 *threads]);
            atomicAdd(ejk+ka*3+0, reduce[6 *threads]);
            atomicAdd(ejk+ka*3+1, reduce[7 *threads]);
            atomicAdd(ejk+ka*3+2, reduce[8 *threads]);
            atomicAdd(ejk+la*3+0, reduce[9 *threads]);
            atomicAdd(ejk+la*3+1, reduce[10*threads]);
            atomicAdd(ejk+la*3+2, reduce[11*threads]);
        }
    }
}

__global__
void rys_ejk_ip1_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                        ShellQuartet *pool, double *dd_pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;

    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int nf = nfij * nfkl;
    double *dd_cache = dd_pool + b_id * nf * blockDim.x;

    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
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
            rys_ejk_ip1_general(envs, jk, bounds, shl_quartet_idx,
                                dd_cache, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
